"""Multi-Directional Local Search (Sec. 3.7), ported from algoMDLS.jl.

The idea, from Tricoire (2012): a point on the Pareto front has to beat every
other solution on at least one objective, so improving any single objective is
a way of reaching it. MDLS keeps an archive of non-dominated solutions, picks
one at random each iteration, runs a separate single-objective local search for
each objective, and offers all the results back to the archive. Nothing is
scalarised and no weights are chosen.

Three operators, one per objective, matching Table 4 of the paper:

    delta-V          shift a block of arrival times by one grid cell
    fleet size       remove the least-used vehicles
    unrecovered value  build a new tour out of unassigned demand

The two LNS operators in the Julia (`destroy_and_repair`, `shaw_removal_repair`)
are commented out there and are not ported. That is a deliberate concession:
they would give MDLS a large-neighbourhood machinery the genetic algorithm has
no counterpart to, and the comparison this paper makes is between directed
local search and population search, not between operator libraries.

Two departures from the Julia, both about not carrying numbers that are no
longer true:

  * Every leg a move touches is re-priced through `oos.schedule.leg()`.
    `opt_times_combined` re-priced only the leg it was targeting, even though
    shifting a block of arrivals changes the departure of every visit in it and
    therefore the cost of the legs after it too. Those stale costs went
    straight into f1.

  * The unassigned set is derived from the schedule rather than tracked beside
    it. The Julia threaded a parallel `unassigned` dictionary through every
    operator, which is a standing invitation for the two to disagree about who
    is served. Here `Schedule.served()` is the single source of truth.

A candidate enters the archive only if `evaluate()` says it is feasible, so an
operator that produces something unusable costs an evaluation and nothing else.
"""

from __future__ import annotations

import time

import numpy as np

from .archive import Archive, SearchArchive
from ..constants import DAY
from .greedy import (_best_leg, _best_next, _earliest_return,
                     greedy_restarts)
from ..schedule import DEPOT_UID, Schedule, evaluate, is_feasible, leg, snap_departure


# ---------------------------------------------------------------------------
# Routes: a per-vehicle view, for operators that add and remove whole vehicles
# ---------------------------------------------------------------------------


def _to_routes(s):
    """Schedule -> list of per-vehicle dicts of plain arrays."""
    out = []
    for v in range(s.n_vehicles):
        lo, hi = int(s.veh_start[v]), int(s.veh_start[v + 1])
        if hi <= lo:
            continue
        out.append(dict(node=list(s.node[lo:hi]), uid=list(s.uid[lo:hi]),
                        arrival=list(s.arrival[lo:hi]),
                        departure=list(s.departure[lo:hi]),
                        cost=list(s.cost[lo:hi])))
    return out


def _from_routes(routes):
    veh_start, node, uid, arr, dep, cost = [0], [], [], [], [], []
    for r in routes:
        node += list(r["node"]); uid += list(r["uid"])
        arr += list(r["arrival"]); dep += list(r["departure"]); cost += list(r["cost"])
        veh_start.append(len(node))
    return Schedule(veh_start=np.array(veh_start, dtype=np.int64),
                    node=np.array(node, dtype=np.int64),
                    uid=np.array(uid, dtype=np.int64),
                    arrival=np.array(arr, dtype=np.float64),
                    departure=np.array(dep, dtype=np.float64),
                    cost=np.array(cost, dtype=np.float64))


def unassigned_of(schedule, n_demands):
    """Which demands nobody serves. Derived, never tracked separately."""
    served = np.zeros(n_demands, dtype=bool)
    s = schedule.served()
    if len(s):
        served[s] = True
    return np.flatnonzero(~served)


def _reprice(route, ct, depot_node, demands, refuel_time):
    """Re-cost and re-validate a whole route from its first visit, in place.

    Returns False if the route no longer works: a leg the table has no entry
    for, or a client visit that has fallen outside its release/deadline
    window. Both have to be checked here rather than left to `evaluate()`,
    because a timing move is only worth taking if the result is usable, and a
    candidate that violates a window would otherwise be offered to the archive
    and rejected there having cost a full evaluation.

    The window check covers the whole route, not only the visits the move
    shifted. It has to: an arrival is what the table grants, not what the move
    asked for, so re-pricing can move visits the shift never touched.
    """
    state_node, state_dep = int(depot_node), 0.0
    for k in range(len(route["node"])):
        target = int(route["node"][k])
        t_arr = float(route["arrival"][k])
        dv, true_arr, ok = leg(ct.dv, ct.phasing, ct.dep_s, ct.tof_s,
                               state_node, target, state_dep, t_arr - state_dep)
        if not ok:
            return False
        uid = int(route["uid"][k])
        if uid == DEPOT_UID:
            dwell = refuel_time
        else:
            dwell = float(demands.service[uid])
            if true_arr < demands.release[uid] - 1e-9:
                return False
            if true_arr + dwell > demands.deadline[uid] + 1e-9:
                return False
        route["arrival"][k] = true_arr
        route["cost"][k] = dv
        route["departure"][k] = true_arr + dwell
        state_node, state_dep = target, true_arr + dwell
    return True


# ---------------------------------------------------------------------------
# Operator 1 -- delta-V: shift a block of arrivals by one grid cell
# ---------------------------------------------------------------------------


def op_shift_times(schedule, ct, demands, depot_node, refuel_time,
                   shift=15.0, top_pct=0.5):
    """Move the costliest legs' timing and keep whichever direction is cheaper.

    For each of the `top_pct` costliest legs, two block moves are tried: push
    this visit and everything after it later by `shift` days, or pull
    everything before it earlier by the same. Both preserve the route's
    internal spacing; what changes is the departure epoch the leg is priced
    at, which is what the cost depends on.
    """
    routes = _to_routes(schedule)
    if not routes:
        return schedule

    legs = [(float(r["cost"][k]), vi, k)
            for vi, r in enumerate(routes) for k in range(1, len(r["node"]))
            if np.isfinite(r["cost"][k])]
    if not legs:
        return schedule
    legs.sort(key=lambda t: -t[0])
    topn = max(1, int(round(len(legs) * top_pct)))

    for _cost, vi, k in legs[:topn]:
        base = routes[vi]
        best, best_total = None, sum(c for c in base["cost"] if np.isfinite(c))

        for direction in ("later", "earlier"):
            cand = {key: list(val) for key, val in base.items()}
            if direction == "later":
                for j in range(k, len(cand["node"])):
                    cand["arrival"][j] += shift
            else:
                if any(cand["arrival"][j] - shift < 0.0 for j in range(k)):
                    continue
                for j in range(k):
                    cand["arrival"][j] -= shift
            if not _reprice(cand, ct, depot_node, demands, refuel_time):
                continue
            total = sum(cand["cost"])
            if total < best_total - 1e-9:
                best, best_total = cand, total

        if best is not None:
            routes[vi] = best

    return _from_routes(routes)


# ---------------------------------------------------------------------------
# Operator 1b -- delta-V: re-order a route by nodal geometry
# ---------------------------------------------------------------------------


def raan_model(cost_table, nodes):
    """(raan0, raan_rate) re-indexed onto the cost table's node numbering.

    The ephemeris and the cost table are built from the same instance and at
    present share an index order, but they are separate files: rebuilding one
    without the other would silently attribute every orbit to the wrong
    spacecraft, and nothing downstream would look wrong. Aligning by name costs
    one dictionary and removes the possibility.
    """
    pos = {n: i for i, n in enumerate(nodes.names)}
    missing = [n for n in cost_table.names if n not in pos]
    if missing:
        raise SystemExit(
            f"{len(missing)} node(s) in the cost table are absent from the "
            f"ephemeris (first: {missing[0]}); the two were built from "
            "different instances")
    idx = np.array([pos[n] for n in cost_table.names], dtype=np.int64)
    return nodes.raan0[idx].copy(), nodes.raan_rate[idx].copy()


def _wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def op_raan_resequence(schedule, ct, demands, depot_node, refuel_time, raan):
    """Re-order the costliest vehicle's clients by nearest nodal separation.

    Ported from `raan_walk_resequence` in algoMDLS.jl, which is defined there
    and never called (F37). It fills a hole the other operators leave: none of
    them changes visit order. `op_shift_times` moves a block of arrivals and
    says so -- "Both preserve the route's internal spacing" -- while remove and
    create work on whole vehicles. The GA, by contrast, derives visit order
    from the sort order of its arrival genes, so a single mutation re-sequences
    a route. Sec. 4 was therefore comparing a method that can re-order against
    one that cannot, in MDLS's disfavour (F39).

    Why nodal separation rather than distance or cost. A transfer here is
    priced almost entirely by the plane change it needs, and the plane change
    between two orbits is set by how far apart their nodes are at the moment of
    flight -- which moves, because each orbit's node regresses at its own rate.
    So the cheap next client is the one whose node will be nearest this one's
    when the vehicle would arrive, and that is what the walk selects on. The
    Julia recomputed the drift from osculating elements with the analytic
    secular formula; this uses `Nodes`, whose rate is fitted by least squares
    to the unwrapped node history on mean elements, because the analytic route
    imports the J2 short-period signature -- about 11 m/s of jitter on each
    velocity plane (see oos/nodes.py).

    Two deliberate departures from the Julia. It rebuilt a vehicle by
    concatenating the re-ordered clients and then appending every depot visit
    at the end, which silently moves refuelling stops to the end of the route:
    that breaks the chronology and, worse, the delta-V budget between depot
    visits that g3 exists to enforce (D19), since a segment that was two
    depot-to-depot runs becomes one. Here the route is split at its depot
    visits and each segment's clients are re-ordered within it, so the depot
    structure is preserved exactly. Second, the arrival written for each visit
    is the time-of-flight *request* `state_dep + tof_grid[q]`, not the arrival
    the probe computed: the request is what `_reprice` snaps, and recording an
    arrival that includes phasing would make it snap to a different cell than
    the one chosen here (the same convention, and the same 1e-6 margin against
    the snapping boundary, as `encode_for_ga`; F25).

    Returns the schedule unchanged unless the whole vehicle re-sequences and
    the result is strictly cheaper -- a local search on f1, like the operator
    it sits beside in the pool.
    """
    raan0, raan_rate = raan
    routes = _to_routes(schedule)
    if not routes:
        return schedule

    totals = [float(np.nansum(r["cost"])) for r in routes]
    vi = int(np.argmax(totals))
    base = routes[vi]
    if sum(1 for u in base["uid"] if int(u) != DEPOT_UID) < 2:
        return schedule                      # nothing to permute

    dvt, pht, dg, tg = ct.dv, ct.phasing, ct.dep_s, ct.tof_s

    def raan_at(node, t_days):
        return raan0[node] + raan_rate[node] * (t_days * DAY)

    # Split at depot visits: each segment is the clients flown between two
    # refuellings, and only their order is in play.
    segments, current = [], []
    for u, nd in zip(base["uid"], base["node"]):
        if int(u) == DEPOT_UID:
            segments.append((current, int(nd)))
            current = []
        else:
            current.append((int(u), int(nd)))
    if current:
        segments.append((current, None))

    new_uid, new_node, new_arrival = [], [], []
    state_node, state_dep = int(depot_node), 0.0   # where _reprice starts too

    for clients, seg_depot in segments:
        remaining = list(clients)
        while remaining:
            p = snap_departure(dg, state_dep)
            here = raan_at(state_node, state_dep)
            best = None
            for uid, nd in remaining:
                _dv, arr, q = _best_leg(dvt, pht, dg, tg, state_node, nd, p,
                                        state_dep, demands.release[uid],
                                        demands.deadline[uid],
                                        demands.service[uid], np.inf)
                if q < 0:
                    continue
                sep = abs(_wrap_pi(raan_at(nd, arr) - here))
                if best is None or sep < best[0]:
                    best = (sep, uid, nd, q, arr)
            if best is None:
                return schedule              # cannot place them all; abandon
            _sep, uid, nd, q, arr = best
            new_uid.append(uid)
            new_node.append(nd)
            new_arrival.append(state_dep + float(tg[q]) - 1e-6)
            remaining = [c for c in remaining if c[0] != uid]
            state_node = nd
            state_dep = arr + float(demands.service[uid])

        if seg_depot is not None:
            p = snap_departure(dg, state_dep)
            _dv, arr, q = _earliest_return(dvt, pht, dg, tg, state_node,
                                           seg_depot, p, state_dep, np.inf)
            if q < 0:
                return schedule
            new_uid.append(DEPOT_UID)
            new_node.append(seg_depot)
            new_arrival.append(state_dep + float(tg[q]) - 1e-6)
            state_node = seg_depot
            state_dep = arr + refuel_time

    cand = dict(node=new_node, uid=new_uid, arrival=new_arrival,
                departure=[0.0] * len(new_uid), cost=[0.0] * len(new_uid))
    if not _reprice(cand, ct, depot_node, demands, refuel_time):
        return schedule
    if float(np.sum(cand["cost"])) >= totals[vi] - 1e-9:
        return schedule                      # no improvement; keep the original
    routes[vi] = cand
    return _from_routes(routes)


def _rechain(route, ct, demands, depot_node, refuel_time):
    """Re-time a route onto earliest-feasible legs, then price it.

    The fallback when a move leaves a route's existing arrival requests
    unusable -- a swapped-in client whose window the old request misses. Only
    the timing is rebuilt; the visit order is untouched.
    """
    dvt, pht, dg, tg = ct.dv, ct.phasing, ct.dep_s, ct.tof_s
    nu, nn, na = [], [], []
    state_node, state_dep = int(depot_node), 0.0
    for u, n in zip(route["uid"], route["node"]):
        u, n = int(u), int(n)
        p = snap_departure(dg, state_dep)
        if u == DEPOT_UID:
            _dv, arr, q = _earliest_return(dvt, pht, dg, tg, state_node, n, p,
                                           state_dep, np.inf)
            dwell = refuel_time
        else:
            _dv, arr, q = _best_leg(dvt, pht, dg, tg, state_node, n, p,
                                    state_dep, demands.release[u],
                                    demands.deadline[u], demands.service[u],
                                    np.inf)
            dwell = float(demands.service[u])
        if q < 0:
            return None
        nu.append(u)
        nn.append(n)
        na.append(state_dep + float(tg[q]) - 1e-6)     # a tof request, per F25
        state_node, state_dep = n, arr + dwell
    cand = dict(node=nn, uid=nu, arrival=na,
                departure=[0.0] * len(nu), cost=[0.0] * len(nu))
    return cand if _reprice(cand, ct, depot_node, demands, refuel_time) else None


def _price_after_move(route, ct, demands, depot_node, refuel_time):
    """Keep the route's timing if it still works, re-chain it if it does not."""
    cand = {k: list(v) for k, v in route.items()}
    if _reprice(cand, ct, depot_node, demands, refuel_time):
        return cand
    return _rechain(route, ct, demands, depot_node, refuel_time)


def op_swap_cross_vehicle(schedule, ct, demands, depot_node, refuel_time,
                          top_n=10, max_pairs=4000, rng=None):
    """Exchange two clients between two vehicles, if it costs less delta-V.

    The gap this fills. Ordering the clients *within* a vehicle is a
    one-dimensional problem -- the only epoch-dependent part of a transfer's
    cost is the nodal separation, so a route is a sorted walk in nodal phase
    and the Sec. 3.6 heuristic already solves it. Measured, no re-ordering rule
    improves a single archive route, exhaustive enumeration included (F40).
    *Partitioning* clients across vehicles is the combinatorial half, and there
    the heuristic is crude: it fills one vehicle until it cannot continue, then
    starts the next, which is first-fit. Nothing in the operator pool moved a
    client between two vehicles, so that partition stood as built.

    There is a great deal in it. On one 25-vehicle schedule, 199 of 900
    feasible swaps reduce delta-V and the best saves 2,889 m/s in one move
    (F41).

    Ranking. Evaluating all 900 costs 0.2 s, which at 3,300 iterations is
    hours, so candidates are scored cheaply and only `top_n` are priced in
    full. The score is the cost of the two *incoming* legs under the swap:
    `prev1 -> client2` leaving when `prev1` already leaves, and `prev2 ->
    client1` likewise. Those two departure epochs are fixed by construction --
    a predecessor departs when it departs, whoever follows it -- so the score
    varies only the target and needs two table lookups.

    That it works is measured, not assumed, and the obvious alternative does
    not. Over all 900 feasible swaps this score correlates 0.76 with the true
    gain and its top 10 contains the best swap in the neighbourhood; the slack
    ranking of `swap_cross_vehicle` in algoMDLS.jl correlates -0.04 and its top
    50 reaches only -1,105 m/s, worse than picking 10 at random. Ranking on
    what a move costs beats ranking on how much room it has.

    Timing is preserved where it survives and rebuilt only where it does not,
    so a swap does not quietly undo `op_shift_times`. The comparison is against
    the incumbent's own cost, so an accepted swap is an improvement on what is
    in the archive rather than on a re-timed version of it.
    """
    routes = _to_routes(schedule)
    if len(routes) < 2:
        return schedule
    dvt, pht, dg, tg = ct.dv, ct.phasing, ct.dep_s, ct.tof_s

    # every client position, with the predecessor it is reached from
    slots = []
    for vi, r in enumerate(routes):
        for p, u in enumerate(r["uid"]):
            if int(u) == DEPOT_UID:
                continue
            prev_dep = float(r["departure"][p - 1]) if p else 0.0
            slots.append((vi, p, int(u), int(r["node"][p]),
                          int(r["node"][p - 1]) if p else int(depot_node),
                          prev_dep, float(r["arrival"][p]) - prev_dep))
    if len(slots) < 2:
        return schedule

    def incoming(node, uid, prev_node, prev_dep, request):
        """Cost of reaching `node` from `prev_node` on the incumbent's own
        timing request, and whether it lands inside the client's window.

        Pricing at the incumbent's request rather than at the earliest feasible
        time of flight is what keeps the score honest. A route that is up
        against the delta-V budget deliberately flies later, cheaper legs, so
        re-timing it onto earliest-feasible transfers makes it far more
        expensive -- by up to 1,646 m/s on a single leg here. Scoring in one
        regime and pricing the move in another produced a ranking that
        promised 2,019 m/s and delivered nothing (F41).
        """
        dv, arr, ok = leg(dvt, pht, dg, tg, prev_node, node, prev_dep, request)
        if not ok:
            return np.inf
        if arr < demands.release[uid] - 1e-9:
            return np.inf
        if arr + demands.service[uid] > demands.deadline[uid] + 1e-9:
            return np.inf
        return float(dv)

    # The incumbent's own incoming legs, on its own requests: this reproduces
    # route["cost"] exactly, so a score is a like-for-like difference.
    base_in = [incoming(n, u, pn, pd, req)
               for _vi, _p, u, n, pn, pd, req in slots]

    pairs = [(i, j) for i in range(len(slots)) for j in range(i + 1, len(slots))
             if slots[i][0] != slots[j][0]]
    if not pairs:
        return schedule
    if max_pairs and len(pairs) > max_pairs:
        draw = (rng if rng is not None else np.random.default_rng(0))
        pairs = [pairs[k] for k in draw.choice(len(pairs), max_pairs,
                                               replace=False)]

    scored = []
    for i, j in pairs:
        if not (np.isfinite(base_in[i]) and np.isfinite(base_in[j])):
            continue
        _vi, _pi, ui, ni, pni, pdi, reqi = slots[i]
        _vj, _pj, uj, nj, pnj, pdj, reqj = slots[j]
        a = incoming(nj, uj, pni, pdi, reqi)
        if not np.isfinite(a):
            continue
        b = incoming(ni, ui, pnj, pdj, reqj)
        if not np.isfinite(b):
            continue
        scored.append((a + b - base_in[i] - base_in[j], i, j))
    if not scored:
        return schedule
    scored.sort(key=lambda t: t[0])

    base_cost = [float(np.nansum(r["cost"])) for r in routes]
    best, best_gain = None, 1e-9
    for _score, i, j in scored[:top_n]:
        vi, pi, ui, ni = slots[i][:4]
        vj, pj, uj, nj = slots[j][:4]
        r1 = {k: list(v) for k, v in routes[vi].items()}
        r2 = {k: list(v) for k, v in routes[vj].items()}
        r1["uid"][pi], r1["node"][pi] = uj, nj
        r2["uid"][pj], r2["node"][pj] = ui, ni
        if not _reprice(r1, ct, depot_node, demands, refuel_time):
            continue
        if not _reprice(r2, ct, depot_node, demands, refuel_time):
            continue
        c1, c2 = r1, r2
        gain = (base_cost[vi] + base_cost[vj]
                - float(np.sum(c1["cost"])) - float(np.sum(c2["cost"])))
        if gain > best_gain:
            best, best_gain = (vi, vj, c1, c2), gain

    if best is None:
        return schedule
    vi, vj, c1, c2 = best
    routes[vi], routes[vj] = c1, c2
    return _from_routes(routes)


# ---------------------------------------------------------------------------
# Shared repair: 2-regret insertion (Sec. 3.7, Algorithm 2)
# ---------------------------------------------------------------------------


def _tof_requests(route):
    """Each visit's requested time of flight, measured from its predecessor's
    departure. A route is really a chain of (node, duration): absolute arrivals
    are what the table grants, not what the schedule asks for, so inserting a
    visit and leaving the later absolute arrivals in place would silently
    shorten every request downstream of it (F22 in a different disguise)."""
    reqs, state_dep = [], 0.0
    for k in range(len(route["node"])):
        reqs.append(float(route["arrival"][k]) - state_dep)
        state_dep = float(route["departure"][k])
    return reqs


def _price_chain(items, ct, depot_node, demands, refuel_time, dv_budget):
    """Price a chain of (node, uid, tof_request) from the depot at day 0.

    Returns a route dict, or None if any leg is missing from the table, any
    client visit falls outside its window, or the delta-V spent between two
    depot visits exceeds the budget. That last check is g3 (D19), which
    `_reprice` leaves to `evaluate`; here it has to be enforced in the operator,
    because an insertion that breaks the budget makes the whole schedule
    infeasible and the archive would reject it after a full evaluation had been
    spent.
    """
    node_l, uid_l, arr_l, dep_l, cost_l = [], [], [], [], []
    state_node, state_dep, spend = int(depot_node), 0.0, 0.0
    for node, uid, req in items:
        node, uid = int(node), int(uid)
        dv, arr, ok = leg(ct.dv, ct.phasing, ct.dep_s, ct.tof_s,
                          state_node, node, state_dep, req)
        if not ok:
            return None
        if uid == DEPOT_UID:
            dwell = refuel_time
            spend = 0.0                      # refuelled: the budget resets
        else:
            dwell = float(demands.service[uid])
            if arr < demands.release[uid] - 1e-9:
                return None
            if arr + dwell > demands.deadline[uid] + 1e-9:
                return None
            spend += dv
            if spend > dv_budget + 1e-6:
                return None
        node_l.append(node); uid_l.append(uid)
        arr_l.append(arr); dep_l.append(arr + dwell); cost_l.append(dv)
        state_node, state_dep = node, arr + dwell
    return dict(node=node_l, uid=uid_l, arrival=arr_l, departure=dep_l,
                cost=cost_l)


def _insertion_cost(route, k, uid, ct, demands, depot_node, refuel_time,
                    dv_budget):
    """Cost of putting demand `uid` before position `k` of `route`, or None.

    Returns (delta, priced_route): the change in the route's total delta-V, and
    the re-priced route itself so the caller does not have to redo the work.

    algoMDLS.jl scored this as two new legs less the one they replace, which is
    O(1), but only for insertions that fit inside the route's existing slack so
    that the next visit keeps its arrival. On these schedules that admits
    nothing at all. The constructive heuristic departs the moment a vehicle is
    free and flies the earliest feasible transfer, so each visit arrives at the
    earliest time it possibly could and there is no slack anywhere: measured,
    the slack-preserving rule inserted 0 of 302 unassigned demands. Any real
    insertion has to push the rest of the tour later, so the tour is re-priced
    in full and the window and budget checks are applied to all of it. At four
    to six visits a tour that is about seven table lookups, against two for the
    cheap version that cannot fire (F45).
    """
    dvt, pht, dg, tg = ct.dv, ct.phasing, ct.dep_s, ct.tof_s
    node = int(demands.node[uid])

    if k == 0:
        prev_node, prev_dep = int(depot_node), 0.0
    else:
        prev_node = int(route["node"][k - 1])
        prev_dep = float(route["departure"][k - 1])

    # cheapest screen first: is this pair connected at all, at any epoch. One
    # byte, and it answers "no" for 86 % of candidates on the study instance.
    if not ct.reach[prev_node, node]:
        return None
    # then: is there a leg from *here* that lands inside this demand's window
    _dv, _arr, q = _best_leg(dvt, pht, dg, tg, prev_node, node,
                             snap_departure(dg, prev_dep), prev_dep,
                             demands.release[uid], demands.deadline[uid],
                             demands.service[uid], np.inf)
    if q < 0:
        return None

    items = list(zip(route["node"], route["uid"], _tof_requests(route)))
    items.insert(k, (node, int(uid), float(tg[q]) - 1e-6))
    priced = _price_chain(items, ct, depot_node, demands, refuel_time,
                          dv_budget)
    if priced is None:
        return None
    old = float(np.nansum(route["cost"]))
    return float(np.sum(priced["cost"])) - old, priced


def regret2_insert(routes, unassigned, ct, demands, depot_node, refuel_time,
                   dv_budget, max_candidates=8, rng=None):
    """Insert what nobody serves, hardest-to-place first. Returns (routes, left).

    Algorithm 2 of Sec. 3.7. For each unassigned demand the best and
    second-best insertions over all routes and positions are found; the demand
    whose regret --- the gap between them --- is largest is placed first, on
    the argument that a demand with one good position and no other will lose it
    if something else is placed there, while a demand with many equally good
    positions can wait.

    This is the operator the method was missing. Removing a vehicle freed its
    demands to the unassigned set and nothing ever put them back into an
    existing tour; only `op_create_vehicle` could recover them, and only by
    starting a new vehicle. Nothing could move a demand into a tour that
    already passed near it, so the partition the constructive heuristic built
    stood essentially unchanged for the whole search (F45).
    """
    left = [int(u) for u in unassigned]
    if not routes or not left:
        return routes, left

    # The unassigned set is most of the instance -- 302 of 400 on a full-fleet
    # seed -- and scoring every one of them against every position on every
    # route, once per placement, is quadratic in something large. A sample is
    # taken instead, which is what makes this a repair operator rather than a
    # search: the operator runs thousands of times over a race, and a different
    # sample each time covers the set without any single call paying for it.
    if rng is not None and len(left) > max_candidates:
        left = [left[i] for i in rng.choice(len(left), max_candidates,
                                            replace=False)]
    elif len(left) > max_candidates:
        left = left[:max_candidates]

    placed = []
    while left:
        best = None                      # (regret, uid, vi, priced route)
        for uid in left:
            b1 = b2 = np.inf
            b1_at = None
            for vi, r in enumerate(routes):
                for k in range(len(r["node"]) + 1):
                    got = _insertion_cost(r, k, uid, ct, demands, depot_node,
                                          refuel_time, dv_budget)
                    if got is None:
                        continue
                    delta, priced = got
                    if delta < b1:
                        b2, b1, b1_at = b1, delta, (vi, priced)
                    elif delta < b2:
                        b2 = delta
            if b1_at is None:
                continue
            regret = (b1 if b2 == np.inf else b2) - b1
            if best is None or regret > best[0]:
                best = (regret, uid, b1_at)
        if best is None:
            break                        # nothing left can be placed anywhere

        _reg, uid, (vi, priced) = best
        routes[vi] = priced
        left.remove(uid)
        placed.append(uid)

    return routes, left


# ---------------------------------------------------------------------------
# Operator 2 -- fleet size: remove the least-used vehicles
# ---------------------------------------------------------------------------


def op_remove_vehicles(schedule, k_remove, ct=None, demands=None,
                       depot_node=None, refuel_time=0.5, dv_budget=None):
    """Drop the `k_remove` vehicles serving the fewest clients, then repair.

    The freed demands are offered back to the surviving vehicles by 2-regret
    insertion (Algorithm 2, Sec. 3.7); whatever will not fit anywhere stays
    unassigned and costs f3. This is what `tab:local_search_operators` means by
    redistributing them, and until the repair existed the table was wrong: the
    demands went to the dummy vehicle and only a later `op_create_vehicle`
    could recover them, and only by starting a new tour (F38, F45).

    Passing the table and demands is optional so the bare removal is still
    available, mainly for tests that want to isolate one effect. At least one
    vehicle is always kept.
    """
    routes = _to_routes(schedule)
    if len(routes) <= 1:
        return schedule
    k = min(int(k_remove), len(routes) - 1)
    if k < 1:
        return schedule
    counts = [sum(1 for u in r["uid"] if u != DEPOT_UID) for r in routes]
    victims = set(np.argsort(counts, kind="stable")[:k].tolist())
    freed = [int(u) for i, r in enumerate(routes) if i in victims
             for u in r["uid"] if int(u) != DEPOT_UID]
    kept = [r for i, r in enumerate(routes) if i not in victims]

    if ct is not None and demands is not None and depot_node is not None:
        budget = float(ct.dv_budget if dv_budget is None else dv_budget)
        # everything the schedule already failed to serve is offered too, since
        # removing a vehicle may free the slack that was blocking it
        already = unassigned_of(schedule, len(demands)).tolist()
        kept, _left = regret2_insert(kept, freed + already, ct, demands,
                                     int(depot_node), refuel_time, budget)
    return _from_routes(kept)


# ---------------------------------------------------------------------------
# Operator 3 -- unrecovered value: build a tour from what nobody serves
# ---------------------------------------------------------------------------


def op_create_vehicle(schedule, ct, demands, depot_node, refuel_time,
                      max_vehicles, dv_budget):
    """Add one vehicle serving demands currently assigned to nobody.

    Same nearest-neighbour construction as Sec. 3.6, run over the unassigned
    set only, which is what `create_vehicle` in the Julia does. Returns the
    schedule unchanged when there is nothing to add or no vehicle to spare.
    """
    routes = _to_routes(schedule)
    if len(routes) >= max_vehicles:
        return schedule
    free = unassigned_of(schedule, len(demands))
    if len(free) == 0:
        return schedule

    routed = np.ones(len(demands), dtype=np.bool_)
    routed[free] = False

    node_l, uid_l, arr_l, dep_l, cost_l = [], [], [], [], []
    cur_node, cur_time, leg_spend = int(depot_node), 0.0, 0.0

    while True:
        uid, dv, arr, _q = _best_next(
            ct.dv, ct.phasing, ct.dep_s, ct.tof_s, demands.node,
            demands.release, demands.deadline, demands.service,
            routed, cur_node, cur_time, dv_budget - leg_spend)

        if uid < 0:
            if cur_node == depot_node or leg_spend == 0.0:
                break
            d_dv, d_arr, d_q = _earliest_return(
                ct.dv, ct.phasing, ct.dep_s, ct.tof_s, cur_node,
                int(depot_node), snap_departure(ct.dep_s, cur_time),
                cur_time, dv_budget - leg_spend)
            if d_q < 0:
                break
            node_l.append(int(depot_node)); uid_l.append(DEPOT_UID)
            arr_l.append(d_arr); dep_l.append(d_arr + refuel_time)
            cost_l.append(d_dv)
            cur_node, cur_time, leg_spend = int(depot_node), d_arr + refuel_time, 0.0
            continue

        node_l.append(int(demands.node[uid])); uid_l.append(int(uid))
        arr_l.append(arr); dep_l.append(arr + float(demands.service[uid]))
        cost_l.append(dv)
        routed[uid] = True
        cur_node = int(demands.node[uid])
        cur_time = arr + float(demands.service[uid])
        leg_spend += dv

    if not node_l:
        return schedule

    routes.append(dict(node=node_l, uid=uid_l, arrival=arr_l,
                       departure=dep_l, cost=cost_l))

    # No repair sweep here. The new tour is already built greedily from the
    # unassigned set, so a 2-regret pass over the remainder is largely the same
    # question asked a second time, and the fleet operator's repair covers the
    # case that matters -- demands freed by a removal finding a home in a tour
    # that survives. Measured, the sweep roughly doubled the cost of an
    # iteration and changed no front.
    return _from_routes(routes)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def mdls(cost_table, demands, depot_node, max_vehicles=25, refuel_time=0.5,
         max_iter=1000, seed=0, shift=15.0, top_pct=0.5, dv_budget=None,
         init=None, time_limit=None, verbose=False, raan=None,
         swap=False, swap_top_n=50, swap_max_pairs=None,
         remove_lo=0.20, remove_hi=0.90, add_lo=0.10, add_hi=1.00,
         return_search_archive=False, search_archive_capacity=4096):
    """Run MDLS. Returns (archive, n_evaluations).

    One iteration is one draw from the archive and one application of each
    objective's operator, so it costs three full evaluations -- the unit the
    equal-budget comparison against the GA is counted in.

    `swap` adds `op_swap_cross_vehicle` to that pool. Off by default, on the
    evidence: it finds real per-route savings (up to 265 m/s on one seed) but
    over six paired runs it left the front's extremes unchanged in every one,
    for two to three times the wall clock (F41). One flag turns it on.

    `raan`, when given, is `raan_model(cost_table, nodes)` and adds
    `op_raan_resequence` to the delta-V pool. One of the two is sampled per
    iteration, which is the structure algoMDLS.jl already uses for that
    objective (`dv_idx = rand(1:length(dv_operators))`) and keeps the cost at
    three evaluations an iteration. Evaluating both and keeping the better
    would cost four and would require `run_mdls_trial.py`'s budget conversion
    to divide by four, or MDLS would quietly be given a third more work than
    the GA under an equal-budget heading (D32).
    """
    rng = np.random.default_rng(seed)
    budget = float(cost_table.dv_budget if dv_budget is None else dv_budget)
    n_dem = len(demands)
    t0 = time.time()

    # The same spread of starting schedules the GA is seeded with, so neither
    # method begins from a better position than the other. A single start is
    # not enough for either: it fixes the fleet size the search departs from,
    # and both methods then have to discover the whole f2 range from one point.
    if init is None:
        init = greedy_restarts(cost_table, demands, depot_node, max_vehicles,
                               refuel_time, dv_budget=budget)
    elif isinstance(init, Schedule):
        init = [init]

    archive = Archive(n_obj=3)
    search_archive = SearchArchive(capacity=search_archive_capacity)
    n_eval = 0
    for s in init:
        f, g = evaluate(s, demands, budget)
        n_eval += 1
        search_archive.add(f, g, s, unassigned_of(s, n_dem),
                           iteration=-1, operator="initial")
        if is_feasible(g):
            archive.add(f, s, unassigned_of(s, n_dem))
    if len(archive) == 0:
        raise SystemExit(
            "no initial schedule is feasible, so there is nothing to search "
            "from. Sec. 3.6 requires the constructive heuristic to guarantee "
            "feasibility; check tests/test_greedy.py.")

    for it in range(max_iter):
        if time_limit is not None and time.time() - t0 > time_limit:
            break
        x = archive.schedules[archive.sample(rng)]
        n_veh = max(1, x.n_vehicles)

        # fleet: remove Uniform[remove_lo, remove_hi] of the vehicles. The
        # defaults are algoMDLS.jl's [0.20, 0.90]; Sec. 3.7 said [0.10, 0.90],
        # which no implementation ever did, and now says [0.20, 0.90] (F38).
        lo, hi = remove_lo, max(remove_hi, remove_lo)
        k_remove = max(1, int(round((lo + rng.random() * (hi - lo)) * n_veh)))
        # coverage: up to Uniform[add_lo, add_hi] more vehicles, cap allowing
        alo, ahi = add_lo, max(add_hi, add_lo)
        k_add = max(1, min(int(round((alo + rng.random() * (ahi - alo)) * n_veh)),
                           max_vehicles - n_veh))

        # delta-V: one operator sampled from the pool, so the objective still
        # costs one evaluation an iteration however many operators serve it
        pool = [0]
        if swap:
            pool.append(1)
        if raan is not None:
            pool.append(2)
        pick = pool[int(rng.integers(len(pool)))]
        if pick == 1:
            dv_candidate = op_swap_cross_vehicle(
                x, cost_table, demands, depot_node, refuel_time,
                top_n=swap_top_n, max_pairs=swap_max_pairs, rng=rng)
        elif pick == 2:
            dv_candidate = op_raan_resequence(x, cost_table, demands,
                                              depot_node, refuel_time, raan)
        else:
            dv_candidate = op_shift_times(x, cost_table, demands, depot_node,
                                          refuel_time, shift, top_pct)

        candidates = [op_remove_vehicles(x, k_remove, cost_table, demands,
                                         depot_node, refuel_time, budget),
                      dv_candidate]
        operators = ["remove_vehicles", "delta_v"]

        if len(unassigned_of(x, n_dem)) and max_vehicles > n_veh:
            s = x
            for _ in range(k_add):
                s2 = op_create_vehicle(s, cost_table, demands, depot_node,
                                       refuel_time, max_vehicles, budget)
                if s2 is s:
                    break
                s = s2
            candidates.append(s)
            operators.append("create_vehicle")

        for cand, operator in zip(candidates, operators):
            f, g = evaluate(cand, demands, budget)
            n_eval += 1
            search_archive.add(f, g, cand, unassigned_of(cand, n_dem),
                               iteration=it, operator=operator)
            if is_feasible(g):
                archive.add(f, cand, unassigned_of(cand, n_dem))

        if verbose and (it + 1) % 100 == 0:
            fr = archive.objectives
            print(f"  iter {it+1:5d}  archive {len(archive):4d}  "
                  f"dV {fr[:,0].min():8.0f}-{fr[:,0].max():8.0f}  "
                  f"veh {fr[:,1].min():.0f}-{fr[:,1].max():.0f}  "
                  f"unrec ${fr[:,2].min()/1e6:.1f}-{fr[:,2].max()/1e6:.1f} M")

    if return_search_archive:
        return archive, n_eval, search_archive
    return archive, n_eval
