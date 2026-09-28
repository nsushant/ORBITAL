"""The constructive heuristic: a feasible schedule to start from (Sec. 3.6).

MDLS is a local search, so it needs somewhere to start, and the GA wants the
same schedule as a warm start -- a population seeded from a feasible point
rather than from uniform noise. On a real S2 demand set a random genome flies
773 visits of which 760 have no cost-table entry, so "somewhere to start" is
not a nicety here.

The algorithm is the paper's: nearest neighbour on delta-V over minimum-time
transfers. Each vehicle leaves the depot, repeatedly takes whichever unrouted
demand is cheapest to reach next, inserts a refuelling visit whenever the next
leg would break the budget, and stops when nothing else can be reached in time.
Then the next vehicle starts, until every demand is placed or the fleet runs
out. Whatever is left over is unassigned, which costs f3 and violates nothing
(D19).

Ported from `make_init_schedule` / `get_best_next` in sol_utils.jl. Three
things are deliberately different, each because the Julia's version would be
wrong here rather than merely different:

  * Legs are priced through `oos.schedule.leg()`, the same function the
    evaluator and the GA decoder use, so the arrival times this writes are the
    arrival times `evaluate()` will read back. The Julia used a separate
    `MinTOFTable` of per-pair (min_tof, dv), which is what let its schedules
    and its objective disagree about when a vehicle arrives (F7).

  * The Julia closed every route with `get(MinTOFTable, key, (0.0, 0.0))` --
    a missing depot-return leg silently became free and instantaneous. Here a
    route is closed only if the closing leg actually exists and fits the
    budget; otherwise it is left open. An open route costs nothing and breaks
    nothing, whereas a fabricated free leg quietly understates f1.

  * Waiting is a departure-epoch search rather than an arrival-time fudge.
    The Julia set `arr = max(current_time + tof, available[uid])`, moving the
    arrival without moving the transfer that produced it. Since the cost table
    is indexed by departure epoch, waiting here means letting the vehicle idle
    and depart at a later grid epoch, which is a real manoeuvre with a real
    price.

Feasibility is the property that matters and the one Sec. 3.6 claims: every
demand this places arrives inside its own release/deadline window on a leg the
table actually has, and no vehicle exceeds the budget between depot visits.
`tests/test_greedy.py` asserts exactly that through `is_feasible()` rather than
trusting the construction.
"""

from __future__ import annotations

import numpy as np
from .compat import njit

from ..schedule import DEPOT_UID, Schedule, snap_departure


@njit(cache=True)
def _best_leg(dv_tab, ph_tab, dep_grid, tof_grid, i, j, p, depart_day,
              release, deadline, service, budget_left):
    """Cheapest leg i->j leaving at grid epoch p that lands inside the window.

    Every time of flight is considered, not just the shortest. That matters
    because it is how waiting is expressed here: the decoder always departs the
    moment a vehicle comes free, so the only way to reach a demand released
    later is to fly longer, not to leave later. Searching q is therefore not an
    embellishment -- it is what keeps this heuristic inside the language the
    Sec. 3.8 encoding can actually speak.

    This is a deliberate departure from `get_best_next` in sol_utils.jl, which
    set `arr = max(current_time + tof, available)`: it moved the arrival
    without moving the transfer that produced it, so its schedules recorded
    arrivals no leg in them achieved.
    """
    # The *earliest* feasible time of flight, not the cheapest -- Sec. 3.6
    # specifies "the lowest delta-V minimum time transfer", so the search over
    # q is for feasibility (a later release may need a longer flight) and the
    # comparison between candidates is on the cost of their earliest legs.
    #
    # This is not a detail. Taking the cheapest q instead makes the initial
    # schedule cost-minimal at every step, which leaves the timing operator of
    # Sec. 3.7 with nothing to improve: measured, it fired on 0 of 40
    # applications and gained 0 m/s. The heuristic is meant to hand MDLS a
    # feasible but fuel-inefficient schedule and let the local search reclaim
    # the difference; a greedy that already minimises f1 quietly removes one of
    # the three search directions the method is built on.
    for q in range(tof_grid.shape[0]):
        dv = dv_tab[i, j, p, q]
        if np.isnan(dv) or dv > budget_left:
            continue
        arr = dep_grid[p] + tof_grid[q] + ph_tab[i, j, p, q]
        # No landing before the vehicle set out. snap_departure clamps past the
        # end of the grid, so without this a vehicle whose clock has run past
        # the last departure epoch picks up legs that arrive in its own past.
        if arr < depart_day:
            continue
        if arr < release:
            continue
        if arr + service > deadline:
            continue
        return dv, arr, q
    return np.inf, 0.0, -1


@njit(cache=True)
def _earliest_return(dv_tab, ph_tab, dep_grid, tof_grid, i, depot, p,
                     depart_day, budget_left):
    """Earliest usable leg back to the depot, which has no window to respect."""
    for q in range(tof_grid.shape[0]):
        dv = dv_tab[i, depot, p, q]
        if np.isnan(dv) or dv > budget_left:
            continue
        arr = dep_grid[p] + tof_grid[q] + ph_tab[i, depot, p, q]
        if arr < depart_day:
            continue
        return dv, arr, q
    return np.inf, 0.0, -1


@njit(cache=True)
def _best_next(dv_tab, ph_tab, dep_grid, tof_grid, node_of_dem,
               release, deadline, service, routed, cur_node, cur_time,
               budget_left):
    """The cheapest reachable unrouted demand from `cur_node`, or -1."""
    p = snap_departure(dep_grid, cur_time)
    best_uid = -1
    best_dv = np.inf
    best_arr = 0.0
    best_q = -1
    for uid in range(node_of_dem.shape[0]):
        if routed[uid]:
            continue
        dv, arr, q = _best_leg(dv_tab, ph_tab, dep_grid, tof_grid,
                               cur_node, node_of_dem[uid], p, cur_time,
                               release[uid], deadline[uid], service[uid],
                               budget_left)
        if q >= 0 and dv < best_dv:
            best_dv = dv
            best_uid = uid
            best_arr = arr
            best_q = q
    return best_uid, best_dv, best_arr, best_q


def greedy_schedule(cost_table, demands, depot_node, max_vehicles,
                    refuel_time, start_time=0.0, dv_budget=None):
    """Build a feasible schedule, nearest neighbour on delta-V (Sec. 3.6).

    Returns (schedule, unassigned) where `unassigned` is the array of demand
    ids nobody could reach in time. Those are a legitimate outcome that costs
    f3, not a failure -- on this instance most of the population is out of
    reach of the depot at any epoch, which is the depot study's subject.
    """
    dv_tab, ph_tab = cost_table.dv, cost_table.phasing
    dep_grid, tof_grid = cost_table.dep_s, cost_table.tof_s
    budget = float(cost_table.dv_budget if dv_budget is None else dv_budget)

    n_dem = len(demands)
    routed = np.zeros(n_dem, dtype=np.bool_)

    veh_start = [0]
    node_l, uid_l, arr_l, dep_l, cost_l = [], [], [], [], []

    for _v in range(max_vehicles):
        if routed.all():
            break

        cur_node = int(depot_node)
        cur_time = float(start_time)
        leg_spend = 0.0
        placed_any = False
        first = len(node_l)

        while True:
            uid, dv, arr, _q = _best_next(
                dv_tab, ph_tab, dep_grid, tof_grid, demands.node,
                demands.release, demands.deadline, demands.service,
                routed, cur_node, cur_time, budget - leg_spend)

            if uid < 0:
                # Nothing reachable within what is left of this run's budget.
                # A refuelling stop restores it, so it is worth one try -- but
                # only if the vehicle is away from the depot and can get back.
                if cur_node == depot_node or leg_spend == 0.0:
                    break
                d_dv, d_arr, d_q = _earliest_return(
                    dv_tab, ph_tab, dep_grid, tof_grid, cur_node, int(depot_node),
                    snap_departure(dep_grid, cur_time), cur_time,
                    budget - leg_spend)
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
            placed_any = True
            cur_node = int(demands.node[uid])
            cur_time = arr + float(demands.service[uid])
            leg_spend += dv

        # Close the route at the depot when the leg exists and is affordable.
        # The Julia defaulted a missing one to zero cost and zero time, which
        # understates f1; an open route is the honest alternative.
        if placed_any and cur_node != depot_node:
            d_dv, d_arr, d_q = _earliest_return(
                dv_tab, ph_tab, dep_grid, tof_grid, cur_node, int(depot_node),
                snap_departure(dep_grid, cur_time), cur_time, budget - leg_spend)
            if d_q >= 0:
                node_l.append(int(depot_node)); uid_l.append(DEPOT_UID)
                arr_l.append(d_arr); dep_l.append(d_arr + refuel_time)
                cost_l.append(d_dv)

        if not placed_any:
            # This vehicle reached nothing, so no later one will either: they
            # all start from the same depot at the same epoch.
            del node_l[first:], uid_l[first:], arr_l[first:], dep_l[first:], cost_l[first:]
            break
        veh_start.append(len(node_l))

    schedule = Schedule(
        veh_start=np.array(veh_start, dtype=np.int64),
        node=np.array(node_l, dtype=np.int64),
        uid=np.array(uid_l, dtype=np.int64),
        arrival=np.array(arr_l, dtype=np.float64),
        departure=np.array(dep_l, dtype=np.float64),
        cost=np.array(cost_l, dtype=np.float64),
    )
    return schedule, np.flatnonzero(~routed)


def encode_for_ga(schedule, demands, problem):
    """Turn a greedy schedule into a genome for `ScheduleProblem` (2N genes).

    The warm start the old `GreedySamplingRK2` could not provide: it wrote its
    fractional gene as a position in the visit order, while the decoder reads
    that same fraction as arrival / T_horizon. Seeding with it would have put
    order indices where arrival times belong -- wrong in a way nothing would
    have reported.

    Writing the schedule's own arrival times is not enough either, and that is
    worth spelling out because it is not obvious. The decoder does not read a
    gene as "arrive here"; it reads it as a *request*, derives a time of flight
    by subtracting where the vehicle already is, and rounds that up to the grid
    (D20). A greedy arrival already contains the departure epoch and the
    phasing time, so feeding it back in overshoots by exactly those, the
    request rounds up into the next cell, and the decoder prices a different
    leg -- 36 missing legs and 4.3 km/s of phantom overspend on a schedule that
    was feasible before the round trip (F22).

    So this walks the schedule forward carrying the decoder's state, and for
    each visit emits the request that lands on the cell the greedy chose:
    `state_dep + tof_grid[q]`, which snap_tof resolves to exactly q. The state
    then advances the way the decoder will advance it, so the two stay in step
    the whole way down the route.

    One thing the encoding cannot express: the greedy may idle and depart on a
    later epoch, but the decoder always departs when the vehicle came free, so
    a waited departure has no genome that reproduces it. Those legs are emitted
    as best they can be and re-priced by the decoder. That is a limit of the
    Sec. 3.8 encoding -- it carries a vehicle and an arrival per demand, and
    nothing that says "wait" -- not of this function.
    """
    from ..schedule import leg, snap_departure

    n = len(demands)
    ct = problem.ct
    dv_tab, ph_tab = ct.dv, ct.phasing
    dep_grid, tof_grid = ct.dep_s, ct.tof_s

    x = np.zeros(2 * n)
    depot_slot = 0

    for v in range(schedule.n_vehicles):
        lo, hi = schedule.veh_start[v], schedule.veh_start[v + 1]
        state_node = int(problem.depot_node)
        state_dep = 0.0

        for k in range(lo, hi):
            target = int(schedule.node[k])
            p = snap_departure(dep_grid, state_dep)

            # The greedy took the first time of flight the table has an entry
            # for; ask for exactly that so snap_tof resolves back to it.
            # The greedy's arrival came from a specific cell; recover which
            # one, then ask for state_dep + tof_grid[q] so snap_tof resolves
            # back to exactly it.
            q = -1
            for qq in range(tof_grid.shape[0]):
                if np.isnan(dv_tab[state_node, target, p, qq]):
                    continue
                arr_qq = (dep_grid[p] + tof_grid[qq]
                          + ph_tab[state_node, target, p, qq])
                if abs(arr_qq - float(schedule.arrival[k])) < 1e-6:
                    q = qq
                    break
            # Land just *inside* the intended cell, not exactly on its edge.
            # snap_tof returns the smallest grid point >= the request, and the
            # request survives a round trip through frac = request / T_horizon,
            # so an overshoot of one ULP at the boundary jumps a whole cell --
            # 450 days of flight becoming 480, and every visit after it drifting
            # with it. A microsecond of slack is nine orders above the float
            # error at these magnitudes and nine below anything the model
            # resolves.
            request = (state_dep + float(tof_grid[q]) - 1e-6 if q >= 0
                       else float(schedule.arrival[k]))

            # A request above T_horizon cannot be written down: the gene's
            # fraction saturates and the decoder prices a leg that does not
            # exist. That happens for a handful of visits at the very end of
            # the horizon, where the request (state_dep + tof) runs past the
            # ceiling even though the arrival does not. Truncate the route
            # there rather than emit a gene that decodes to a broken leg --
            # the remaining demands stay unassigned, which costs f3 and
            # violates nothing (D19). A seed that is feasible and serves a few
            # fewer clients is worth more than one that serves them and is not.
            if request > problem.T_horizon:
                break
            frac = min(max(request, 0.0) / problem.T_horizon, 1.0 - 1e-9)
            if schedule.uid[k] == DEPOT_UID:
                if depot_slot < n:
                    x[n + depot_slot] = (v + 1) + frac
                    depot_slot += 1
            else:
                x[int(schedule.uid[k])] = (v + 1) + frac

            # Advance exactly as the decoder will, so the next request is
            # measured from the same place the decoder will measure it.
            _dv, true_arr, ok = leg(dv_tab, ph_tab, dep_grid, tof_grid,
                                    state_node, target, state_dep,
                                    request - state_dep)
            arrival = true_arr if ok else request
            if schedule.uid[k] == DEPOT_UID:
                state_dep = arrival + float(problem.refuel_time)
            else:
                state_dep = arrival + float(demands.service[int(schedule.uid[k])])
            state_node = target

    # Unrouted demands keep gene 0.x: floor 0 is the dummy vehicle, unassigned.
    return np.clip(x, problem.xl, problem.xu)


def greedy_restarts(cost_table, demands, depot_node, max_vehicles, refuel_time,
                    n=18, dv_budget=None):
    """A spread of feasible starting schedules, capped at different fleet sizes.

    One starting schedule is not enough for either method, for the same reason
    in both cases. Seeded with a single feasible point in a population where
    nothing else is feasible, constrained domination makes every survivor a
    descendant of it and the population collapses: measured on S2, NSGA-II
    reached full feasibility by generation 40 but held only 1-7 distinct
    objective vectors, and never once reduced the fleet below the cap. Ten
    times the evaluation budget bought 15 distinct vectors and still no fleet
    reduction; eighteen distinct seeds bought 40 and a fleet range of 3-25.

    Varying the fleet cap is what makes the restarts genuinely different rather
    than perturbations of one schedule: a tighter cap changes which demands the
    heuristic can reach and in what order, so the tours differ in composition
    and not only in length.

    Both methods are initialised from this same set, so neither is handed a
    better start than the other. It does mean the fleet-size spread of the
    initial population is given rather than discovered, which the write-up
    should say plainly; what the comparison then measures is what each method
    does *from* that common start.
    """
    caps = np.unique(np.linspace(2, max_vehicles, n).astype(int))
    out = []
    for cap in caps:
        s, _ = greedy_schedule(cost_table, demands, depot_node, int(cap),
                               refuel_time, dv_budget=dv_budget)
        if len(s.node):
            out.append(s)
    return out
