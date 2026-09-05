

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
        dv, arr, ok = leg(ct.dv, ct.phasing, ct.dep_days, ct.tof_days,
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


def _insertion_delta(route, k, uid, ct, demands, depot_node):
    """Cost of putting demand `uid` before position `k` of `route`, or None.

    Scored the way algoMDLS.jl scores it, and for the same reason: the
    insertion must fit inside the slack the route already has, so the visit
    after it keeps the arrival it had and nothing downstream moves. That makes
    the delta exactly the two new legs less the one they replace --- two table
    lookups rather than a re-pricing of the whole tail --- which is what lets
    every unassigned demand be scored against every position on every route.
    An insertion that would push the next visit later is refused rather than
    cascaded.
    """
    dvt, pht, dg, tg = ct.dv, ct.phasing, ct.dep_days, ct.tof_days
    node = int(demands.node[uid])

    if k == 0:
        prev_node, prev_dep = int(depot_node), 0.0
    else:
        prev_node = int(route["node"][k - 1])
        prev_dep = float(route["departure"][k - 1])

    dv_in, arr, q = _best_leg(dvt, pht, dg, tg, prev_node, node,
                              snap_departure(dg, prev_dep), prev_dep,
                              demands.release[uid], demands.deadline[uid],
                              demands.service[uid], np.inf)
    if q < 0:
        return None
    dep_r = arr + float(demands.service[uid])
    req_in = float(tg[q]) - 1e-6

    if k >= len(route["node"]):                       # appended at the end
        return dv_in, req_in, None

    arr_next = float(route["arrival"][k])
    if dep_r > arr_next + 1e-9:
        return None                                   # would delay the next visit
    dv_out, true_next, ok = leg(dvt, pht, dg, tg, node,
                                int(route["node"][k]), dep_r,
                                arr_next - dep_r)
    if not ok or true_next > arr_next + 1e-9:
        return None                                   # the slack does not hold
    old = float(route["cost"][k])
    if not np.isfinite(old):
        return None
    return dv_in + dv_out - old, req_in, arr_next - dep_r


def regret2_insert(routes, unassigned, ct, demands, depot_node, refuel_time,
                   dv_budget):
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

    while left:
        best = None                      # (regret, uid, route, position, req)
        for uid in left:
            b1 = b2 = np.inf
            b1_at = None
            for vi, r in enumerate(routes):
                for k in range(len(r["node"]) + 1):
                    got = _insertion_delta(r, k, uid, ct, demands, depot_node)
                    if got is None:
                        continue
                    delta, req_in, _slack = got
                    if delta < b1:
                        b2, b1, b1_at = b1, delta, (vi, k, req_in)
                    elif delta < b2:
                        b2 = delta
            if b1_at is None:
                continue
            regret = (b1 if b2 == np.inf else b2) - b1
            if best is None or regret > best[0]:
                best = (regret, uid, b1_at)
        if best is None:
            break                        # nothing left can be placed anywhere

        _reg, uid, (vi, k, req_in) = best
        r = routes[vi]
        items = list(zip(r["node"], r["uid"], _tof_requests(r)))
        items.insert(k, (int(demands.node[uid]), int(uid), req_in))
        priced = _price_chain(items, ct, depot_node, demands, refuel_time,
                              dv_budget)
        if priced is None:
            # The slack-preserving score said this fits and the exact re-price
            # disagrees -- the budget between depot visits, most often. Drop
            # the demand rather than the position: retrying every other
            # position here would turn one repair into a search.
            left.remove(uid)
            continue
        routes[vi] = priced
        left.remove(uid)

    return routes, left
