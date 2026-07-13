"""
greedy_init.py — Python port of Julia's make_init_schedule / get_best_next
                 (sol_utils.jl lines 235–347) plus a pymoo Sampling wrapper.

Public API
----------
make_greedy_schedule(d, min_tof_table, name_to_idx, depot_id, nvehicles, ...)
    -> (tours, unserved_set)

encode_greedy_to_flat(tours, unserved, N, deadlines, max_deadline)
    -> ndarray shape (4*N,)

GreedySampling(greedy_x)  — pymoo Sampling subclass
"""

import numpy as np
from pymoo.core.sampling import Sampling


# ---------------------------------------------------------------------------
# Greedy next-demand selection
# ---------------------------------------------------------------------------

def _get_best_next(current_idx, current_time, unrouted,
                   sat_idx_for_demand, service_times, deadlines, min_tof_table):
    """
    Return (best_demand_pos, best_dv) with minimum ΔV among feasible candidates.
    Returns (None, inf) if no feasible candidate exists.
    """
    best_pos = None
    best_dv  = np.inf

    for pos in unrouted:
        key = (current_idx, sat_idx_for_demand[pos])
        if key not in min_tof_table:
            continue
        tof, dv = min_tof_table[key]
        if dv >= best_dv:
            continue
        if current_time + tof + service_times[pos] > deadlines[pos]:
            continue
        best_pos = pos
        best_dv  = dv

    return best_pos, best_dv


# ---------------------------------------------------------------------------
# Greedy schedule construction
# ---------------------------------------------------------------------------

def make_greedy_schedule(d, min_tof_table, name_to_idx, depot_id,
                         nvehicles=5, dv_budget=5000.0, refuel_time=0.5):
    """
    Port of sol_utils.jl make_init_schedule.

    Returns
    -------
    tours : list[list[dict]]
        One list per vehicle. Each entry is a dict:
          {'type': 'demand', 'pos': int, 'arrival': float, 'departure': float}
        or
          {'type': 'depot', 'arrival': float, 'departure': float}

    unserved : set[int]
        0-based demand positions that were not assigned to any vehicle.
    """
    sat_ids_by_name = [name_to_idx[s] for s in d["sat_identifiers"]]
    service_times   = d["service_times"]
    deadlines       = d["demand_deadlines"]
    N               = len(deadlines)

    unrouted = set(range(N))
    tours    = []

    for v in range(nvehicles):
        if not unrouted:
            tours.append([])
            continue

        current_idx  = depot_id
        current_time = 0.0
        leg_dv       = 0.0
        tour         = []

        while True:
            best_pos, best_dv = _get_best_next(
                current_idx, current_time, unrouted,
                sat_ids_by_name, service_times, deadlines, min_tof_table
            )

            if best_pos is None:
                break

            if leg_dv + best_dv > dv_budget:
                if current_idx == depot_id:
                    unrouted.discard(best_pos)
                    continue
                dep_key = (current_idx, depot_id)
                tof_dep, dv_dep = min_tof_table.get(dep_key, (0.0, 0.0))
                arr_depot = current_time + tof_dep
                dep_depot = arr_depot + refuel_time
                tour.append({'type': 'depot', 'arrival': arr_depot, 'departure': dep_depot})
                current_time = dep_depot
                current_idx  = depot_id
                leg_dv       = 0.0
                continue

            tof, _ = min_tof_table[(current_idx, sat_ids_by_name[best_pos])]
            arrival   = current_time + tof
            departure = arrival + service_times[best_pos]

            tour.append({
                'type':      'demand',
                'pos':       best_pos,
                'arrival':   arrival,
                'departure': departure,
            })

            current_idx  = sat_ids_by_name[best_pos]
            current_time = departure
            leg_dv      += best_dv
            unrouted.discard(best_pos)

        tours.append(tour)

    return tours, unrouted


# ---------------------------------------------------------------------------
# Encode greedy schedule → flat pymoo array (4N layout)
# ---------------------------------------------------------------------------

def encode_greedy_to_flat(tours, unserved, N, deadlines, max_deadline):
    """
    Convert a greedy schedule into the flat array used by OOSProblem (4N encoding).

    Layout:
      x[0:N]    vehicle_assignments   [0, maxV]
      x[N:2N]   arrivals_norm         [0, 1]   (arrival / deadline)
      x[2N:3N]  depot_visits          [0, 1]
      x[3N:4N]  depot_arrivals_norm   [0, 1]   (depot_arrival / max_deadline)

    Unserved demands get vehicle_assignment=0, arrivals_norm=0, depot_visit=0,
    depot_arrivals_norm=0.

    # OLD 5N layout (revert by restoring 5*N and old segment names):
    # x = np.zeros(5 * N)
    # visit_order_seg  = x[0:N]       # visit_order_seg[pos] = k / K
    # veh_assign_seg   = x[N:2*N]
    # arrival_seg      = x[2*N:3*N]   # absolute arrival times
    # depot_vis_seg    = x[3*N:4*N]
    # depot_arr_seg    = x[4*N:5*N]   # absolute depot arrival times
    """
    x = np.zeros(4 * N)

    veh_assign_seg   = x[0:N]
    arrival_norm_seg = x[N:2*N]
    depot_vis_seg    = x[2*N:3*N]
    depot_arr_seg    = x[3*N:4*N]

    deadlines = np.asarray(deadlines, dtype=float)

    for v_idx, tour in enumerate(tours):
        v = v_idx + 1
        demand_steps = [s for s in tour if s['type'] == 'demand']
        if not demand_steps:
            continue

        for step in demand_steps:
            pos = step['pos']
            veh_assign_seg[pos]   = v
            # normalise arrival by per-demand deadline
            arrival_norm_seg[pos] = step['arrival'] / deadlines[pos] if deadlines[pos] > 0 else 0.0

        # Mark depot visits and capture normalised depot arrival times
        for step_idx, step in enumerate(tour):
            if step['type'] != 'demand':
                continue
            pos = step['pos']
            next_steps = tour[step_idx + 1:]
            if next_steps and next_steps[0]['type'] == 'depot':
                depot_vis_seg[pos] = 1
                depot_arr_seg[pos] = next_steps[0]['arrival'] / max_deadline if max_deadline > 0 else 0.0

        # Circular-wrap: last demand always has depot_visit=1
        last_demand = demand_steps[-1]
        depot_vis_seg[last_demand['pos']] = 1

    return x


# ---------------------------------------------------------------------------
# Load greedy solution from Julia JSON → flat 4N encoding
# ---------------------------------------------------------------------------

def load_greedy_from_json(path, N):
    """
    Read the JSON written by Julia's save_schedule_json and convert to the
    flat 4N OOSProblem encoding.

    Julia visitedUID convention:
      uid > 0  → demand index (1-based) → Python pos = uid - 1
      uid < 0  → depot visit marker (ignored for encoding)

    # OLD 5N layout (revert by restoring 5*N and old segment names):
    # x = np.zeros(5 * N)
    # visit_order_seg  = x[0:N]       # visit_order_seg[pos] = k / K
    # veh_assign_seg   = x[N:2*N]
    # arrival_seg      = x[2*N:3*N]   # absolute arrival times
    # depot_vis_seg    = x[3*N:4*N]
    # depot_arr_seg    = x[4*N:5*N]   # absolute depot arrivals
    """
    import json
    with open(path) as f:
        data = json.load(f)

    x = np.zeros(4 * N)
    veh_assign_seg   = x[0:N]
    arrival_norm_seg = x[N:2*N]
    depot_vis_seg    = x[2*N:3*N]
    depot_arr_seg    = x[3*N:4*N]

    # Collect all absolute arrival times to compute max_deadline for normalisation
    all_arrivals = []
    for veh in data["schedule"]:
        all_arrivals.extend(veh["arrivals"])
    max_arr = max(all_arrivals) if all_arrivals else 1.0
    # Use 2× max observed arrival as max_deadline proxy (matches OOSProblem convention)
    max_deadline = 2.0 * max_arr if max_arr > 0 else 1.0

    for v_idx, veh in enumerate(data["schedule"]):
        v        = v_idx + 1
        uids     = veh["visitedUID"]
        arrivals = veh["arrivals"]

        # Collect per-vehicle demand deadline to normalise arrival times
        demand_positions = [
            (i, uid - 1, arrivals[i])
            for i, uid in enumerate(uids) if uid > 0
        ]
        K = len(demand_positions)
        if K == 0:
            continue

        for k, (tour_i, pos, arr) in enumerate(demand_positions):
            veh_assign_seg[pos]   = v
            # Normalise: we don't have per-demand deadlines here, use arr/max_deadline
            # as a conservative proxy (arr <= deadline always, so norm <= 1)
            arrival_norm_seg[pos] = arr / max_deadline if max_deadline > 0 else 0.0

            if k == K - 1:
                depot_vis_seg[pos] = 1
            else:
                next_tour_i = demand_positions[k + 1][0]
                depot_indices = [j for j in range(tour_i + 1, next_tour_i) if uids[j] < 0]
                if depot_indices:
                    depot_vis_seg[pos] = 1
                    depot_arr_seg[pos] = arrivals[depot_indices[0]] / max_deadline

    unassigned = data.get("unassigned")
    if unassigned and unassigned.get("UIDs"):
        for uid in unassigned["UIDs"]:
            pos = uid - 1
            if 0 <= pos < N:
                veh_assign_seg[pos]   = 0
                arrival_norm_seg[pos] = 0.0
                depot_vis_seg[pos]    = 0

    return x


# ---------------------------------------------------------------------------
# pymoo Sampling wrapper
# ---------------------------------------------------------------------------

def load_greedy_RK_from_json(path, N):
    """
    Read Julia greedy JSON and convert to the flat N random-keys encoding
    used by OOSProblemRK.

    x[i] = vehicle + (k + 0.5) / K   for demand i on vehicle v at position k of K
    x[i] = 0.5                         for unserved demands (floor=0 → unserved)
    """
    import json
    with open(path) as f:
        data = json.load(f)

    x = np.full(N, 0.5)   # default: unserved (floor=0)

    for v_idx, veh in enumerate(data["schedule"]):
        v    = v_idx + 1
        uids = veh["visitedUID"]
        demand_positions = [uid - 1 for uid in uids if uid > 0]
        K = len(demand_positions)
        if K == 0:
            continue
        for k, pos in enumerate(demand_positions):
            if 0 <= pos < N:
                x[pos] = v + (k + 0.5) / K   # vehicle + fractional order key

    unassigned = data.get("unassigned")
    if unassigned and unassigned.get("UIDs"):
        for uid in unassigned["UIDs"]:
            pos = uid - 1
            if 0 <= pos < N:
                x[pos] = 0.5   # unserved

    return x


class GreedySamplingRK(Sampling):
    """
    Seed the initial population with the greedy solution (N random-keys encoding)
    as the first individual; fill the rest with uniform random samples.
    """

    def __init__(self, greedy_x):
        super().__init__()
        self.greedy_x = greedy_x   # shape (N,)

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.uniform(
            problem.xl, problem.xu,
            (n_samples, problem.n_var)
        )
        X[0] = np.clip(self.greedy_x, problem.xl, problem.xu)
        return X


class GreedySamplingRK2(Sampling):
    """
    Seed population for the 2N RK encoding (OOSProblemRK_OT).
    First N genes: greedy RK solution.
    Second N genes: depot positions inferred from a budget simulation on the
    greedy chromosome — so the seed already encodes where refuels are needed.
    Rest of population: random demand genes, depot genes all inactive (frac=0).
    """

    def __init__(self, greedy_x):
        super().__init__()
        self.greedy_x = greedy_x   # shape (N,)

    def _do(self, problem, n_samples, **kwargs):
        N = problem.Ndems

        # Random population: depot genes all inactive (floor part = 0 since frac < 1
        # and vehicle assignments ≥ 1, so keeping values in [0,1) makes them inactive)
        X = np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
        X[:, N:] = np.random.uniform(0.0, 1.0, (n_samples, N))

        # Build greedy seed
        seed = np.zeros(2 * N)
        seed[:N] = np.clip(self.greedy_x, problem.xl[:N], problem.xu[:N])

        vehicle_d = np.floor(seed[:N]).astype(int)
        order_d   = seed[:N] - vehicle_d

        depot_slot = 0  # next free depot gene slot in seed[N:]

        for v in range(1, problem.maxV + 1):
            dem_idx = np.where(vehicle_d == v)[0]
            if len(dem_idx) == 0:
                continue
            dem_sorted = dem_idx[np.argsort(order_d[dem_idx])]

            state_sat = problem.depot_id
            state_dep = 0.0
            cum_dv    = 0.0
            prev_key  = None

            for i in dem_sorted:
                sid     = int(problem.sat_ids[i])
                key_dir = (state_sat, sid)
                tof_dir = problem.min_tof_table[key_dir][0] if key_dir in problem.min_tof_table else 0.0
                dep_s   = float(problem.dep_grid[np.searchsorted(problem.dep_grid, state_dep)
                                                  .clip(0, len(problem.dep_grid) - 1)])
                arr     = None
                dv_dir  = np.inf
                for ai in range(int(np.searchsorted(problem.arr_grid, state_dep + tof_dir)
                                    .clip(0, len(problem.arr_grid) - 1)), len(problem.arr_grid)):
                    a_cand = float(problem.arr_grid[ai])
                    if a_cand + problem.service_times[i] > problem.deadlines[i]:
                        break
                    c = problem.cost_table.get((state_sat, sid, dep_s, a_cand), np.inf)
                    if not np.isinf(c) and c <= problem.dv_budget:
                        arr    = a_cand
                        dv_dir = c
                        break

                if arr is None:
                    continue

                new_cum = dv_dir if state_sat == problem.depot_id else cum_dv + dv_dir

                if new_cum > problem.dv_budget and state_sat != problem.depot_id:
                    # Record a depot gene between the previous demand key and this one
                    curr_key   = order_d[i]
                    depot_key  = (prev_key + curr_key) / 2.0 if prev_key is not None else curr_key * 0.5
                    if depot_slot < N:
                        seed[N + depot_slot] = v + depot_key
                        depot_slot += 1

                    # Approximate depot visit: reset state
                    state_sat = problem.depot_id
                    state_dep = state_dep + problem.refuel_time + 1.0
                    cum_dv    = 0.0

                    # Re-attempt this demand from depot
                    key_fr  = (problem.depot_id, sid)
                    tof_fr  = problem.min_tof_table[key_fr][0] if key_fr in problem.min_tof_table else 0.0
                    dep_s2  = float(problem.dep_grid[np.searchsorted(problem.dep_grid, state_dep)
                                                      .clip(0, len(problem.dep_grid) - 1)])
                    arr     = None
                    dv_dir  = np.inf
                    for ai in range(int(np.searchsorted(problem.arr_grid, state_dep + tof_fr)
                                        .clip(0, len(problem.arr_grid) - 1)), len(problem.arr_grid)):
                        a_cand = float(problem.arr_grid[ai])
                        if a_cand + problem.service_times[i] > problem.deadlines[i]:
                            break
                        c = problem.cost_table.get((problem.depot_id, sid, dep_s2, a_cand), np.inf)
                        if not np.isinf(c) and c <= problem.dv_budget:
                            arr    = a_cand
                            dv_dir = c
                            break

                    if arr is None:
                        continue

                state_sat = sid
                state_dep = arr + problem.service_times[i]
                cum_dv    = dv_dir if state_sat == problem.depot_id else cum_dv + dv_dir
                prev_key  = order_d[i]

        X[0] = seed
        return X


def load_greedy_2N_from_json(path, N):
    """
    Read Julia greedy JSON and convert to the flat 2N OOSProblemDecoder encoding.

    Layout:
      x[0:N]   visit_order keys   float [0, 1]   (order within vehicle / K)
      x[N:2N]  vehicle_assignments int  [0, maxV]

    Unserved demands get vehicle_assignment=0, visit_key=0.
    """
    import json
    with open(path) as f:
        data = json.load(f)

    x = np.zeros(2 * N)
    visit_key_seg  = x[0:N]
    veh_assign_seg = x[N:2*N]

    for v_idx, veh in enumerate(data["schedule"]):
        v = v_idx + 1
        uids = veh["visitedUID"]
        demand_positions = [uid - 1 for uid in uids if uid > 0]
        K = len(demand_positions)
        if K == 0:
            continue
        for k, pos in enumerate(demand_positions):
            if 0 <= pos < N:
                veh_assign_seg[pos] = v
                visit_key_seg[pos]  = k / K   # [0, 1) order keys

    unassigned = data.get("unassigned")
    if unassigned and unassigned.get("UIDs"):
        for uid in unassigned["UIDs"]:
            pos = uid - 1
            if 0 <= pos < N:
                veh_assign_seg[pos] = 0
                visit_key_seg[pos]  = 0.0

    return x


class GreedySamplingDecoder(Sampling):
    """
    Seed the initial population with the greedy solution (2N encoding) as the
    first individual; fill the rest with uniform random samples within bounds.
    """

    def __init__(self, greedy_x):
        super().__init__()
        self.greedy_x = greedy_x   # shape (2*N,)

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.uniform(
            problem.xl, problem.xu,
            (n_samples, problem.n_var)
        )
        X[0] = np.clip(self.greedy_x, problem.xl, problem.xu)
        return X


class GreedySampling(Sampling):
    """
    Seed the initial population with the greedy solution as the first individual;
    fill the rest with uniform random samples within bounds.
    """

    def __init__(self, greedy_x):
        super().__init__()
        self.greedy_x = greedy_x   # shape (4*N,)

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.uniform(
            problem.xl, problem.xu,
            (n_samples, problem.n_var)
        )
        X[0] = np.clip(self.greedy_x, problem.xl, problem.xu)
        return X
