"""
greedy_init.py — Python port of Julia's make_init_schedule / get_best_next
                 (sol_utils.jl lines 235–347) plus a pymoo Sampling wrapper.

Public API
----------
make_greedy_schedule(d, min_tof_table, name_to_idx, depot_id, nvehicles, ...)
    -> (tours, unserved_set)

encode_greedy_to_flat(tours, unserved, N, deadlines)
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
# Encode greedy schedule → flat pymoo array (4-group layout)
# ---------------------------------------------------------------------------

def encode_greedy_to_flat(tours, unserved, N, deadlines):
    """
    Convert a greedy schedule into the flat array used by OOSProblem.

    Layout: x[0:4N] = [visit_order | vehicle_assignments | arrivals | depot_visits]

    Unserved demands get vehicle_assignment=0, arrival=0, depot_visit=0.
    """
    x = np.zeros(4 * N)

    visit_order_seg = x[0:N]
    veh_assign_seg  = x[N:2*N]
    arrival_seg     = x[2*N:3*N]
    depot_vis_seg   = x[3*N:4*N]

    for v_idx, tour in enumerate(tours):
        v = v_idx + 1
        demand_steps = [s for s in tour if s['type'] == 'demand']
        if not demand_steps:
            continue

        K = len(demand_steps)

        for k, step in enumerate(demand_steps):
            pos = step['pos']
            veh_assign_seg[pos]  = v
            visit_order_seg[pos] = k / K
            arrival_seg[pos]     = step['arrival']

        # Mark depot visits: a demand has depot_visit=1 if the next tour step is a depot
        for step_idx, step in enumerate(tour):
            if step['type'] != 'demand':
                continue
            pos = step['pos']
            next_steps = tour[step_idx + 1:]
            if next_steps and next_steps[0]['type'] == 'depot':
                depot_vis_seg[pos] = 1

        # Circular-wrap: last demand always has depot_visit=1 (forced in repair anyway)
        last_demand = demand_steps[-1]
        depot_vis_seg[last_demand['pos']] = 1

    return x


# ---------------------------------------------------------------------------
# Load greedy solution from Julia JSON → flat 4-group encoding
# ---------------------------------------------------------------------------

def load_greedy_from_json(path, N):
    """
    Read the JSON written by Julia's save_schedule_json and convert to the
    flat 4-group NSGA-III encoding (same output shape as encode_greedy_to_flat).

    Julia visitedUID convention:
      uid > 0  → demand index (1-based) → Python pos = uid - 1
      uid < 0  → depot visit marker (ignored for encoding)
    """
    import json
    with open(path) as f:
        data = json.load(f)

    x = np.zeros(4 * N)
    visit_order_seg = x[0:N]
    veh_assign_seg  = x[N:2*N]
    arrival_seg     = x[2*N:3*N]
    depot_vis_seg   = x[3*N:4*N]

    for v_idx, veh in enumerate(data["schedule"]):
        v        = v_idx + 1
        uids     = veh["visitedUID"]
        arrivals = veh["arrivals"]

        demand_positions = [
            (i, uid - 1, arrivals[i])
            for i, uid in enumerate(uids) if uid > 0
        ]
        K = len(demand_positions)
        if K == 0:
            continue

        for k, (tour_i, pos, arr) in enumerate(demand_positions):
            veh_assign_seg[pos]  = v
            visit_order_seg[pos] = k / K
            arrival_seg[pos]     = arr

            # depot_visit = 1 if the next tour entry after this demand is a
            # depot marker (uid < 0), or if this is the last demand in the tour
            if k == K - 1:
                depot_vis_seg[pos] = 1
            else:
                next_tour_i = demand_positions[k + 1][0]
                if any(uids[j] < 0 for j in range(tour_i + 1, next_tour_i)):
                    depot_vis_seg[pos] = 1

    unassigned = data.get("unassigned")
    if unassigned and unassigned.get("UIDs"):
        for uid in unassigned["UIDs"]:
            pos = uid - 1
            if 0 <= pos < N:
                veh_assign_seg[pos] = 0
                arrival_seg[pos]    = 0.0
                depot_vis_seg[pos]  = 0

    return x


# ---------------------------------------------------------------------------
# pymoo Sampling wrapper
# ---------------------------------------------------------------------------

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
