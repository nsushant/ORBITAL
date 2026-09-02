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

