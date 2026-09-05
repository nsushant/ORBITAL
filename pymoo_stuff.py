import numpy as np
from pymoo.core.problem import Problem
from loaders import snap_cost


class OOSProblem(Problem):
    """
    Flat-array encoding (n_var = 4 * Ndems):

      x[0:N]    vehicle_assignments   int   [0, maxV]   (rounded in repair/decode)
      x[N:2N]   arrivals_norm[i]      float [0, 1]      (arrival[i] = x[N+i] * deadline[i])
      x[2N:3N]  depot_visits[i]       int   [0, 1]      (rounded in repair/decode)
      x[3N:4N]  depot_arrivals_norm   float [0, 1]      (depot_arr[i] = x[3N+i] * max_deadline)

    Visit order within each vehicle is derived from sorted arrival times — no
    separate order variable needed.

    vehicle_assignments = 0  → demand is unserved.
    vehicle_assignments = 1..maxV → actual vehicle index.

    departure[i] = arrival[i] + service_times[i]  (no wait variable).

    n_constr = 0 -- not because a repair operator runs before evaluation
    (no such class exists in this file or anywhere in the tree; that claim
    was checked and withdrawn, see the F19 correction in
    PAPER_COMPLETION_PLAN.md), but because the RK/RK_OT decoders below build
    every schedule by construction: a leg is only ever flown if it exists in
    the cost table and respects the window and budget, so there is nothing
    for a constraint vector to report. An unassigned demand is not repaired
    into feasibility, it is simply left unassigned and costs f3 (D19).

    # ── OLD 5N ENCODING (revert by restoring these bounds and updating
    #    repair/decode/operators to match) ────────────────────────────────────
    # x[0:N]    visit_order           float [0, 1]
    # x[N:2N]   vehicle_assignments   int   [0, maxV]
    # x[2N:3N]  arrival[i]            float [0, deadline[i]]
    # x[3N:4N]  depot_visits[i]       int   [0, 1]
    # x[4N:5N]  depot_arrivals[i]     float [0, 2*max_deadline]
    # xl = np.zeros(5 * N)
    # xu = np.concatenate([
    #     np.ones(N),
    #     np.full(N, maxV),
    #     deadlines,
    #     np.ones(N),
    #     np.full(N, 2.0 * float(deadlines.max())),
    # ])
    # super().__init__(n_var=5*N, ...)
    """

    def __init__(self, Ndems, maxV, deadlines, service_times, sat_ids,
                 depot_id, cost_table, dep_grid, arr_grid, min_tof_table,
                 asset_values=None, refuel_time=0.5, dv_budget=5000.0):
        N = Ndems
        deadlines = np.asarray(deadlines, dtype=float)

        # 4N encoding — all normalised to [0, 1] except vehicle_assignments
        xl = np.zeros(4 * N)
        xu = np.concatenate([
            np.full(N, maxV),   # vehicle_assignments [0, maxV]
            np.ones(N),         # arrivals_norm       [0, 1]
            np.ones(N),         # depot_visits        [0, 1]
            np.ones(N),         # depot_arrivals_norm [0, 1]
        ])

        self.Ndems        = Ndems
        self.max_deadline = 2.0 * float(deadlines.max())
        self.maxV         = maxV
        self.deadlines    = deadlines
        self.service_times = np.asarray(service_times, dtype=float)
        self.asset_values  = (np.asarray(asset_values, dtype=float)
                              if asset_values is not None
                              else self.service_times)   # fallback: old behaviour
        self.sat_ids      = np.asarray(sat_ids, dtype=int)
        self.depot_id     = int(depot_id)
        self.cost_table   = cost_table
        self.dep_grid     = np.asarray(dep_grid, dtype=float)
        self.arr_grid     = np.asarray(arr_grid, dtype=float)
        self.min_tof_table = min_tof_table
        self.refuel_time  = float(refuel_time)
        self.dv_budget    = float(dv_budget)

        super().__init__(n_var=4*N, n_obj=3, n_constr=0, xl=xl, xu=xu)

# ── Random-keys problem (N encoding, no repair, standard SBX+PM operators) ───

class OOSProblemRK(OOSProblem):
    """
    NSGA-III variant using random-keys (BRKGA-style) encoding.

    Encoding (n_var = N):
      x[i] ∈ [0, maxV+1)
        integer part  floor(x[i]) = vehicle assignment (0=unserved, 1..maxV)
        fractional part x[i]-floor(x[i]) = position key within vehicle's tour

    Example: x[i]=2.35 → vehicle 2, ordered 35th percentile within that vehicle.

    Vehicle assignment and visit order are coupled in a single gene, so SBX
    preserving x[i]=2.35 keeps both properties together. Standard pymoo SBX
    and PM operators work directly — no custom operators needed.

    The decoder computes earliest feasible arrivals (same logic as
    OOSProblemDecoder) — no repair operator needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N  = self.Ndems
        xl = np.zeros(N)
        xu = np.full(N, self.maxV + 1.0 - 1e-9)
        from pymoo.core.problem import Problem
        Problem.__init__(self, n_var=N, n_obj=3, n_constr=0, xl=xl, xu=xu)

    def _decode_rk(self, x):
        N       = self.Ndems
        vehicle = np.floor(x).astype(int)          # 0=unserved, 1..maxV
        order_k = x - vehicle                       # fractional key [0,1)
        served  = vehicle.copy()
        arrivals   = np.zeros(N)
        departures = np.zeros(N)
        legs       = []
        total_dv   = 0.0

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle == v)[0]
            if len(indices) == 0:
                continue
            sorted_idx = indices[np.argsort(order_k[indices])]

            state_sat = self.depot_id
            state_dep = 0.0
            cum_dv    = 0.0

            for i in sorted_idx:
                sid = int(self.sat_ids[i])

                # Direct leg: scan arr_grid for first feasible, budget-respecting entry
                key_dir = (state_sat, sid)
                tof_dir = self.min_tof_table[key_dir][0] if key_dir in self.min_tof_table else 0.0
                min_arr = state_dep + tof_dir
                dep_s   = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                arr     = None
                dv_dir  = np.inf
                for ai in range(int(np.searchsorted(self.arr_grid, min_arr).clip(0, len(self.arr_grid) - 1)),
                                len(self.arr_grid)):
                    a_cand = float(self.arr_grid[ai])
                    if a_cand + self.service_times[i] > self.deadlines[i]:
                        break
                    c = self.cost_table.get((state_sat, sid, dep_s, a_cand), np.inf)
                    if not np.isinf(c) and c <= self.dv_budget:
                        arr    = a_cand
                        dv_dir = c
                        break
                if arr is None:
                    served[i] = 0
                    continue
                new_cum = (dv_dir if state_sat == self.depot_id else cum_dv + dv_dir)

                # Force depot visit if over budget
                if new_cum > self.dv_budget and state_sat != self.depot_id:
                    key_to = (state_sat, self.depot_id)
                    tof_to = self.min_tof_table[key_to][0] if key_to in self.min_tof_table else 0.0
                    min_arr_dep = state_dep + tof_to
                    arr_dep  = None
                    dv_depot = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, min_arr_dep).clip(0, len(self.arr_grid) - 1)),
                                    len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        c = self.cost_table.get((state_sat, self.depot_id, dep_s, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr_dep  = a_cand
                            dv_depot = c
                            break
                    if arr_dep is None:
                        served[i] = 0
                        continue

                    dep_dep = arr_dep + self.refuel_time
                    key_fr  = (self.depot_id, sid)
                    tof_fr  = self.min_tof_table[key_fr][0] if key_fr in self.min_tof_table else 0.0
                    min_arr2 = dep_dep + tof_fr
                    dep_s2  = float(self.dep_grid[np.searchsorted(self.dep_grid, dep_dep).clip(0, len(self.dep_grid) - 1)])
                    arr     = None
                    dv_via  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, min_arr2).clip(0, len(self.arr_grid) - 1)),
                                    len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        if a_cand + self.service_times[i] > self.deadlines[i]:
                            break
                        c = self.cost_table.get((self.depot_id, sid, dep_s2, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr    = a_cand
                            dv_via = c
                            break
                    if arr is None:
                        served[i] = 0
                        continue

                    legs.append((state_sat, self.depot_id, dep_s, arr_dep))
                    legs.append((self.depot_id, sid, dep_s2, arr))
                    total_dv  += dv_depot + dv_via
                    state_sat  = sid
                    state_dep  = arr + self.service_times[i]
                    cum_dv     = dv_via

                else:
                    legs.append((state_sat, sid, dep_s, arr))
                    total_dv  += dv_dir
                    state_sat  = sid
                    state_dep  = arr + self.service_times[i]
                    cum_dv     = new_cum

                arrivals[i]   = arr
                departures[i] = state_dep

        return arrivals, departures, served, legs, total_dv

    def _evaluate(self, X, out, *args, **kwargs):
        N        = self.Ndems
        pop_size = len(X)
        F        = np.zeros((pop_size, self.n_obj))

        for idx in range(pop_size):
            x = X[idx]
            arrivals, departures, served, legs, total_dv = self._decode_rk(x)
            served_mask = served > 0

            F[idx, 0] = total_dv
            F[idx, 1] = float(len(np.unique(served[served_mask]))) if served_mask.any() else 0.0
            F[idx, 2] = float(np.sum(self.asset_values[~served_mask]))

        out["F"] = F


class OOSProblemRK_OT(OOSProblemRK):
    """
    NSGA-III with 2N random-keys encoding:
      x[0:N]  — demand genes:  floor = vehicle (0=unserved), frac = normalised arrival
      x[N:2N] — depot genes:   floor = vehicle (0=inactive), frac = normalised arrival

    arrival[i] = frac(x[i]) * T_horizon, where T_horizon = max(deadlines) - max(service_times).
    departure[i] = arrival[i] + service_time[i].
    Visit order = sorted by arrival time.  Legs where arr < prev_dep + min_tof
    or arr + service > deadline are dropped (served=0).  Single cost-table lookup
    per leg (snap dep/arr to grids); no searching loop, no post-hoc opt_times pass.

    If the ΔV budget is exceeded on a leg and no depot gene covers it, the demand
    is dropped (served=0).  Depot genes are ignored when: already at depot, or no
    demands follow them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        N = self.Ndems
        self.n_var = 2 * N
        self.xl    = np.zeros(2 * N)
        self.xu    = np.full(2 * N, float(self.maxV))

    def _decode_rk2(self, x):
        N = self.Ndems
        T_horizon = float(np.max(self.deadlines) - np.max(self.service_times))

        # Demand genes: floor = vehicle, frac = normalised arrival time
        vehicle_d = np.floor(x[:N]).astype(int)
        frac_d    = x[:N] - vehicle_d
        served    = vehicle_d.copy()

        # Depot genes: floor = vehicle, frac = normalised arrival time
        vehicle_dep = np.floor(x[N:]).astype(int)
        frac_dep    = x[N:] - vehicle_dep

        arrivals   = np.zeros(N)
        departures = np.zeros(N)
        legs       = []
        total_dv   = 0.0
        vehicles   = []

        for v in range(1, self.maxV + 1):
            dem_idx = np.where(vehicle_d == v)[0]
            if len(dem_idx) == 0:
                continue
            dem_sorted = dem_idx[np.argsort(frac_d[dem_idx])]

            dep_idx = np.where(vehicle_dep == v)[0]

            # Merged sequence sorted by arrival time
            sequence = [(frac_d[i] * T_horizon, 'D', i) for i in dem_sorted]
            sequence += [(frac_dep[j] * T_horizon, 'R', j) for j in dep_idx]
            sequence.sort(key=lambda t: t[0])

            # Precompute whether any demand follows each position
            has_future_demand = [False] * len(sequence)
            seen = False
            for pos in range(len(sequence) - 1, -1, -1):
                has_future_demand[pos] = seen
                if sequence[pos][1] == 'D':
                    seen = True

            state_sat = self.depot_id
            state_dep = 0.0
            cum_dv    = 0.0

            veh = {'sat_ids': [self.depot_id], 'arrivals': [0.0],
                   'departures': [0.0], 'costs': [0.0], 'demand_idxs': [-1]}

            for pos, (arr_raw, stype, idx) in enumerate(sequence):

                if stype == 'R':
                    if state_sat == self.depot_id or not has_future_demand[pos]:
                        continue

                    key_to = (state_sat, self.depot_id)
                    tof_to = self.min_tof_table[key_to][0] if key_to in self.min_tof_table else 0.0
                    earliest_dep = max(arr_raw, state_dep + tof_to)

                    dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                    arr_dep = None
                    dv_dep  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, earliest_dep).clip(0, len(self.arr_grid) - 1)), len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        c = self.cost_table.get((state_sat, self.depot_id, dep_s, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr_dep = a_cand
                            dv_dep  = c
                            break
                    if arr_dep is None:
                        continue

                    dep_dep = arr_dep + self.refuel_time
                    legs.append((state_sat, self.depot_id, dep_s, arr_dep))
                    total_dv += dv_dep

                    veh['sat_ids'].append(self.depot_id)
                    veh['arrivals'].append(arr_dep)
                    veh['departures'].append(dep_dep)
                    veh['costs'].append(dv_dep)
                    veh['demand_idxs'].append(-1)

                    state_sat = self.depot_id
                    state_dep = dep_dep
                    cum_dv    = 0.0

                else:
                    i   = idx
                    sid = int(self.sat_ids[i])

                    key_dir = (state_sat, sid)
                    tof_dir = self.min_tof_table[key_dir][0] if key_dir in self.min_tof_table else 0.0
                    earliest = max(arr_raw, state_dep + tof_dir)

                    dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                    arr     = None
                    dv_dir  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, earliest).clip(0, len(self.arr_grid) - 1)), len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        if a_cand + self.service_times[i] > self.deadlines[i]:
                            break
                        c = self.cost_table.get((state_sat, sid, dep_s, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr    = a_cand
                            dv_dir = c
                            break
                    if arr is None:
                        served[i] = 0
                        continue

                    new_cum = dv_dir if state_sat == self.depot_id else cum_dv + dv_dir

                    if new_cum > self.dv_budget and state_sat != self.depot_id:
                        served[i] = 0
                        continue

                    legs.append((state_sat, sid, dep_s, arr))
                    total_dv += dv_dir

                    veh['sat_ids'].append(sid)
                    veh['arrivals'].append(arr)
                    veh['departures'].append(arr + self.service_times[i])
                    veh['costs'].append(dv_dir)
                    veh['demand_idxs'].append(i)

                    arrivals[i]   = arr
                    departures[i] = arr + self.service_times[i]

                    state_sat = sid
                    state_dep = arr + self.service_times[i]
                    cum_dv    = new_cum

            if len(veh['sat_ids']) > 1:
                vehicles.append(veh)

        return arrivals, departures, served, legs, total_dv

    def _opt_times_pass(self, vehicles, shift=15.0, top_pct=0.5):
        """Single opt_times sweep (non-iterative) used inside the decoder."""
        all_legs = []
        for v_idx, veh in enumerate(vehicles):
            for i in range(1, len(veh['sat_ids'])):
                all_legs.append((veh['costs'][i], v_idx, i))
        if not all_legs:
            return sum(sum(veh['costs']) for veh in vehicles)

        all_legs.sort(key=lambda t: -t[0])
        topn = max(1, int(len(all_legs) * top_pct))

        for _, v_idx, i in all_legs[:topn]:
            veh      = vehicles[v_idx]
            from_sat = veh['sat_ids'][i - 1]
            to_sat   = veh['sat_ids'][i]
            dep_prev = veh['departures'][i - 1]
            arr_i    = veh['arrivals'][i]
            key      = (from_sat, to_sat)
            min_tof  = self.min_tof_table[key][0] if key in self.min_tof_table else 0.0

            best_cost = veh['costs'][i]
            best_move = None

            new_arr_i = arr_i + shift
            if new_arr_i - dep_prev >= min_tof:
                feas = True
                for j in range(i, len(veh['sat_ids'])):
                    d = veh['demand_idxs'][j]
                    if d >= 0 and veh['arrivals'][j] + shift + self.service_times[d] > self.deadlines[d]:
                        feas = False; break
                if feas:
                    dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, dep_prev).clip(0, len(self.dep_grid) - 1)])
                    arr_s = float(self.arr_grid[np.searchsorted(self.arr_grid, new_arr_i).clip(0, len(self.arr_grid) - 1)])
                    c = self.cost_table.get((from_sat, to_sat, dep_s, arr_s), np.inf)
                    if not np.isinf(c) and c < best_cost:
                        best_cost = c; best_move = 'C'

            new_dep_prev = dep_prev - shift
            if new_dep_prev >= 0.0:
                dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, new_dep_prev).clip(0, len(self.dep_grid) - 1)])
                arr_s = float(self.arr_grid[np.searchsorted(self.arr_grid, arr_i).clip(0, len(self.arr_grid) - 1)])
                c = self.cost_table.get((from_sat, to_sat, dep_s, arr_s), np.inf)
                if not np.isinf(c) and c < best_cost:
                    best_cost = c; best_move = 'D'

            if best_move == 'C':
                for j in range(i, len(veh['arrivals'])):
                    veh['arrivals'][j] += shift; veh['departures'][j] += shift
                veh['costs'][i] = best_cost
            elif best_move == 'D':
                for j in range(i):
                    veh['arrivals'][j] -= shift; veh['departures'][j] -= shift
                veh['costs'][i] = best_cost

        return sum(sum(veh['costs']) for veh in vehicles)

    def _evaluate(self, X, out, *args, **kwargs):
        N        = self.Ndems
        pop_size = len(X)
        F        = np.zeros((pop_size, self.n_obj))
        for idx in range(pop_size):
            arrivals, departures, served, legs, total_dv = self._decode_rk2(X[idx])
            served_mask = served > 0
            F[idx, 0] = total_dv
            F[idx, 1] = float(len(np.unique(served[served_mask]))) if served_mask.any() else 0.0
            F[idx, 2] = float(np.sum(self.asset_values[~served_mask]))
        out["F"] = F

    def _build_vehicles(self, x):
        """
        Decode x and build per-vehicle schedule structures needed for opt_times.
        Returns (arrivals, departures, served, legs, total_dv, vehicles).

        Each vehicle dict has:
            sat_ids:     list of int   (satellite indices incl. depot stops)
            arrivals:    list of float
            departures:  list of float
            costs:       list of float (leg cost at each position; 0 at start)
            demand_idxs: list of int   (-1 for depot stops, else demand index)
        """
        N       = self.Ndems
        vehicle = np.floor(x).astype(int)
        order_k = x - vehicle
        served  = vehicle.copy()
        arrivals   = np.zeros(N)
        departures = np.zeros(N)
        legs       = []
        total_dv   = 0.0
        vehicles   = []

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle == v)[0]
            if len(indices) == 0:
                continue
            sorted_idx = indices[np.argsort(order_k[indices])]

            state_sat = self.depot_id
            state_dep = 0.0
            cum_dv    = 0.0

            veh = {'sat_ids': [self.depot_id], 'arrivals': [0.0],
                   'departures': [0.0], 'costs': [0.0], 'demand_idxs': [-1]}

            for i in sorted_idx:
                sid = int(self.sat_ids[i])

                key_dir = (state_sat, sid)
                tof_dir = self.min_tof_table[key_dir][0] if key_dir in self.min_tof_table else 0.0
                min_arr = state_dep + tof_dir
                dep_s   = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                arr     = None
                dv_dir  = np.inf
                for ai in range(int(np.searchsorted(self.arr_grid, min_arr).clip(0, len(self.arr_grid) - 1)),
                                len(self.arr_grid)):
                    a_cand = float(self.arr_grid[ai])
                    if a_cand + self.service_times[i] > self.deadlines[i]:
                        break
                    c = self.cost_table.get((state_sat, sid, dep_s, a_cand), np.inf)
                    if not np.isinf(c) and c <= self.dv_budget:
                        arr    = a_cand
                        dv_dir = c
                        break
                if arr is None:
                    served[i] = 0
                    continue
                new_cum = (dv_dir if state_sat == self.depot_id else cum_dv + dv_dir)

                if new_cum > self.dv_budget and state_sat != self.depot_id:
                    key_to = (state_sat, self.depot_id)
                    tof_to = self.min_tof_table[key_to][0] if key_to in self.min_tof_table else 0.0
                    min_arr_dep = state_dep + tof_to
                    arr_dep  = None
                    dv_depot = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, min_arr_dep).clip(0, len(self.arr_grid) - 1)),
                                    len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        c = self.cost_table.get((state_sat, self.depot_id, dep_s, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr_dep  = a_cand
                            dv_depot = c
                            break
                    if arr_dep is None:
                        served[i] = 0
                        continue

                    dep_dep = arr_dep + self.refuel_time
                    key_fr  = (self.depot_id, sid)
                    tof_fr  = self.min_tof_table[key_fr][0] if key_fr in self.min_tof_table else 0.0
                    min_arr2 = dep_dep + tof_fr
                    dep_s2  = float(self.dep_grid[np.searchsorted(self.dep_grid, dep_dep).clip(0, len(self.dep_grid) - 1)])
                    arr     = None
                    dv_via  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, min_arr2).clip(0, len(self.arr_grid) - 1)),
                                    len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        if a_cand + self.service_times[i] > self.deadlines[i]:
                            break
                        c = self.cost_table.get((self.depot_id, sid, dep_s2, a_cand), np.inf)
                        if not np.isinf(c) and c <= self.dv_budget:
                            arr    = a_cand
                            dv_via = c
                            break
                    if arr is None:
                        served[i] = 0
                        continue

                    legs.append((state_sat, self.depot_id, dep_s, arr_dep))
                    legs.append((self.depot_id, sid, dep_s2, arr))
                    total_dv += dv_depot + dv_via

                    veh['sat_ids'].append(self.depot_id)
                    veh['arrivals'].append(arr_dep)
                    veh['departures'].append(dep_dep)
                    veh['costs'].append(dv_depot)
                    veh['demand_idxs'].append(-1)

                    veh['sat_ids'].append(sid)
                    veh['arrivals'].append(arr)
                    veh['departures'].append(arr + self.service_times[i])
                    veh['costs'].append(dv_via)
                    veh['demand_idxs'].append(i)

                    state_sat = sid
                    state_dep = arr + self.service_times[i]
                    cum_dv    = dv_via

                else:
                    legs.append((state_sat, sid, dep_s, arr))
                    total_dv += dv_dir

                    veh['sat_ids'].append(sid)
                    veh['arrivals'].append(arr)
                    veh['departures'].append(arr + self.service_times[i])
                    veh['costs'].append(dv_dir)
                    veh['demand_idxs'].append(i)

                    state_sat = sid
                    state_dep = arr + self.service_times[i]
                    cum_dv    = new_cum

                arrivals[i]   = arr
                departures[i] = state_dep

            if len(veh['sat_ids']) > 1:
                vehicles.append(veh)

        return arrivals, departures, served, legs, total_dv, vehicles

    def _opt_times(self, vehicles, shift=15.0, top_pct=0.5):
        """
        Python port of opt_times_combined (algoMDLS.jl).
        Modifies vehicles in-place. Returns improved total_dv.

        For each of the top_pct most expensive legs, tries two block moves:
          C: shift block [i:end] arrivals+departures later  → cheaper leg cost
          D: shift block [0:i-1] arrivals+departures earlier → cheaper leg cost
        """
        all_legs = []
        for v_idx, veh in enumerate(vehicles):
            for i in range(1, len(veh['sat_ids'])):
                all_legs.append((veh['costs'][i], v_idx, i))

        if not all_legs:
            return sum(sum(veh['costs']) for veh in vehicles)

        all_legs.sort(key=lambda t: -t[0])
        topn = max(1, int(len(all_legs) * top_pct))

        while True:
            prev_dv = sum(sum(veh['costs']) for veh in vehicles)

            for _, v_idx, i in all_legs[:topn]:
                veh      = vehicles[v_idx]
                from_sat = veh['sat_ids'][i - 1]
                to_sat   = veh['sat_ids'][i]
                dep_prev = veh['departures'][i - 1]
                arr_i    = veh['arrivals'][i]

                key     = (from_sat, to_sat)
                min_tof = self.min_tof_table[key][0] if key in self.min_tof_table else 0.0

                best_cost = veh['costs'][i]
                best_move = None

                # Move C: shift block [i:] later
                new_arr_i = arr_i + shift
                if new_arr_i - dep_prev >= min_tof:
                    feas = True
                    for j in range(i, len(veh['sat_ids'])):
                        d = veh['demand_idxs'][j]
                        if d >= 0 and veh['arrivals'][j] + shift + self.service_times[d] > self.deadlines[d]:
                            feas = False
                            break
                    if feas:
                        dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, dep_prev).clip(0, len(self.dep_grid) - 1)])
                        arr_s = float(self.arr_grid[np.searchsorted(self.arr_grid, new_arr_i).clip(0, len(self.arr_grid) - 1)])
                        c = self.cost_table.get((from_sat, to_sat, dep_s, arr_s), np.inf)
                        if not np.isinf(c) and c < best_cost:
                            best_cost = c
                            best_move = 'C'

                # Move D: shift block [:i] earlier
                new_dep_prev = dep_prev - shift
                if new_dep_prev >= 0.0:
                    dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, new_dep_prev).clip(0, len(self.dep_grid) - 1)])
                    arr_s = float(self.arr_grid[np.searchsorted(self.arr_grid, arr_i).clip(0, len(self.arr_grid) - 1)])
                    c = self.cost_table.get((from_sat, to_sat, dep_s, arr_s), np.inf)
                    if not np.isinf(c) and c < best_cost:
                        best_cost = c
                        best_move = 'D'

                if best_move is None:
                    continue

                if best_move == 'C':
                    for j in range(i, len(veh['arrivals'])):
                        veh['arrivals'][j]   += shift
                        veh['departures'][j] += shift
                    veh['costs'][i] = best_cost
                else:
                    for j in range(i):
                        veh['arrivals'][j]   -= shift
                        veh['departures'][j] -= shift
                    veh['costs'][i] = best_cost

            new_dv = sum(sum(veh['costs']) for veh in vehicles)
            if new_dv >= prev_dv - 1e-6:
                break
            # Re-sort legs by updated costs for next iteration
            all_legs = []
            for v_idx, veh in enumerate(vehicles):
                for i in range(1, len(veh['sat_ids'])):
                    all_legs.append((veh['costs'][i], v_idx, i))
            all_legs.sort(key=lambda t: -t[0])
            topn = max(1, int(len(all_legs) * top_pct))

        return sum(sum(veh['costs']) for veh in vehicles)

    def post_opt(self, X, F, n_workers=None):
        """
        Apply opt_times to each solution in X in parallel (ThreadPoolExecutor).
        Returns a copy of F with updated f1_dv values.
        """
        from concurrent.futures import ThreadPoolExecutor

        F_out = F.copy()

        def _process(idx):
            _, _, _, _, _, vehicles = self._build_vehicles(X[idx])
            if not vehicles:
                return idx, F[idx, 0]
            return idx, self._opt_times(vehicles)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for idx, dv in pool.map(_process, range(len(X))):
                F_out[idx, 0] = min(dv, F[idx, 0])

        return F_out


