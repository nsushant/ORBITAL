import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from loaders import snap_cost


# ── SBX / PM helpers (used by OOSCrossover and OOSMutation) ──────────────────

def _sbx(p1, p2, xl, xu, eta):
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta + 1)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1)))
    c1 = np.clip(0.5 * ((p1 + p2) - beta * np.abs(p2 - p1)), xl, xu)
    c2 = np.clip(0.5 * ((p1 + p2) + beta * np.abs(p2 - p1)), xl, xu)
    return c1, c2


def _pm(x, xl, xu, eta):
    delta = np.maximum(xu - xl, 1e-10)
    u = np.random.rand(len(x))
    d = np.where(u < 0.5,
                 (2.0 * u) ** (1.0 / (eta + 1)) - 1.0,
                 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1)))
    return np.clip(x + d * delta, xl, xu)


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

    All solutions are made feasible by OOSRepair before evaluation,
    so n_constr = 0.

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

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair_individual(self, x, stochastic=False, n_slots=4):
        """
        Walk each vehicle's sorted tour and clamp arrivals up to satisfy
        min-tof ordering. Demands that cannot be feasibly reached are
        marked unserved (vehicle_assignment = 0).

        stochastic : if True, pick randomly among the first n_slots valid
                     arrival grid slots instead of always taking the earliest.
        n_slots    : max candidate slots to collect before sampling.

        Modifies x in-place and returns it.
        """
        N             = self.Ndems
        deadlines     = self.deadlines
        service_times = self.service_times
        sat_ids       = self.sat_ids
        depot_id      = self.depot_id
        min_tof_table = self.min_tof_table
        refuel_time   = self.refuel_time

        # 4N read — OLD 5N:
        # visit_order         = x[0:N]
        # vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
        # arrivals            = x[2*N:3*N].copy()
        # depot_visits        = np.clip(np.round(x[3*N:4*N]), 0, 1).astype(int)
        # depot_arrivals      = x[4*N:5*N].copy()
        vehicle_assignments = np.clip(np.round(x[0:N]),     0, self.maxV).astype(int)
        arrivals            = x[N:2*N].copy() * deadlines        # denormalise
        depot_visits        = np.clip(np.round(x[2*N:3*N]), 0, 1).astype(int)
        depot_arrivals      = x[3*N:4*N].copy() * self.max_deadline  # denormalise

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle_assignments == v)[0]
            if len(indices) == 0:
                continue

            # Sort by arrival time instead of visit_order variable
            # OLD 5N: sorted_local = np.argsort(visit_order[indices])
            sorted_local = np.argsort(arrivals[indices])
            sorted_idx   = indices[sorted_local]
            depot_vis    = depot_visits[sorted_idx].copy()
            depot_vis[-1] = 1   # circular wrap: last demand always has depot-return flag

            last_dep_time    = 0.0
            last_sat_idx     = depot_id
            last_depot_vis   = False
            last_demand_idx  = -1
            cum_dv           = 0.0

            for k, i in enumerate(sorted_idx):
                if last_depot_vis:
                    key_to_dep   = (last_sat_idx, depot_id)
                    key_from_dep = (depot_id, sat_ids[i])
                    tof_to_dep   = min_tof_table[key_to_dep][0]   if key_to_dep   in min_tof_table else 0.0
                    tof_from_dep = min_tof_table[key_from_dep][0] if key_from_dep in min_tof_table else 0.0
                    arr_depot_min = last_dep_time + tof_to_dep
                    arr_depot = max(depot_arrivals[last_demand_idx], arr_depot_min)
                    dep_depot = arr_depot + refuel_time
                    depot_arrivals[last_demand_idx] = arr_depot
                    min_arr   = dep_depot + tof_from_dep
                else:
                    key     = (last_sat_idx, sat_ids[i])
                    min_tof = min_tof_table[key][0] if key in min_tof_table else 0.0
                    min_arr = last_dep_time + min_tof

                arrivals[i] = max(arrivals[i], min_arr)

                leg_from = depot_id      if last_depot_vis else last_sat_idx
                leg_dep  = dep_depot     if last_depot_vis else last_dep_time
                dep_grid = self.dep_grid
                arr_grid = self.arr_grid
                dep_s = float(dep_grid[np.searchsorted(dep_grid, leg_dep).clip(0, len(dep_grid) - 1)])
                arr_start_idx = np.searchsorted(arr_grid, arrivals[i]).clip(0, len(arr_grid) - 1)
                found = False
                if stochastic:
                    valid_slots = []
                    for ai in range(int(arr_start_idx), len(arr_grid)):
                        arr_s = float(arr_grid[ai])
                        if arr_s > deadlines[i]:
                            break
                        c = self.cost_table.get((int(leg_from), int(sat_ids[i]), dep_s, arr_s), np.inf)
                        if c < 1e7:
                            valid_slots.append(arr_s)
                            if len(valid_slots) >= n_slots:
                                break
                    if valid_slots:
                        n = len(valid_slots)
                        weights = np.arange(n, 0, -1, dtype=float)
                        weights /= weights.sum()
                        arrivals[i] = valid_slots[np.random.choice(n, p=weights)]
                        found = True
                else:
                    for ai in range(int(arr_start_idx), len(arr_grid)):
                        arr_s = float(arr_grid[ai])
                        if arr_s > deadlines[i]:
                            break
                        c = self.cost_table.get((int(leg_from), int(sat_ids[i]), dep_s, arr_s), np.inf)
                        if c < 1e7:
                            arrivals[i] = arr_s
                            found = True
                            break

                if not found or arrivals[i] + service_times[i] > deadlines[i]:
                    vehicle_assignments[i] = 0
                    arrivals[i]  = 0.0
                    depot_vis[k] = 0
                else:
                    leg_dv = self.cost_table.get(
                        (int(leg_from), int(sat_ids[i]), dep_s, float(arrivals[i])), 0.0)
                    cum_dv = leg_dv if last_depot_vis else cum_dv + leg_dv
                    if cum_dv > self.dv_budget:
                        depot_vis[k] = 1
                        cum_dv = 0.0
                    dep_i = arrivals[i] + service_times[i]
                    last_dep_time   = dep_i
                    last_sat_idx    = sat_ids[i]
                    last_demand_idx = i
                    last_depot_vis  = bool(depot_vis[k])

            for k, i in enumerate(sorted_idx):
                depot_visits[i] = depot_vis[k]

        # 4N write-back (normalise arrivals and depot_arrivals) — OLD 5N:
        # x[N:2*N]   = vehicle_assignments.astype(float)
        # x[2*N:3*N] = arrivals
        # x[3*N:4*N] = depot_visits.astype(float)
        # x[4*N:5*N] = depot_arrivals
        x[0:N]     = vehicle_assignments.astype(float)
        x[N:2*N]   = np.where(deadlines > 0, arrivals / deadlines, 0.0)
        x[2*N:3*N] = depot_visits.astype(float)
        x[3*N:4*N] = depot_arrivals / self.max_deadline
        return x

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, x):
        """
        Build arrivals, departures, vehicle tours, and leg list from a
        repaired solution vector.

        Returns
        -------
        arrivals     : ndarray (Ndems,)
        departures   : ndarray (Ndems,)
        sorted_tours : list[list[int]]
        legs         : list of (from_idx, to_idx, dep_time, arr_time)
        """
        N             = self.Ndems
        deadlines     = self.deadlines
        service_times = self.service_times
        sat_ids       = self.sat_ids
        depot_id      = self.depot_id
        min_tof_table = self.min_tof_table
        refuel_time   = self.refuel_time

        # 4N read — OLD 5N:
        # visit_order         = x[0:N]
        # vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
        # arrivals            = x[2*N:3*N].copy()
        # depot_visits        = np.clip(np.round(x[3*N:4*N]), 0, 1).astype(int)
        # depot_arrivals      = np.clip(x[4*N:5*N], 0.0, self.max_deadline)
        vehicle_assignments = np.clip(np.round(x[0:N]),     0, self.maxV).astype(int)
        arrivals            = x[N:2*N].copy() * deadlines        # denormalise
        depot_visits        = np.clip(np.round(x[2*N:3*N]), 0, 1).astype(int)
        depot_arrivals      = np.clip(x[3*N:4*N] * self.max_deadline, 0.0, self.max_deadline)

        departures   = np.zeros(N)
        sorted_tours = []
        legs         = []

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle_assignments == v)[0]
            if len(indices) == 0:
                sorted_tours.append([])
                continue

            # Sort by arrival time — OLD 5N: sorted_local = np.argsort(visit_order[indices])
            sorted_local = np.argsort(arrivals[indices])
            sorted_idx   = indices[sorted_local]
            depot_vis    = depot_visits[sorted_idx].copy()
            depot_vis[-1] = 1

            sorted_tours.append(sorted_idx.tolist())

            for k, i in enumerate(sorted_idx):
                departures[i] = arrivals[i] + service_times[i]

                if k == 0:
                    dep_s = float(self.dep_grid[np.searchsorted(self.dep_grid, 0.0).clip(0, len(self.dep_grid) - 1)])
                    legs.append((depot_id, sat_ids[i], dep_s, arrivals[i]))
                else:
                    prev = sorted_idx[k - 1]
                    if depot_vis[k - 1]:
                        key_to_dep    = (sat_ids[prev], depot_id)
                        tof_to_dep    = min_tof_table[key_to_dep][0] if key_to_dep in min_tof_table else 0.0
                        arr_depot_min = departures[prev] + tof_to_dep
                        arr_depot     = max(depot_arrivals[prev], arr_depot_min)
                        dep_depot     = arr_depot + self.refuel_time
                        dep_depot_s   = float(self.dep_grid[np.searchsorted(self.dep_grid, dep_depot).clip(0, len(self.dep_grid) - 1)])
                        legs.append((sat_ids[prev], depot_id, departures[prev], arr_depot))
                        legs.append((depot_id, sat_ids[i], dep_depot_s, arrivals[i]))
                    else:
                        legs.append((sat_ids[prev], sat_ids[i], departures[prev], arrivals[i]))

        return arrivals, departures, sorted_tours, legs

    # ------------------------------------------------------------------
    # ΔV cost
    # ------------------------------------------------------------------

    _INFEASIBLE_LEG_COST = 1e6  # m/s

    def compute_dv(self, legs):
        total = 0.0
        for f, t, d, a in legs:
            c = snap_cost(self.cost_table, self.dep_grid, self.arr_grid, f, t, d, a)
            total += self._INFEASIBLE_LEG_COST if np.isinf(c) else c
        return total

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def _evaluate(self, X, out, *args, **kwargs):
        N        = self.Ndems
        pop_size = len(X)
        F        = np.zeros((pop_size, self.n_obj))

        for idx in range(pop_size):
            x = X[idx]
            arrivals, departures, sorted_tours, legs = self.decode(x)

            # OLD 5N: vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
            vehicle_assignments = np.clip(np.round(x[0:N]), 0, self.maxV).astype(int)
            served_mask = vehicle_assignments > 0

            F[idx, 0] = self.compute_dv(legs)
            F[idx, 1] = float(len(np.unique(vehicle_assignments[served_mask]))) if served_mask.any() else 0.0
            F[idx, 2] = float(np.sum(self.asset_values[~served_mask]))

        out["F"] = F


class OOSRepair(Repair):
    """pymoo Repair wrapper that calls OOSProblem.repair_individual."""

    def __init__(self, stochastic=False, n_slots=4):
        super().__init__()
        self.stochastic = stochastic
        self.n_slots    = n_slots

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = problem.repair_individual(X[i].copy(),
                                             stochastic=self.stochastic,
                                             n_slots=self.n_slots)
        return X


# ── Mixed-encoding operators ──────────────────────────────────────────────────

class OOSCrossover(Crossover):
    """
    Segment-aware crossover for the 4N OOS encoding:
      x[0:N]    vehicle_assignments → uniform (categorical integer)
      x[N:2N]   arrivals_norm       → SBX  (continuous)
      x[2N:3N]  depot_visits        → uniform (binary)
      x[3N:4N]  depot_arrivals_norm → SBX  (continuous)

    # OLD 5N segments:
    # SBX:     slice(0,N) visit_order, slice(2N,3N) arrivals, slice(4N,5N) depot_arrivals
    # uniform: slice(N,2N) vehicle_assignments, slice(3N,4N) depot_visits
    """

    def __init__(self, eta=20, prob=0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)
        self.eta = eta

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        N   = problem.Ndems
        eta = self.eta
        Y   = np.empty_like(X)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            c1, c2 = p1.copy(), p2.copy()

            # SBX on continuous segments (arrivals_norm, depot_arrivals_norm)
            for sl in (slice(N, 2 * N), slice(3 * N, 4 * N)):
                c1[sl], c2[sl] = _sbx(p1[sl], p2[sl],
                                       problem.xl[sl], problem.xu[sl], eta)

            # Uniform crossover on integer / binary segments
            for sl in (slice(0, N), slice(2 * N, 3 * N)):
                mask      = np.random.rand(N) < 0.5
                c1[sl]    = np.where(mask, p1[sl], p2[sl])
                c2[sl]    = np.where(mask, p2[sl], p1[sl])

            Y[0, k], Y[1, k] = c1, c2

        return Y


class OOSMutation(Mutation):
    """
    Segment-aware mutation for the 4N OOS encoding:
      x[0:N]    vehicle_assignments → random integer in [0, maxV]
      x[N:2N]   arrivals_norm       → polynomial mutation
      x[2N:3N]  depot_visits        → bitflip
      x[3N:4N]  depot_arrivals_norm → polynomial mutation

    # OLD 5N segments:
    # PM:      slice(0,N) visit_order, slice(2N,3N) arrivals, slice(4N,5N) depot_arrivals
    # randint: x[N:2N] vehicle_assignments
    # bitflip: x[3N:4N] depot_visits
    """

    def __init__(self, eta=20, prob_var=None, **kwargs):
        super().__init__(prob=1.0, **kwargs)
        self.eta      = eta
        self.prob_var = prob_var

    def _do(self, problem, X, **kwargs):
        N   = problem.Ndems
        eta = self.eta
        p   = self.prob_var if self.prob_var is not None else 1.0 / problem.n_var
        Y   = X.copy()

        for i in range(len(X)):
            # Polynomial mutation on continuous segments (arrivals_norm, depot_arrivals_norm)
            for sl in (slice(N, 2 * N), slice(3 * N, 4 * N)):
                mask = np.random.rand(N) < p
                if mask.any():
                    Y[i][sl][mask] = _pm(X[i][sl][mask],
                                         problem.xl[sl][mask],
                                         problem.xu[sl][mask], eta)

            # Random integer replacement for vehicle_assignments
            mask = np.random.rand(N) < p
            if mask.any():
                Y[i, 0:N][mask] = np.random.randint(
                    0, int(problem.maxV) + 1, int(mask.sum())
                ).astype(float)

            # Bitflip for depot_visits
            mask = np.random.rand(N) < p
            if mask.any():
                Y[i, 2 * N:3 * N][mask] = 1.0 - X[i, 2 * N:3 * N][mask]

        return Y


# ── Decoder-based problem (2N encoding, no repair needed) ────────────────────

class OOSProblemDecoder(OOSProblem):
    """
    NSGA-III variant with earliest-feasible-arrival decoder.

    Encoding (n_var = 2 * N):
      x[0:N]   visit_order keys   float [0, 1]   — sort to get visit sequence per vehicle
      x[N:2N]  vehicle_assignments int  [0, maxV]

    Decoder assigns the earliest feasible arrival to each demand in visit-order
    sequence. Depot visits are inserted automatically when cumulative ΔV would
    exceed dv_budget. Demands that cannot be served (arrival > deadline after
    depot insertion) are dropped (vehicle_assignment = 0). No repair operator needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N    = self.Ndems
        xl   = np.zeros(2 * N)
        xu   = np.concatenate([np.ones(N), np.full(N, self.maxV)])
        # re-initialise with 2N bounds (super sets 4N)
        from pymoo.core.problem import Problem
        Problem.__init__(self, n_var=2*N, n_obj=3, n_constr=0, xl=xl, xu=xu)

    def _decode_order(self, x):
        N             = self.Ndems
        visit_keys    = x[0:N]
        vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
        served        = vehicle_assignments.copy()
        arrivals      = np.zeros(N)
        departures    = np.zeros(N)
        legs          = []

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle_assignments == v)[0]
            if len(indices) == 0:
                continue
            sorted_idx = indices[np.argsort(visit_keys[indices])]

            state_sat = self.depot_id
            state_dep = 0.0
            cum_dv    = 0.0

            for i in sorted_idx:
                sid = int(self.sat_ids[i])

                # ── Direct leg: scan arr_grid forward for first valid entry ──
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
                    if not np.isinf(c):
                        arr    = a_cand
                        dv_dir = c
                        break
                if arr is None:
                    served[i] = 0
                    continue
                new_cum = (dv_dir if state_sat == self.depot_id else cum_dv + dv_dir)

                # ── Force depot visit if over budget ──────────────────────
                if new_cum > self.dv_budget and state_sat != self.depot_id:
                    # Leg 1: current satellite → depot (scan for valid arrival)
                    key_to  = (state_sat, self.depot_id)
                    tof_to  = self.min_tof_table[key_to][0] if key_to in self.min_tof_table else 0.0
                    min_arr_dep = state_dep + tof_to
                    arr_dep = None
                    for ai in range(int(np.searchsorted(self.arr_grid, min_arr_dep).clip(0, len(self.arr_grid) - 1)),
                                    len(self.arr_grid)):
                        a_cand = float(self.arr_grid[ai])
                        c = self.cost_table.get((state_sat, self.depot_id, dep_s, a_cand), np.inf)
                        if not np.isinf(c):
                            arr_dep = a_cand
                            break
                    if arr_dep is None:
                        served[i] = 0
                        continue

                    dep_dep = arr_dep + self.refuel_time

                    # Leg 2: depot → target demand (scan for valid arrival)
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
                        if not np.isinf(c):
                            arr    = a_cand
                            dv_via = c
                            break
                    if arr is None:
                        served[i] = 0
                        continue

                    legs.append((state_sat, self.depot_id, dep_s, arr_dep))
                    legs.append((self.depot_id, sid, dep_s2, arr))
                    state_sat = sid
                    state_dep = arr + self.service_times[i]
                    cum_dv    = dv_via

                else:
                    # ── Direct leg (deadline already checked in scan loop) ──
                    legs.append((state_sat, sid, dep_s, arr))
                    state_sat = sid
                    state_dep = arr + self.service_times[i]
                    cum_dv    = new_cum

                arrivals[i]   = arr
                departures[i] = state_dep

        return arrivals, departures, served, legs

    def _evaluate(self, X, out, *args, **kwargs):
        N        = self.Ndems
        pop_size = len(X)
        F        = np.zeros((pop_size, self.n_obj))

        for idx in range(pop_size):
            x = X[idx]
            arrivals, departures, served, legs = self._decode_order(x)
            served_mask = served > 0

            F[idx, 0] = self.compute_dv(legs)
            F[idx, 1] = float(len(np.unique(served[served_mask]))) if served_mask.any() else 0.0
            F[idx, 2] = float(np.sum(self.asset_values[~served_mask]))

        out["F"] = F


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
      x[0:N]  — demand genes:  floor = vehicle (0=unserved), frac = visit order key
      x[N:2N] — depot genes:   floor = vehicle (0=inactive), frac = insertion key

    The decoder merges GA-placed depot visits into each vehicle's visit sequence
    by fractional key. If the ΔV budget is exceeded on a leg and no depot gene
    covers it, the demand is dropped (served=0) rather than force-inserting a
    depot. This makes the GA fully responsible for depot placement: missing depots
    increase unrecovered value, driving selection toward better placements.
    A single opt_times pass polishes arrival timing after the topology is built.

    Depot genes are ignored when: already at depot, or no demands follow them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        N = self.Ndems
        self.n_var = 2 * N
        self.xl    = np.zeros(2 * N)
        self.xu    = np.full(2 * N, float(self.maxV))

    def _decode_rk2(self, x):
        N = self.Ndems

        # Demand genes
        vehicle_d = np.floor(x[:N]).astype(int)
        order_d   = x[:N] - vehicle_d
        served    = vehicle_d.copy()

        # Depot genes
        vehicle_dep = np.floor(x[N:]).astype(int)
        order_dep   = x[N:] - vehicle_dep

        arrivals   = np.zeros(N)
        departures = np.zeros(N)
        legs       = []
        total_dv   = 0.0
        vehicles   = []

        for v in range(1, self.maxV + 1):
            dem_idx = np.where(vehicle_d == v)[0]
            if len(dem_idx) == 0:
                continue
            dem_sorted = dem_idx[np.argsort(order_d[dem_idx])]
            dem_keys   = order_d[dem_sorted]

            dep_idx  = np.where(vehicle_dep == v)[0]
            dep_keys = order_dep[dep_idx]

            # Merged sequence: (key, type, index)
            sequence = [(k, 'D', i) for k, i in zip(dem_keys, dem_sorted)]
            sequence += [(k, 'R', j) for k, j in zip(dep_keys, dep_idx)]
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

            for pos, (key, stype, idx) in enumerate(sequence):

                if stype == 'R':
                    # Skip if already at depot or no future demands
                    if state_sat == self.depot_id or not has_future_demand[pos]:
                        continue

                    key_to  = (state_sat, self.depot_id)
                    tof_to  = self.min_tof_table[key_to][0] if key_to in self.min_tof_table else 0.0
                    dep_s   = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                    arr_dep = None
                    dv_dep  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, state_dep + tof_to).clip(0, len(self.arr_grid) - 1)), len(self.arr_grid)):
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
                    dep_s   = float(self.dep_grid[np.searchsorted(self.dep_grid, state_dep).clip(0, len(self.dep_grid) - 1)])
                    arr     = None
                    dv_dir  = np.inf
                    for ai in range(int(np.searchsorted(self.arr_grid, state_dep + tof_dir).clip(0, len(self.arr_grid) - 1)), len(self.arr_grid)):
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

                    # Budget exceeded and no GA depot covered it → drop demand
                    if new_cum > self.dv_budget and state_sat != self.depot_id:
                        served[i] = 0
                        continue
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

        # Single opt_times pass to polish arrival timing
        if vehicles:
            total_dv = self._opt_times_pass(vehicles)

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


class OOSCrossoverDecoder(Crossover):
    """Crossover for 2N decoder encoding: SBX on visit_order, uniform on vehicle_assignments."""

    def __init__(self, eta=20, prob=0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)
        self.eta = eta

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        N   = problem.Ndems
        Y   = np.empty_like(X)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            c1, c2 = p1.copy(), p2.copy()

            # SBX on visit_order keys
            c1[0:N], c2[0:N] = _sbx(p1[0:N], p2[0:N],
                                      problem.xl[0:N], problem.xu[0:N], self.eta)
            # Uniform on vehicle_assignments
            mask     = np.random.rand(N) < 0.5
            c1[N:2*N] = np.where(mask, p1[N:2*N], p2[N:2*N])
            c2[N:2*N] = np.where(mask, p2[N:2*N], p1[N:2*N])

            Y[0, k], Y[1, k] = c1, c2
        return Y


class OOSMutationDecoder(Mutation):
    """Mutation for 2N decoder encoding: PM on visit_order, random-int on vehicle_assignments."""

    def __init__(self, eta=20, prob_var=None, **kwargs):
        super().__init__(prob=1.0, **kwargs)
        self.eta      = eta
        self.prob_var = prob_var

    def _do(self, problem, X, **kwargs):
        N   = problem.Ndems
        p   = self.prob_var if self.prob_var is not None else 1.0 / problem.n_var
        Y   = X.copy()

        for i in range(len(X)):
            # PM on visit_order keys
            mask = np.random.rand(N) < p
            if mask.any():
                Y[i, 0:N][mask] = _pm(X[i, 0:N][mask],
                                       problem.xl[0:N][mask],
                                       problem.xu[0:N][mask], self.eta)
            # Random integer replacement for vehicle_assignments
            mask = np.random.rand(N) < p
            if mask.any():
                Y[i, N:2*N][mask] = np.random.randint(
                    0, int(problem.maxV) + 1, int(mask.sum())
                ).astype(float)
        return Y


class MOPSO_CD_Repair(MOPSO_CD):
    """MOPSO_CD with OOSRepair injected after each position update."""

    def __init__(self, repair=None, **kwargs):
        super().__init__(**kwargs)
        self._oos_repair = repair

    def _initialize_infill(self):
        pop = super()._initialize_infill()
        if self._oos_repair is not None:
            pop = self._oos_repair.do(self.problem, pop)
        return pop

    def _infill(self):
        off = super()._infill()
        if self._oos_repair is not None:
            off = self._oos_repair.do(self.problem, off)
        return off
