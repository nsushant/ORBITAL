import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from loaders import snap_cost


class OOSProblem(Problem):
    """
    Flat-array encoding (n_var = 4 * Ndems):

      x[0:N]    visit_order           float [0, 1]
      x[N:2N]   vehicle_assignments   int   [0, maxV]   (rounded in repair/decode)
      x[2N:3N]  arrival[i]            float [0, deadline[i]]
      x[3N:4N]  depot_visits[i]       int   [0, 1]       (rounded in repair/decode)

    vehicle_assignments = 0  → demand is unserved.
    vehicle_assignments = 1..maxV → actual vehicle index.

    departure[i] = arrival[i] + service_times[i]  (no wait variable).

    All solutions are made feasible by OOSRepair before evaluation,
    so n_constr = 0.
    """

    def __init__(self, Ndems, maxV, deadlines, service_times, sat_ids,
                 depot_id, cost_table, dep_grid, arr_grid, min_tof_table, refuel_time=0.5):
        N = Ndems
        deadlines = np.asarray(deadlines, dtype=float)

        xl = np.zeros(4 * N)
        xu = np.concatenate([
            np.ones(N),                  # visit_order  [0, 1]
            np.full(N, maxV),            # vehicle_assignments [0, maxV]
            deadlines,                   # arrival[i]   [0, deadline[i]]
            np.ones(N),                  # depot_visits [0, 1]
        ])

        self.Ndems        = Ndems
        self.maxV         = maxV
        self.deadlines    = deadlines
        self.service_times = np.asarray(service_times, dtype=float)
        self.sat_ids      = np.asarray(sat_ids, dtype=int)
        self.depot_id     = int(depot_id)
        self.cost_table   = cost_table
        self.dep_grid     = np.asarray(dep_grid, dtype=float)
        self.arr_grid     = np.asarray(arr_grid, dtype=float)
        self.min_tof_table = min_tof_table
        self.refuel_time  = float(refuel_time)

        super().__init__(n_var=4*N, n_obj=3, n_constr=0, xl=xl, xu=xu)

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair_individual(self, x):
        """
        Walk each vehicle's sorted tour and clamp arrivals up to satisfy
        min-tof ordering. Demands that cannot be feasibly reached are
        marked unserved (vehicle_assignment = 0).

        Modifies x in-place and returns it.
        """
        N             = self.Ndems
        deadlines     = self.deadlines
        service_times = self.service_times
        sat_ids       = self.sat_ids
        depot_id      = self.depot_id
        min_tof_table = self.min_tof_table
        refuel_time   = self.refuel_time

        visit_order         = x[0:N]
        vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
        arrivals            = x[2*N:3*N].copy()
        depot_visits        = np.clip(np.round(x[3*N:4*N]), 0, 1).astype(int)

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle_assignments == v)[0]
            if len(indices) == 0:
                continue

            sorted_local = np.argsort(visit_order[indices])
            sorted_idx   = indices[sorted_local]
            depot_vis    = depot_visits[sorted_idx].copy()
            depot_vis[-1] = 1   # circular wrap: last demand always has depot-return flag

            # Track the last successfully served stop so dropped demands don't
            # corrupt the arrival time of the next valid demand.
            last_dep_time    = 0.0        # departure time of last valid stop
            last_sat_idx     = depot_id  # satellite index of last valid stop
            last_depot_vis   = False     # whether a depot visit follows the last valid stop

            for k, i in enumerate(sorted_idx):
                # Compute minimum feasible arrival from the last valid position
                if last_depot_vis:
                    key_to_dep   = (last_sat_idx, depot_id)
                    key_from_dep = (depot_id, sat_ids[i])
                    tof_to_dep   = min_tof_table[key_to_dep][0]   if key_to_dep   in min_tof_table else 0.0
                    tof_from_dep = min_tof_table[key_from_dep][0] if key_from_dep in min_tof_table else 0.0
                    arr_depot    = last_dep_time + tof_to_dep
                    dep_depot    = arr_depot + refuel_time
                    min_arr      = dep_depot + tof_from_dep
                else:
                    key     = (last_sat_idx, sat_ids[i])
                    min_tof = min_tof_table[key][0] if key in min_tof_table else 0.0
                    min_arr = last_dep_time + min_tof

                arrivals[i] = max(arrivals[i], min_arr)

                # Cost-table feasibility: find the earliest arr grid point at or
                # after the timing-feasible arrival that has a valid entry.
                leg_from = depot_id      if last_depot_vis else last_sat_idx
                leg_dep  = dep_depot     if last_depot_vis else last_dep_time
                dep_grid = self.dep_grid
                arr_grid = self.arr_grid
                dep_s = float(dep_grid[np.searchsorted(dep_grid, leg_dep).clip(0, len(dep_grid) - 1)])
                arr_start_idx = np.searchsorted(arr_grid, arrivals[i]).clip(0, len(arr_grid) - 1)
                found = False
                for ai in range(int(arr_start_idx), len(arr_grid)):
                    arr_s = float(arr_grid[ai])
                    if arr_s > deadlines[i]:
                        break
                    if (int(leg_from), int(sat_ids[i]), dep_s, arr_s) in self.cost_table:
                        arrivals[i] = arr_s
                        found = True
                        break

                if not found or arrivals[i] + service_times[i] > deadlines[i]:
                    vehicle_assignments[i] = 0
                    arrivals[i]  = 0.0
                    depot_vis[k] = 0
                else:
                    dep_i = arrivals[i] + service_times[i]
                    last_dep_time  = dep_i
                    last_sat_idx   = sat_ids[i]
                    last_depot_vis = bool(depot_vis[k])

            # Write back depot_vis flags
            for k, i in enumerate(sorted_idx):
                depot_visits[i] = depot_vis[k]

        # Write back to x
        x[N:2*N]   = vehicle_assignments.astype(float)
        x[2*N:3*N] = arrivals
        x[3*N:4*N] = depot_visits.astype(float)
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

        visit_order         = x[0:N]
        vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
        arrivals            = x[2*N:3*N].copy()
        depot_visits        = np.clip(np.round(x[3*N:4*N]), 0, 1).astype(int)

        departures   = np.zeros(N)
        sorted_tours = []
        legs         = []

        for v in range(1, self.maxV + 1):
            indices = np.where(vehicle_assignments == v)[0]
            if len(indices) == 0:
                sorted_tours.append([])
                continue

            sorted_local = np.argsort(visit_order[indices])
            sorted_idx   = indices[sorted_local]
            depot_vis    = depot_visits[sorted_idx].copy()
            depot_vis[-1] = 1

            sorted_tours.append(sorted_idx.tolist())

            for k, i in enumerate(sorted_idx):
                departures[i] = arrivals[i] + service_times[i]

                if k == 0:
                    legs.append((depot_id, sat_ids[i], 0.0, arrivals[i]))
                else:
                    prev = sorted_idx[k - 1]
                    if depot_vis[k - 1]:
                        key_to_dep   = (sat_ids[prev], depot_id)
                        key_from_dep = (depot_id, sat_ids[i])
                        tof_to_dep   = min_tof_table[key_to_dep][0]   if key_to_dep   in min_tof_table else 0.0
                        tof_from_dep = min_tof_table[key_from_dep][0] if key_from_dep in min_tof_table else 0.0
                        arr_depot    = departures[prev] + tof_to_dep
                        dep_depot    = arr_depot + refuel_time
                        legs.append((sat_ids[prev], depot_id, departures[prev], arr_depot))
                        legs.append((depot_id, sat_ids[i], dep_depot, arrivals[i]))
                    else:
                        legs.append((sat_ids[prev], sat_ids[i], departures[prev], arrivals[i]))

        return arrivals, departures, sorted_tours, legs

    # ------------------------------------------------------------------
    # ΔV cost
    # ------------------------------------------------------------------

    # Large but finite penalty for legs that fall outside valid cost-table windows.
    # Sat-to-sat pairs have ~73% empty grid cells (geometrically infeasible windows);
    # using a penalty instead of inf keeps ΔV finite so pymoo's dominance sort works.
    _INFEASIBLE_LEG_COST = 1e6  # m/s — well above any realistic transfer (~20k max)

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
        """
        X   : ndarray (pop_size, 4*Ndems)
        out : dict with F (pop×3); no G since repair guarantees feasibility.

        Objectives
        ──────────
        F[:,0]  total ΔV  [m/s]
        F[:,1]  number of vehicles used
        F[:,2]  unserved time  [days]
        """
        N        = self.Ndems
        pop_size = len(X)
        F        = np.zeros((pop_size, self.n_obj))

        for idx in range(pop_size):
            x = X[idx]
            arrivals, departures, sorted_tours, legs = self.decode(x)

            vehicle_assignments = np.clip(np.round(x[N:2*N]), 0, self.maxV).astype(int)
            served_mask = vehicle_assignments > 0

            F[idx, 0] = self.compute_dv(legs)
            F[idx, 1] = float(len(np.unique(vehicle_assignments[served_mask]))) if served_mask.any() else 0.0
            F[idx, 2] = float(np.sum(self.service_times[~served_mask]))

        out["F"] = F


class OOSRepair(Repair):
    """pymoo Repair wrapper that calls OOSProblem.repair_individual."""

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = problem.repair_individual(X[i].copy())
        return X
