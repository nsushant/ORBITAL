"""Solution representation and the one objective function every algorithm uses.

This is the shared contract. MDLS, NSGA-III and the particle swarm all score
solutions here, which is the point: the published comparison was confounded
because they did not. Three differences had crept in --- the two stacks snapped
epochs to the cost-table grid differently, they disagreed on what an unusable
leg costs, and NSGA-III had no constraints at all because a repair operator made
every solution feasible before evaluation. Any of those alone makes the
objectives incomparable.

The three objectives, from Eq. (objective) of the paper
-------------------------------------------------------
    f1  cumulative transfer delta-V over every leg flown          [m/s]
    f2  active fleet size, vehicles that serve at least one client
    f3  unrecovered value: the summed recovery potential of the
        demands left unserved                                     [$]

f3 is money, not a count and not a duration. The Julia computed it as
`sum(get(new_u, "asset_values", new_u["service_times"]))`, which silently
becomes summed service time in days when the demand file carries no asset
values --- a different objective in different units, six orders of magnitude
smaller. There is no fallback here: a demand without a value is an error.

The three constraints, from D19
-------------------------------
Leaving a demand unassigned is *not* a violation. It is a legitimate choice
that costs f3, and both families of algorithm must be able to make it. What is
a violation:

    g1  a leg with no cost-table entry was flown
    g2  arrival outside the demand's release/deadline window
    g3  delta-V spent between two depot visits exceeds the budget

Each is returned as a non-negative violation, zero when satisfied, which is
pymoo's convention.

Representation
--------------
Flat arrays with CSR offsets rather than nested objects, so the hot loop can be
compiled and so an operator can copy a schedule with a slice instead of a deep
copy. A vehicle's visits are `veh_start[v] : veh_start[v + 1]`, and every visit
carries the leg that *arrives* at it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

DEPOT_UID = -1


# ---------------------------------------------------------------------------
# Cost table


@dataclass
class CostTable:
    """The pairwise table, plus the epoch grid it is defined on."""

    dv: np.ndarray            # (n, n, n_dep, n_tof) m/s, NaN where infeasible
    phasing: np.ndarray       # (n, n, n_dep, n_tof) days
    names: list
    dep_days: np.ndarray
    tof_days: np.ndarray
    plane_of_node: np.ndarray
    dv_budget: float          # m/s

    @property
    def n_nodes(self):
        return self.dv.shape[0]


def load_cost_table(path="outputs/cost_table.h5"):
    import h5py

    with h5py.File(path, "r") as f:
        return CostTable(
            dv=f["dv"][:], phasing=f["phasing_days"][:],
            names=[n.decode() if isinstance(n, bytes) else n
                   for n in f["names"][:]],
            dep_days=f["departure_days"][:], tof_days=f["tof_days"][:],
            plane_of_node=(f["plane_of_node"][:] if "plane_of_node" in f
                           else np.full(f["dv"].shape[0], -1)),
            dv_budget=float(f.attrs["dv_budget_m_s"]))


@njit(cache=True, inline="always")
def snap_departure(grid, t):
    """Nearest grid point to a departure epoch (D20)."""
    n = grid.shape[0]
    if t <= grid[0]:
        return 0
    if t >= grid[n - 1]:
        return n - 1
    k = 0
    while k < n - 1 and grid[k + 1] < t:
        k += 1
    return k if (t - grid[k]) <= (grid[k + 1] - t) else k + 1


@njit(cache=True, inline="always")
def snap_tof(grid, t):
    """Smallest grid time of flight not shorter than `t` (D20).

    Rounding the flight time *up* means the schedule never claims an arrival
    earlier than the table supports, so the discretisation error is asymmetric
    in the conservative direction. Returns -1 when the horizon cannot cover it.
    """
    n = grid.shape[0]
    for k in range(n):
        if grid[k] >= t:
            return k
    return -1


@njit(cache=True)
def leg(dv_tab, ph_tab, dep_grid, tof_grid, i, j, depart_day, tof_day):
    """Cost and true arrival of one leg.

    Returns (delta-V [m/s], arrival [days], ok). `ok` is False when the pair has
    no table entry at that epoch, which is constraint g1 rather than a large
    number: a finite penalty summed into f1 is how the published NSGA-III-T
    fronts came to contain solutions reported as spending 2.8e8 m/s.
    """
    p = snap_departure(dep_grid, depart_day)
    q = snap_tof(tof_grid, tof_day)
    if q < 0:
        return 0.0, 0.0, False
    dv = dv_tab[i, j, p, q]
    if np.isnan(dv):
        return 0.0, 0.0, False
    # The table's phasing time is spent *after* the transfer, so it is part of
    # the journey and the caller cannot start servicing before it elapses.
    arrival = dep_grid[p] + tof_grid[q] + ph_tab[i, j, p, q]
    return dv, arrival, True


# ---------------------------------------------------------------------------
# Demands


@dataclass
class Demands:
    """One entry per service request."""

    node: np.ndarray        # index into the cost table's node list
    release: np.ndarray     # days, earliest the demand may be served
    deadline: np.ndarray    # days
    service: np.ndarray     # days on station
    value: np.ndarray       # $ recovery potential, required

    def __post_init__(self):
        if np.isnan(self.value).any() or (self.value < 0).any():
            raise ValueError(
                "every demand needs a non-negative recovery potential: f3 is "
                "unrecovered value in dollars (Eq. objective), and there is no "
                "fallback to service time. See F18/D18.")

    def __len__(self):
        return len(self.node)


# ---------------------------------------------------------------------------
# Schedules


@dataclass
class Schedule:
    """Flat vehicle schedules. Visits of vehicle v are veh_start[v:v+2]."""

    veh_start: np.ndarray   # (n_veh + 1,) int
    node: np.ndarray        # (n_visit,) node index
    uid: np.ndarray         # (n_visit,) demand id, DEPOT_UID for a depot visit
    arrival: np.ndarray     # (n_visit,) days
    departure: np.ndarray   # (n_visit,) days
    cost: np.ndarray        # (n_visit,) m/s of the leg arriving at this visit

    @property
    def n_vehicles(self):
        return len(self.veh_start) - 1

    def copy(self):
        return Schedule(self.veh_start.copy(), self.node.copy(), self.uid.copy(),
                        self.arrival.copy(), self.departure.copy(),
                        self.cost.copy())

    def served(self):
        """UIDs this schedule serves, in visit order."""
        return self.uid[self.uid != DEPOT_UID]


@njit(cache=True)
def _evaluate(veh_start, node, uid, arrival, cost,
              dem_release, dem_deadline, dem_value,
              dv_budget, n_demands):
    """Objectives and violations. See the module docstring for the contract."""
    n_veh = veh_start.shape[0] - 1

    f1 = 0.0
    f2 = 0
    g1 = 0.0        # legs with no table entry, counted
    g2 = 0.0        # total lateness and earliness, days
    g3 = 0.0        # total delta-V overspend between depot visits, m/s

    served = np.zeros(n_demands, dtype=np.bool_)

    for v in range(n_veh):
        lo, hi = veh_start[v], veh_start[v + 1]
        if hi <= lo:
            continue
        serves_a_client = False
        leg_spend = 0.0

        for k in range(lo, hi):
            c = cost[k]
            if np.isnan(c):
                # The operator flew a leg the table has no entry for.
                g1 += 1.0
            else:
                f1 += c
                leg_spend += c

            u = uid[k]
            if u == DEPOT_UID:
                # Refuelling resets the budget, so the constraint applies to
                # each depot-to-depot run separately (Eq. fuel-capacity).
                if leg_spend > dv_budget:
                    g3 += leg_spend - dv_budget
                leg_spend = 0.0
            else:
                serves_a_client = True
                served[u] = True
                a = arrival[k]
                if a < dem_release[u]:
                    g2 += dem_release[u] - a
                if a > dem_deadline[u]:
                    g2 += a - dem_deadline[u]

        if leg_spend > dv_budget:
            g3 += leg_spend - dv_budget
        if serves_a_client:
            f2 += 1

    # f3 is the value of what was *not* served. Unassigned demands are a choice,
    # not a violation -- that is what makes this an objective.
    f3 = 0.0
    for u in range(n_demands):
        if not served[u]:
            f3 += dem_value[u]

    return f1, float(f2), f3, g1, g2, g3


def evaluate(schedule, demands, dv_budget):
    """(objectives, violations) for one schedule.

    objectives: (delta-V [m/s], active vehicles, unrecovered value [$])
    violations: (missing legs, window violation [days], budget overspend [m/s]),
    each non-negative and zero when satisfied.
    """
    f1, f2, f3, g1, g2, g3 = _evaluate(
        schedule.veh_start, schedule.node, schedule.uid, schedule.arrival,
        schedule.cost, demands.release, demands.deadline, demands.value,
        float(dv_budget), len(demands))
    return np.array([f1, f2, f3]), np.array([g1, g2, g3])


def is_feasible(violations, tol=1e-9):
    return bool(np.all(np.asarray(violations) <= tol))
