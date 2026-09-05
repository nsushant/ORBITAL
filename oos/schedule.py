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

import os

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
    reachable: np.ndarray = None   # (n, n) bool, or None; see `reach`

    @property
    def n_nodes(self):
        return self.dv.shape[0]

    @property
    def reach(self):
        """(n, n) bool: is any leg i -> j priced at any epoch and duration.

        Read from a sidecar when one exists, computed and cached here
        otherwise, so an operator can screen a proposed leg without touching
        the table.
        """
        if self.reachable is None:
            self.reachable = np.isfinite(self.dv).any(axis=(2, 3))
        return self.reachable


def _npy_sidecars(path):
    """(dv, phasing, reachable) sidecar paths beside an HDF5 cost table."""
    stem = path[:-3] if path.endswith(".h5") else path
    return stem + ".dv.npy", stem + ".phasing.npy", stem + ".reach.npy"


def export_npy(path="outputs/cost_table.h5"):
    """Write the two large arrays beside the table as .npy, once.

    The table is about 2.6 GB and every process that opens it reads the whole
    thing into its own memory. That is tolerable for one run and expensive for
    a tuning race, where each of a thousand experiments is a fresh process:
    the read is paid a thousand times, and running four experiments at once
    costs four copies, which is what caps the parallelism.

    A .npy sidecar can be memory-mapped instead. Every process then shares one
    copy through the operating system's page cache, the read disappears, and
    the worker count stops being a memory question.
    """
    import h5py
    dv_path, ph_path, rc_path = _npy_sidecars(path)
    with h5py.File(path, "r") as f:
        n = f["dv"].shape[0]
        # Which ordered pairs have a priced transfer at all, at any epoch and
        # any duration: a static property of the table, and the cheapest
        # possible first question for an operator proposing a new leg. On the
        # study instance it answers "no" for 86 % of the insertions 2-regret
        # would otherwise have to price (F45). Seconds to build, one byte to
        # ask.
        reach = np.zeros((n, n), dtype=bool)
        for i in range(0, n, 16):
            reach[i:i + 16] = np.isfinite(f["dv"][i:i + 16]).any(axis=(2, 3))
        np.save(rc_path, reach)
        for name, out in (("dv", dv_path), ("phasing_days", ph_path)):
            a = f[name]
            m = np.lib.format.open_memmap(out, mode="w+", dtype=a.dtype,
                                          shape=a.shape)
            # copied in slabs so the export itself does not need 1.3 GB free
            for i in range(0, a.shape[0], 16):
                m[i:i + 16] = a[i:i + 16]
            m.flush()
            del m
    return dv_path, ph_path, rc_path


def load_cost_table(path="outputs/cost_table.h5", mmap=True):
    """Load the table, memory-mapping the two large arrays when they exist.

    `mmap=True` uses the .npy sidecars written by `export_npy` if both are
    present and current, and falls back to reading the HDF5 file otherwise, so
    this is a pure speed and memory change with no new required build step.
    """
    import h5py

    dv = phasing = reach = None
    if mmap:
        dv_path, ph_path, rc_path = _npy_sidecars(path)
        if (os.path.exists(dv_path) and os.path.exists(ph_path)
                and os.path.getmtime(dv_path) >= os.path.getmtime(path)
                and os.path.getmtime(ph_path) >= os.path.getmtime(path)):
            dv = np.load(dv_path, mmap_mode="r")
            phasing = np.load(ph_path, mmap_mode="r")
            if (os.path.exists(rc_path)
                    and os.path.getmtime(rc_path) >= os.path.getmtime(path)):
                reach = np.load(rc_path, mmap_mode="r")

    with h5py.File(path, "r") as f:
        if dv is None:
            dv, phasing = f["dv"][:], f["phasing_days"][:]
        return CostTable(
            dv=dv, phasing=phasing,
            names=[n.decode() if isinstance(n, bytes) else n
                   for n in f["names"][:]],
            dep_days=f["departure_days"][:], tof_days=f["tof_days"][:],
            plane_of_node=(f["plane_of_node"][:] if "plane_of_node" in f
                           else np.full(dv.shape[0], -1)),
            dv_budget=float(f.attrs["dv_budget_m_s"]),
            reachable=reach)


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
    # A leg may not land before the vehicle set out. snap_departure clamps to
    # the last departure epoch, so once a vehicle's clock runs past the end of
    # the grid every lookup returns an arrival drawn from that final epoch --
    # earlier, in wall-clock terms, than where the vehicle actually is. Left
    # unguarded that reads as a perfectly ordinary cheap leg, and a schedule
    # built from those travels backwards in time while evaluate() reports it
    # feasible: the paper's chronological constraint (Sec. 2, a_i^v >= b_i-1^v)
    # is the one constraint g1/g2/g3 do not express. Refusing the leg here
    # enforces it by construction for every caller -- MDLS, the GA decoder and
    # the constructive heuristic alike -- and reports it as what it physically
    # is, an absent leg (g1), rather than as a fourth kind of violation.
    if arrival < depart_day:
        return 0.0, 0.0, False
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
