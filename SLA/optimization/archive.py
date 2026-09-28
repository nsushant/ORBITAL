"""Non-dominated archive over the three objectives.

MDLS keeps every non-dominated solution it has found, samples from that archive
to seed each iteration, and returns it as the Pareto front. The archive has to
hold the schedule alongside its objective vector, because the depot study needs
to know *who* each front point serves, not merely what it costs.

Dominance is minimisation on all three: delta-V, active vehicles, unrecovered
value, compared to a relative tolerance (see EPS_REL). A solution enters if
nothing already held dominates it, and everything it dominates is evicted.

Feasibility. A solution that violates a constraint (D19) never enters. It is not
ranked below feasible ones or penalised into an objective -- that is how the
published NSGA-III-T fronts came to contain rows reporting 2.8e8 m/s of
propellant. The archive is a set of *feasible* non-dominated solutions and
nothing else.

The Julia used an octree to accelerate dominance queries. That is not reproduced
here: at three objectives a linear scan is a few hundred nanoseconds per
comparison in compiled code, the archive holds tens of thousands of points at
most, and a data structure whose only purpose is speed is a poor place to risk a
correctness bug. If profiling later says otherwise, the scan is one function.
"""

from __future__ import annotations

import numpy as np
from .compat import njit


# Objectives are compared to a relative tolerance rather than exactly.
#
# f3 is a sum of a few hundred client values totalling some $2.7e8. Two
# schedules that recover exactly the same set can produce that total in a
# different summation order and so differ in float64's last bit. Compared
# exactly, that reads as a trade-off, and the archive keeps both: one run held
# a point costing 100.6 m/s more than a starting schedule for a gain of
# $3e-8 -- three hundredths of a microcent -- which is not a compromise
# solution, it is arithmetic noise occupying a front slot (F34).
#
# 1e-9 sits far above float64's ~1e-16 and the ~1e-13 an order-dependent sum of
# a few hundred terms accumulates, and far below anything meaningful: $0.27 on
# f3, 5e-6 m/s on f1, and 2.5e-8 on a vehicle count that only moves in whole
# numbers. Absolute zero is handled because the tolerance is taken against the
# larger magnitude of the pair.
EPS_REL = 1e-9


@njit(cache=True, inline="always")
def dominates(a, b):
    """True when `a` is at least as good on every objective and better on one.

    "Better" means better by more than EPS_REL of the values being compared;
    a difference smaller than that is treated as equality.
    """
    at_least_as_good = True
    strictly_better = False
    for k in range(a.shape[0]):
        mag = abs(a[k]) if abs(a[k]) > abs(b[k]) else abs(b[k])
        tol = EPS_REL * mag
        if a[k] > b[k] + tol:
            at_least_as_good = False
            break
        if a[k] < b[k] - tol:
            strictly_better = True
    return at_least_as_good and strictly_better


@njit(cache=True)
def _insert(front, n, candidate):
    """Insert into `front[:n]`, evicting what the candidate dominates.

    Returns the new count, or -1 when the candidate was rejected. Compacting in
    place keeps the front contiguous so callers can slice it.
    """
    for i in range(n):
        if dominates(front[i], candidate):
            return -1
        # A duplicate adds nothing and would let the archive fill with copies
        # of whatever the operators happen to rediscover. Judged to the same
        # tolerance as dominance, so a point that merely re-derives an existing
        # one through a different summation order is not admitted as new.
        same = True
        for k in range(candidate.shape[0]):
            mag = (abs(front[i, k]) if abs(front[i, k]) > abs(candidate[k])
                   else abs(candidate[k]))
            if abs(front[i, k] - candidate[k]) > EPS_REL * mag:
                same = False
                break
        if same:
            return -1

    write = 0
    for i in range(n):
        if not dominates(candidate, front[i]):
            if write != i:
                for k in range(front.shape[1]):
                    front[write, k] = front[i, k]
            write += 1
    for k in range(candidate.shape[0]):
        front[write, k] = candidate[k]
    return write + 1


class Archive:
    """Feasible non-dominated solutions, with the schedule that produced each."""

    def __init__(self, n_obj=3, capacity=4096):
        self._f = np.empty((capacity, n_obj))
        self._n = 0
        self.schedules = []
        self.unassigned = []

    def __len__(self):
        return self._n

    @property
    def objectives(self):
        return self._f[:self._n]

    def _grow(self):
        bigger = np.empty((max(8, 2 * self._f.shape[0]), self._f.shape[1]))
        bigger[:self._n] = self._f[:self._n]
        self._f = bigger

    def add(self, objectives, schedule, unassigned=None):
        """Offer a solution. Returns True if it entered the front."""
        objectives = np.asarray(objectives, dtype=np.float64)
        if self._n + 1 >= self._f.shape[0]:
            self._grow()

        # Mirror the eviction on the Python-side lists, which _insert cannot see.
        keep = [i for i in range(self._n)
                if not _dominates_py(objectives, self._f[i])]
        new_n = _insert(self._f, self._n, objectives)
        if new_n < 0:
            return False
        self.schedules = [self.schedules[i] for i in keep] + [schedule]
        self.unassigned = [self.unassigned[i] for i in keep] + [unassigned]
        self._n = new_n
        assert len(self.schedules) == self._n, "front and schedules out of step"
        return True

    def sample(self, rng):
        """One index, uniformly. MDLS seeds each iteration from the front."""
        return int(rng.integers(self._n))

    def knee(self, nadir):
        """Index of the point closest to the ideal, normalised against `nadir`.

        The compromise solution the paper reports. The ideal is taken from the
        front itself and the nadir is supplied, so the measure does not drift as
        the front grows.
        """
        f = self.objectives
        ideal = f.min(axis=0)
        span = np.maximum(np.asarray(nadir, float) - ideal, 1e-10)
        return int(np.argmin(np.linalg.norm((f - ideal) / span, axis=1)))

    def max_coverage(self):
        """Index of the point serving the most client value (D13a).

        The column each depot location contributes to the selection MILP: f3 is
        unrecovered value, so the least of it is the most served.
        """
        return int(np.argmin(self.objectives[:, 2]))


def _dominates_py(a, b):
    """Python mirror of `dominates`. The two must agree exactly: `add` uses
    this to evict from the Python-side lists and `_insert` to evict from the
    objective array, and a disagreement desynchronises the front from its
    schedules."""
    tol = EPS_REL * np.maximum(np.abs(a), np.abs(b))
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


class SearchArchive:
    """Deduplicated trace of every schedule evaluated by MDLS.

    Unlike :class:`Archive`, this retains dominated and near-feasible points.
    It is not sampled by the search; it exists for audit and counterfactual
    queries after the primary Pareto decision has been made.
    """

    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self._seen = set()
        self.records = []

    @staticmethod
    def _key(schedule):
        # Arrival is snapped to the transfer grid; microsecond-level differences
        # are irrelevant and would otherwise defeat deduplication.
        arrival = np.round(np.asarray(schedule.arrival, dtype=float), 3)
        return (np.asarray(schedule.veh_start, dtype=np.int64).tobytes(),
                np.asarray(schedule.uid, dtype=np.int64).tobytes(),
                np.asarray(schedule.node, dtype=np.int64).tobytes(),
                arrival.tobytes())

    def add(self, objectives, violations, schedule, unassigned=None,
            iteration=-1, operator="unknown"):
        key = self._key(schedule)
        if key in self._seen or len(self.records) >= self.capacity:
            return False
        self._seen.add(key)
        self.records.append({
            "objectives": np.asarray(objectives, dtype=float).copy(),
            "violations": np.asarray(violations, dtype=float).copy(),
            "schedule": schedule,
            "unassigned": np.asarray(unassigned if unassigned is not None else [], dtype=np.int64).copy(),
            "iteration": int(iteration),
            "operator": str(operator),
        })
        return True

    def summaries(self, demand_ids=None):
        ids = list(demand_ids or [])
        out = []
        for record in self.records:
            schedule = record["schedule"]
            served = sorted(int(x) for x in schedule.served())
            out.append({
                "objectives": record["objectives"].tolist(),
                "violations": record["violations"].tolist(),
                "feasible": is_feasible_vector(record["violations"]),
                "served": [ids[i] if i < len(ids) else i for i in served],
                "unassigned": [ids[i] if i < len(ids) else i for i in record["unassigned"]],
                "vehicles": int(schedule.n_vehicles),
                "iteration": record["iteration"],
                "operator": record["operator"],
            })
        return out


def is_feasible_vector(violations, tol=1e-9):
    return bool(np.all(np.asarray(violations) <= tol))
