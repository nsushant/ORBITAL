"""Non-dominated archive over the three objectives.

MDLS keeps every non-dominated solution it has found, samples from that archive
to seed each iteration, and returns it as the Pareto front. The archive has to
hold the schedule alongside its objective vector, because the depot study needs
to know *who* each front point serves, not merely what it costs.

Dominance is minimisation on all three: delta-V, active vehicles, unrecovered
value. A solution enters if nothing already held dominates it, and everything it
dominates is evicted.

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
from numba import njit


@njit(cache=True, inline="always")
def dominates(a, b):
    """True when `a` is at least as good on every objective and better on one."""
    at_least_as_good = True
    strictly_better = False
    for k in range(a.shape[0]):
        if a[k] > b[k]:
            at_least_as_good = False
            break
        if a[k] < b[k]:
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
        # An exact duplicate adds nothing and would let the archive fill with
        # copies of whatever the operators happen to rediscover.
        same = True
        for k in range(candidate.shape[0]):
            if front[i, k] != candidate[k]:
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
    return bool(np.all(a <= b) and np.any(a < b))
