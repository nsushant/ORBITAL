"""Multi-Directional Local Search (Tricoire 2012) with user-supplied moves.

A problem defines two things:

    evaluate(x, instance) -> (f, g) or f     # every objective is minimised
    move(x, instance, rng) -> x or None      # one improving attempt, or None

Moves are tagged with the objective they improve and grouped by it; each
iteration runs one sampled move per objective. A move may retry its attempt up
to `attempts` times to find an improvement. The class owns the archive,
selection, budgets and statistics.
"""

from __future__ import annotations

import copy as _copy
import functools
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ._dominance import _Forest

EPS_REL = 1e-9  # relative tolerance: smaller differences count as equal

# Front size at which Archive switches from the linear scan to the dominance
# index.  Below this the scan is cheaper than maintaining the index; the value
# is read at call time so tests can override it.
INDEX_MIN = 8192

logger = logging.getLogger(__name__)


def move(improves, attempts=1):
    """Tag a move with the objective it improves; call the result to bind parameters.

    The decorated function is one attempt: it returns a candidate that strictly
    improves objective `improves`, or None. The decorator retries the attempt up
    to `attempts` times and returns the first improving candidate. An attempt
    that returns None must leave its argument unchanged.

    @move(improves=0, attempts=4)
    def drop_add(x, instance, rng, n_drop=3): ...

    moves = [drop_add(n_drop=5)]
    """
    if not isinstance(attempts, (int, np.integer)) or isinstance(attempts, bool) or int(attempts) < 1:
        raise ValueError(f"attempts must be a positive integer, got {attempts!r}")
    if not isinstance(improves, (int, np.integer)) or isinstance(improves, bool) or int(improves) < 0:
        raise ValueError(f"improves must be a non-negative integer, got {improves!r}")
    attempts, improves = int(attempts), int(improves)

    def decorate(fn):
        @functools.wraps(fn)
        def make(**params):
            step = functools.partial(fn, **params)

            @functools.wraps(fn)
            def move_fn(x, instance, rng):
                for _ in range(attempts):
                    y = step(x, instance, rng)
                    if y is not None:
                        return y
                return None

            move_fn.improves = improves
            move_fn.attempts = attempts
            return move_fn

        make.improves = improves
        make.attempts = attempts
        make.__move_factory__ = True
        return make
    return decorate


def dominates(a, b, eps_rel=EPS_REL):
    """True when a is no worse than b everywhere and better somewhere (minimisation)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    tol = eps_rel * np.maximum(np.abs(a), np.abs(b))
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


@njit(cache=True)
def _archive_insert(front, point, eps_rel, keep):
    """Dominance scan over a preallocated front.

    Returns (accepted, k). `keep[:len(front)]` marks the points that survive;
    the new point itself is inserted by the Python wrapper.
    """
    n = front.shape[0]
    for i in range(n):
        no_worse = True
        for j in range(point.shape[0]):
            tol = eps_rel * max(abs(front[i, j]), abs(point[j]))
            if front[i, j] > point[j] + tol:
                no_worse = False
                break
        if no_worse:
            return False, 0  # dominated by, or duplicate of, a held point

    k = 0
    for i in range(n):
        point_no_worse, point_better = True, False
        for j in range(point.shape[0]):
            tol = eps_rel * max(abs(front[i, j]), abs(point[j]))
            if point[j] > front[i, j] + tol:
                point_no_worse = False
                break
            if point[j] < front[i, j] - tol:
                point_better = True
        keep[i] = not (point_no_worse and point_better)
        if keep[i]:
            k += 1
    return True, k


class Archive:
    """Feasible non-dominated solutions, each stored with its objective vector."""

    def __init__(self, n_obj, capacity=1024, eps_rel=EPS_REL):
        if not isinstance(n_obj, (int, np.integer)) or isinstance(n_obj, bool) or int(n_obj) < 1:
            raise ValueError(f"n_obj must be a positive integer, got {n_obj!r}")
        if not isinstance(capacity, (int, np.integer)) or isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        eps_rel = float(eps_rel)
        if not np.isfinite(eps_rel) or eps_rel < 0:
            raise ValueError(f"eps_rel must be finite and non-negative, got {eps_rel!r}")
        self.n_obj, self.eps_rel = int(n_obj), eps_rel
        self._f = np.empty((int(capacity), self.n_obj))
        self._keep = np.empty(int(capacity), dtype=np.bool_)
        self._n = 0
        self.solutions = []
        self._index = None          # _Forest, built lazily at INDEX_MIN
        self._index_enabled = True  # cleared if evictions make the index a liability
        self._rebuilds = 0
        self._indexed_inserts = 0

    def __len__(self):
        return self._n

    @property
    def objectives(self):
        """(n, n_obj) view of the front; read only."""
        return self._f[:self._n]

    def add(self, objectives, solution):
        """Offer a feasible solution; evicts what it dominates.
        Returns True if it entered, False if dominated or a duplicate."""
        f = np.asarray(objectives, float).reshape(-1)
        if f.shape[0] != self.n_obj:
            raise ValueError(f"expected {self.n_obj} objectives, got {f.shape[0]}")
        if not np.all(np.isfinite(f)):
            raise ValueError(f"non-finite objectives {f}; reject via constraints instead")
        if self._n == self._f.shape[0]:
            self._grow()

        index = self._index
        if index is None and self._index_enabled and self._n >= INDEX_MIN:
            index = self._index = _Forest.build(self._f[:self._n])
        if index is None:
            return self._add_scan(f, solution)
        return self._add_indexed(index, f, solution)

    def _add_scan(self, f, solution):
        """Linear-scan path: unchanged dominance semantics."""
        n = self._n
        accepted, k = _archive_insert(self._f[:n], f, self.eps_rel, self._keep[:n])
        if not accepted:
            return False
        if k == n:  # nothing evicted: append in place instead of compacting
            self._f[n] = f
            self.solutions.append(solution)
            self._n = n + 1
            return True
        survivors = self._keep[:n]
        self._f[:k] = self._f[:n][survivors]
        self._f[k] = f
        self.solutions = [s for s, kept in zip(self.solutions, survivors) if kept] + [solution]
        self._n = k + 1
        return True

    def _add_indexed(self, index, f, solution):
        """Indexed path: victim-first, as MDLS specifies."""
        n = self._n
        victims = index.victims(f, self.eps_rel)
        if len(victims):
            # With a tolerance, dominance is not transitive, so the second query
            # cannot be skipped: run it before mutating anything.
            if self.eps_rel > 0.0 and index.blocked(f, self.eps_rel):
                return False
            mask = np.ones(n, dtype=np.bool_)
            mask[victims] = False
            k = n - len(victims)
            self._f[:k] = self._f[:n][mask]
            self._f[k] = f
            self.solutions = [s for s, kept in zip(self.solutions, mask) if kept] + [solution]
            self._n = k + 1
            self._index = _Forest.build(self._f[:self._n])
            self._rebuilds += 1
            if self._rebuilds > 8 and self._rebuilds > 0.02 * max(self._indexed_inserts, 1):
                self._index, self._index_enabled = None, False
            return True
        if index.blocked(f, self.eps_rel):
            return False
        self._f[n] = f
        self.solutions.append(solution)
        self._n = n + 1
        index.add(f, n)
        self._indexed_inserts += 1
        return True

    def _grow(self):
        capacity = 2 * self._f.shape[0]
        grown = np.empty((capacity, self.n_obj))
        grown[:self._n] = self._f[:self._n]
        self._f = grown
        self._keep = np.empty(capacity, dtype=np.bool_)

    def knee(self, nadir=None):
        """Index of the point nearest the ideal after normalising by (nadir - ideal).
        Pass a fixed nadir when comparing runs."""
        f = self.objectives
        ideal = f.min(axis=0)
        nad = f.max(axis=0) if nadir is None else np.asarray(nadir, float)
        return int(np.argmin(np.linalg.norm((f - ideal) / np.maximum(nad - ideal, 1e-10), axis=1)))

    def extreme(self, k):
        """Index of the best point on objective k."""
        return int(np.argmin(self.objectives[:, k]))


@dataclass
class MoveStats:
    """Per-move counters for one run."""
    objective: int = -1
    calls: int = 0
    no_move: int = 0
    not_descent: int = 0
    infeasible: int = 0
    dominated: int = 0
    accepted: int = 0
    seconds: float = 0.0

    @property
    def acceptance(self):
        return self.accepted / self.calls if self.calls else 0.0


@dataclass
class Result:
    """Archive plus run accounting returned by MDLS.run."""
    archive: Archive
    n_eval: int
    n_iter: int
    elapsed: float
    stopped_by: str
    stats: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @property
    def objectives(self):
        return self.archive.objectives

    @property
    def solutions(self):
        return self.archive.solutions

    def summary(self):
        """Multi-line text report of the run and each move."""
        out = [f"MDLS: {self.n_iter} iterations, {self.n_eval} evaluations, {self.elapsed:.2f} s, "
               f"stopped by {self.stopped_by}; front holds {len(self.archive)}",
               f"  {'move':24s} {'obj':>3s} {'calls':>7s} {'no-move':>8s} {'no-desc':>7s} {'infeas':>7s} "
               f"{'dom':>7s} {'accepted':>8s} {'acc %':>6s} {'sec':>6s}"]
        for label, s in self.stats.items():
            out.append(f"  {label:24s} {s.objective:3d} {s.calls:7d} {s.no_move:8d} {s.not_descent:7d} "
                       f"{s.infeasible:7d} {s.dominated:7d} {s.accepted:8d} {100 * s.acceptance:6.1f} "
                       f"{s.seconds:6.2f}")
        return "\n".join(out)


def all_feasible(g):
    """Default feasibility rule: every constraint value g <= 0 (None means unconstrained)."""
    return g is None or np.size(g) == 0 or bool(np.all(np.asarray(g, float) <= 0.0))


class MDLS:
    """Archive, selection, loop and budgets; the moves are passed in.

    evaluate(x, instance) -> (f, g) or f
    move(x, instance, rng) -> solution or None, tagged with @move(improves=k, attempts=n)

    A move is one improving attempt, retried up to `attempts` times by its
    decorator. Override select / offer / on_iteration to change one step.
    """

    def __init__(self, evaluate, moves, instance=None, is_feasible=None,
                 selection=None, eps_rel=EPS_REL, verbose=0, copy_solution=None):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        if is_feasible is not None and not callable(is_feasible):
            raise TypeError("is_feasible must be callable or None")
        if selection is not None and not callable(selection):
            raise TypeError("selection must be callable or None")
        if copy_solution is not None and not callable(copy_solution):
            raise TypeError("copy_solution must be callable or None")
        eps_rel = float(eps_rel)
        if not np.isfinite(eps_rel) or eps_rel < 0:
            raise ValueError(f"eps_rel must be finite and non-negative, got {eps_rel!r}")
        if not isinstance(verbose, (int, np.integer)) or isinstance(verbose, bool) or int(verbose) < 0:
            raise ValueError(f"verbose must be a non-negative integer, got {verbose!r}")
        self.evaluate_fn, self.instance = evaluate, instance
        self.pools = self._index_moves(moves)
        if not self.pools:
            raise ValueError("MDLS needs at least one move")
        self._is_feasible = is_feasible if is_feasible is not None else all_feasible
        self._selection = selection
        self._copy_solution_fn = copy_solution
        self.eps_rel, self.verbose = eps_rel, int(verbose)

    @staticmethod
    def _label(mv):
        """Name a move for the stats table; partials are named after their function."""
        if isinstance(mv, functools.partial):
            mv = mv.func
        return getattr(mv, "__name__", type(mv).__name__)

    def _index_moves(self, moves):
        """Group tagged moves by objective: {k: [(label, move), ...]}."""
        pools, seen = {}, {}
        for mv in moves:
            if getattr(mv, "__move_factory__", False):
                name = getattr(mv, "__name__", "move")
                raise ValueError(f"{name} is a move factory; call it first, e.g. {name}(...)")
            tag = getattr(mv, "improves", None)
            if tag is None:
                raise ValueError(f"{self._label(mv)!r} is not tagged; define it with @move(improves=k)")
            if not isinstance(tag, (int, np.integer)) or isinstance(tag, bool):
                raise ValueError(f"{self._label(mv)!r} has a non-integer objective tag {tag!r}")
            mi = getattr(mv, "attempts", 1)
            if not isinstance(mi, (int, np.integer)) or isinstance(mi, bool) or int(mi) < 1:
                raise ValueError(f"{self._label(mv)!r} has invalid attempts {mi!r}")
            base = self._label(mv)
            seen[base] = seen.get(base, 0) + 1
            label = base if seen[base] == 1 else f"{base}#{seen[base]}"
            pools.setdefault(int(tag), []).append((label, mv))
        return dict(sorted(pools.items()))

    def _check_pools(self, n_obj):
        """Refuse tags for missing objectives and require one move per objective."""
        bad = [k for k in self.pools if not 0 <= k < n_obj]
        if bad:
            raise ValueError(f"moves tagged for objective(s) {bad}, problem has {n_obj}")
        missing = [k for k in range(n_obj) if k not in self.pools]
        if missing:
            raise ValueError(f"no move tagged for objective(s) {missing}; add @move(improves=k)")

    def evaluate(self, x):
        """Call the user's evaluate and normalise to (f, g).
        A tuple whose first item is a vector is read as (f, g)."""
        out = self.evaluate_fn(x, self.instance)
        if isinstance(out, tuple) and len(out) == 2:
            if np.ndim(out[0]) >= 1:
                f, g = out
            elif np.ndim(out[1]) >= 1:
                raise ValueError(
                    "evaluate returned a scalar objective with a vector constraint; "
                    "wrap f in a list or array, e.g. return [f], g")
            else:
                f, g = out, None
        else:
            f, g = out, None
        return np.asarray(f, float).reshape(-1), g

    def select(self, archive, rng):
        """Index of the archive member to search from; uniform by default."""
        if self._selection is not None:
            return int(self._selection(archive, rng))
        return int(rng.integers(len(archive)))

    def offer(self, f, x, archive):
        """Offer a feasible candidate to the archive; True if it entered."""
        return archive.add(f, x)

    def copy(self, solution):
        """Return a private copy of an archived solution for a move to mutate.

        Uses `copy_solution` from the constructor when given. Otherwise NumPy
        arrays are copied with `.copy()` and every other object is deep-copied,
        so a move cannot mutate the archive through a nested structure.
        Override this method to change the copy rule for a whole subclass.
        """
        if self._copy_solution_fn is not None:
            return self._copy_solution_fn(solution)
        if isinstance(solution, np.ndarray):
            return solution.copy()
        return _copy.deepcopy(solution)

    def _is_descent(self, f, base_f, k):
        """True when f strictly improves objective k over base_f, within eps."""
        tol = self.eps_rel * max(abs(f[k]), abs(base_f[k]))
        return bool(f[k] < base_f[k] - tol)

    def on_iteration(self, iteration, n_eval, archive):
        """Hook after each iteration; logs progress every `verbose` iterations."""
        if self.verbose and (iteration + 1) % self.verbose == 0:
            fr = archive.objectives
            span = "  ".join(f"f{k} {fr[:, k].min():.4g}..{fr[:, k].max():.4g}"
                             for k in range(fr.shape[1]))
            logger.info("  iter %6d  eval %7d  archive %5d  %s",
                        iteration + 1, n_eval, len(archive), span)

    def run(self, init, max_iter=None, max_eval=None, time_limit=None, seed=0, record_every=0):
        """Search from init until a budget is spent.

        init is one solution, a sequence of solutions, or a callable
        init(rng) -> solutions. A flat list or tuple of scalars is treated as
        one vector solution; wrap it in a list to pass several scalar solutions.
        max_eval counts initial solutions too; at least one budget is required.
        """
        if max_iter is None and max_eval is None and time_limit is None:
            raise ValueError("give at least one of max_iter, max_eval, time_limit")
        for name, value in (("max_iter", max_iter), ("max_eval", max_eval)):
            if value is not None and (not isinstance(value, (int, np.integer))
                                      or isinstance(value, bool) or int(value) < 1):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        if time_limit is not None:
            if isinstance(time_limit, bool):
                raise ValueError(f"time_limit must be positive and finite or None, got {time_limit!r}")
            try:
                time_limit = float(time_limit)
            except (TypeError, ValueError):
                raise ValueError(f"time_limit must be positive and finite or None, got {time_limit!r}") from None
            if not np.isfinite(time_limit) or time_limit <= 0:
                raise ValueError(f"time_limit must be positive and finite or None, got {time_limit!r}")
        if not isinstance(record_every, (int, np.integer)) or isinstance(record_every, bool) or record_every < 0:
            raise ValueError(f"record_every must be a non-negative integer, got {record_every!r}")

        rng, t0 = np.random.default_rng(seed), time.time()
        archive, n_eval = self._start(init, rng, max_eval)
        stats = {label: MoveStats(objective=objective)
                 for objective, entries in self.pools.items() for label, _ in entries}
        history, iteration, stopped_by = [], 0, "max_iter"

        def out_of_evals():
            return max_eval is not None and n_eval >= max_eval

        while True:
            if max_iter is not None and iteration >= max_iter:
                stopped_by = "max_iter"
                break
            if out_of_evals():
                stopped_by = "max_eval"
                break
            if time_limit is not None and time.time() - t0 > time_limit:
                stopped_by = "time_limit"
                break

            idx = self.select(archive, rng)
            base = archive.solutions[idx]
            base_f = archive.objectives[idx].copy()
            for entries in self.pools.values():  # one sampled move per objective
                if out_of_evals():
                    break
                label, mv = entries[int(rng.integers(len(entries)))]
                st = stats[label]
                st.calls += 1
                started = time.perf_counter()
                candidate = mv(self.copy(base), self.instance, rng)
                st.seconds += time.perf_counter() - started
                if candidate is None:
                    st.no_move += 1
                    continue
                f, g = self.evaluate(candidate)
                n_eval += 1
                if not self._is_feasible(g):
                    st.infeasible += 1
                elif not self._is_descent(f, base_f, mv.improves):
                    st.not_descent += 1
                elif self.offer(f, candidate, archive):
                    st.accepted += 1
                else:
                    st.dominated += 1

            iteration += 1
            if record_every and iteration % record_every == 0:
                history.append((iteration, n_eval, len(archive)))
            self.on_iteration(iteration, n_eval, archive)

        return Result(archive, n_eval, iteration, time.time() - t0, stopped_by, stats, history)

    def _start(self, init, rng, max_eval=None):
        """Evaluate the initial solutions, seed the archive and check the move tags."""
        init = init(rng) if callable(init) else init
        if isinstance(init, (list, tuple)):
            if not init:
                solutions = []
            elif all(np.isscalar(item) for item in init):
                solutions = [init]  # a single vector solution such as [1.0, 2.0]
            else:
                solutions = list(init)
        elif hasattr(init, "__iter__") and not isinstance(init, (str, bytes, np.ndarray, dict)):
            solutions = list(init)
        else:
            solutions = [init]
        if not solutions:
            raise ValueError("init produced no solutions")

        archive, n_eval = None, 0
        for solution in solutions:
            if max_eval is not None and n_eval >= max_eval:
                break
            f, g = self.evaluate(solution)
            n_eval += 1
            if archive is None:
                archive = Archive(f.shape[0], eps_rel=self.eps_rel)
                self._check_pools(archive.n_obj)
            if self._is_feasible(g):
                self.offer(f, solution, archive)
        if archive is None or len(archive) == 0:
            raise RuntimeError(
                "no initial solution is feasible; nothing to search from "
                f"(evaluated {n_eval} initial solution(s))")
        return archive, n_eval
