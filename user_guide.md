# MDLS user guide

## Install

```bash
pip install pymdls
```

The distribution is named `pymdls`; the import package is `mdls`.

Python 3.10 or newer is required. To work from a checkout instead, run
`pip install -e .` from the repository root (the `MDLS` branch).

## Motivation

Multi-objective search usually needs weights to collapse the objectives into one, or a population method such as NSGA-II. Multi-Directional Local Search (Tricoire 2012) avoids both. Any point on the Pareto front has to beat every other solution on at least one objective, so making one objective better is always a way of reaching the front. MDLS keeps an archive of feasible non-dominated solutions. Each iteration it picks one solution from the archive and runs a local search from it in each objective's direction. Every result goes back into the archive, and dominated points are removed.

In the ORBITAL code (`oos/mdls.py`), the archive, the loop and the three servicing operators are wired together in a single function. Adding or swapping an operator meant editing the loop. This `mdls` package separates the two parts. The `MDLS` class contains everything that is the same for any problem:

- **Archive:** it holds only feasible solutions, and it compares objectives with a relative tolerance of 1e-9, so float noise never shows up as a trade-off. Duplicates are refused, and dominated points are evicted.
- **Selection:** uniform by default.
- **Loop and budgets:** each iteration starts from one archived solution and applies one sampled move per objective to a fresh copy of it, so all directions are explored from the same base. The run stops on iterations, evaluations or wall clock.
- **Coverage:** every objective must have at least one tagged move; a missing or out-of-range tag is an error at the first evaluation.
- **Statistics:** for each move it counts calls, no-moves, non-descents, infeasible results, dominated results and accepted results.

You only write the problem-specific part: an `evaluate` function and the moves. Pass them in with the instance, and the class runs MDLS with them.

## Quick start: a complete problem

Here is a complete, runnable file. It selects a subset of items to maximise two profits while staying under a weight capacity. The moves flip a few random items and keep the result only when the profit they are responsible for goes up.

```python
"""Bi-objective subset selection solved with mdls.MDLS."""

import numpy as np

from mdls import MDLS, move


def evaluate(x, inst):
    """MDLS minimises: negate both profits. Feasible when g <= 0."""
    return -(inst["obj"] @ x), [inst["weight"] @ x - inst["cap"]]


def _toggle(x, inst, rng, k, n_toggle):
    """One attempt: flip n_toggle random items; keep the copy if profit k rises.

    x is left untouched, so the decorator can retry from the same base.
    """
    obj, weight = inst["obj"][k], inst["weight"]
    y = x.copy()
    flip = rng.choice(x.size, size=n_toggle, replace=False)
    y[flip] = ~y[flip]
    if weight @ y > inst["cap"]:
        return None
    return y if obj @ y > obj @ x else None


@move(improves=0, attempts=8)
def toggle_0(x, inst, rng, n_toggle=2):
    """Move for objective 0: flip items, retry up to 8 times."""
    return _toggle(x, inst, rng, 0, n_toggle)


@move(improves=1, attempts=8)
def toggle_1(x, inst, rng, n_toggle=2):
    """Move for objective 1: flip items, retry up to 8 times."""
    return _toggle(x, inst, rng, 1, n_toggle)


def build(cfg):
    """Return (instance, search, init) for the driver."""
    rng = np.random.default_rng(cfg["instance_seed"])
    n = cfg["n_items"]
    inst = {
        "obj": rng.random((2, n)),
        "weight": rng.random(n),
        "cap": cfg["capacity_fraction"] * n,
    }
    moves = [toggle_0(n_toggle=cfg["n_toggle"]),
             toggle_1(n_toggle=cfg["n_toggle"])]

    def init(rng):
        return [np.zeros(n, dtype=bool)]

    search = MDLS(evaluate, moves=moves, instance=inst, verbose=cfg["verbose"])
    return inst, search, init


if __name__ == "__main__":
    cfg = {"n_items": 20, "capacity_fraction": 0.3, "n_toggle": 2,
           "instance_seed": 0, "verbose": 0}
    inst, search, init = build(cfg)
    res = search.run(init, max_eval=500, seed=0)
    print(res.summary())
    for f, x in zip(res.objectives, res.solutions):
        print(f"profit=({-f[0]:.3f}, {-f[1]:.3f})  items={x.sum()}")
```

Save it as `subset.py` and run `python subset.py`. The output is the run report followed by the front, one line per archived solution. From there, the rest of this guide explains each piece and the choices you can make.

## The `evaluate` function

`evaluate` is passed directly to `MDLS`; it has no decorator. It takes a solution and the instance and returns either the objective vector or a pair `(f, g)`:

```python
def evaluate(x, instance):
    f = ...          # objectives, always minimised
    g = ...          # constraint values, feasible when every g <= 0
    return f, g
```

- Every objective is **minimised**. For a maximisation problem, negate the value, as the quick start does.
- Return just `f` for an unconstrained problem. To return constraints, make `f` a vector (list or array) so the pair is unambiguous. A scalar `f` with a vector `g` raises an error rather than guessing; write `return [f], g`.
- `f` must be finite. Reject a bad solution through `g`; a non-finite objective raises an error.
- `evaluate` is called once for each initial solution (up to `max_eval`) and once for each candidate a move returns. It is **not** called by the retry loop inside a move.

## Moves

A move is an ordinary function with the signature `(x, instance, rng, **params)`. It receives a solution it owns, changes it, and returns either an improving candidate or `None`.

### The move contract

A move makes **one improving attempt** for the objective named in its decorator:

```python
@move(improves=0)
def my_move(x, instance, rng, step=0.5):
    y = x.copy()            # work on a copy
    ...                     # try one change to y
    return y                # only if objective 0 strictly improved
```

The rules are:

1. `x` is a private copy of the archived solution, so mutating it cannot corrupt the archive.
2. If the attempt fails, return `None` and leave `x` unchanged. The decorator reuses the same base for every retry.
3. If the attempt succeeds, return a solution whose tagged objective is strictly better than the base's. The engine checks this centrally and rejects a candidate that is not a descent.
4. Use `rng` for randomness; do not call `np.random` directly, or a seeded run will not be reproducible.
5. Keep `evaluate` out of the move. Internal checks use the instance arithmetic; only the returned candidate is evaluated and counted.

Returning `None` costs no evaluation, which is why a cheap "nothing to do here" is preferable to returning an unchanged solution.

### Parameters and the factory

`@move(improves=k)` tags the function and turns it into a **factory**. Calling the factory binds parameters and returns the move that the engine runs:

```python
@move(improves=0, attempts=4)
def drop_add(x, instance, rng, n_drop=3, noise=0.2):
    ...

moves = [drop_add(n_drop=5, noise=0.3)]
```

The decorated function is never called by the engine directly. The factory exists so the same move can be configured differently in one search, and so the tag and retry bound travel with the move object. Pass the **called** result to `MDLS`:

```python
MDLS(evaluate, moves=[drop_add(n_drop=5)], instance=inst)   # correct
MDLS(evaluate, moves=[drop_add], instance=inst)             # error: bare factory
```

`improves` is a zero-based integer index into the objective vector. `attempts` is a positive integer and defaults to `1`.

### One move per objective

Each iteration samples exactly one move from each objective's pool. The pools are built from the tags, so every objective needs at least one move:

```python
moves = [toggle_0(), toggle_1()]        # objective 0 and objective 1 covered
```

An objective with no move, an out-of-range tag, or a non-integer tag is an error at the first evaluation. An untagged callable or a bare factory is rejected when `MDLS` is constructed. A single move may not serve two objectives; write one decorated function per objective around a shared helper.

### Retries with `attempts`

`attempts` is the number of times the decorator calls the attempt before giving up. It returns the first non-`None` result, so a stochastic move can sample a few neighbours per engine step:

```python
@move(improves=0, attempts=8)
def toggle_0(x, instance, rng, n_toggle=2):
    return _toggle(x, instance, rng, 0, n_toggle)
```

With `attempts=8`, the engine still spends at most one evaluation per objective per iteration: the retries happen inside the move, and only the candidate that is finally returned is evaluated. Use a larger `attempts` for cheap attempts that often fail, and keep `attempts=1` when a single attempt is already expensive or when the attempt itself scans a batch.

`attempts` must be at least 1. The value is fixed when the move is decorated; it is not a per-call override.

### A move is one attempt, not a local search

A move should not loop until it reaches a local optimum. MDLS compares every candidate with the base solution it came from, so a move that returns a distant point hides which step helped and makes the statistics hard to read. Keep the attempt small and let the engine's iterations do the descending:

```python
# good: one swap, one drop-and-refill, one toggle batch
@move(improves=0)
def swap(x, instance, rng, tries=50):
    ...
    return x if found else None
```

The bounded retry loop is not a local search: it only retries the *same* attempt until one succeeds. If a problem genuinely needs a valley-crossing sequence, make that sequence a single compound move and document it, but expect the archive to hold only the endpoint.

### A shared helper for several objectives

Most problems have the same operator for every objective, differing only in which profit it improves. Write the body once and decorate one wrapper per objective:

```python
def _drop_add(x, inst, rng, k, n_drop, noise):
    p, w = inst["p"][k], inst["w"]
    held = np.flatnonzero(x)
    if held.size == 0:
        return None
    y = x.copy()
    ratio = p[held] / w[held]
    noisy = ratio * rng.uniform(1 - noise, 1 + noise, held.size)
    y[held[np.argsort(noisy)[:n_drop]]] = False
    greedy_fill(y, w, inst["cap"], np.argsort(-p / w))   # numba kernel, see below
    return y if p @ y > p @ x else None


@move(improves=0, attempts=4)
def drop_add_0(x, inst, rng, n_drop=3, noise=0.2):
    return _drop_add(x, inst, rng, 0, n_drop, noise)


@move(improves=1, attempts=4)
def drop_add_1(x, inst, rng, n_drop=3, noise=0.2):
    return _drop_add(x, inst, rng, 1, n_drop, noise)
```

The helper returns an improving copy or `None` and never mutates `x`, so the decorator can retry from the same base. The two wrappers are separate moves with separate tags, statistics and retry bounds.

## The `build(cfg)` convention

A problem is a module-level `build(cfg)` that returns the pieces a driver needs:

```python
def build(cfg):
    inst = make_instance(cfg)
    moves = [drop_add_0(n_drop=cfg["n_drop"]),
             drop_add_1(n_drop=cfg["n_drop"]),
             swap(tries=cfg["swap_tries"])]
    search = MDLS(evaluate, moves=moves, instance=inst, verbose=cfg["verbose"])
    return inst, search, init       # init as in the quick start
```

There is no `@evaluate` or `@problem` decorator and no file or name discovery. `build` is ordinary Python: read the config, construct the instance, bind move parameters, construct `MDLS`, and return whatever the driver needs.

`init` tells `run` where to start. It can be a single solution, a list of solutions, any iterable of solutions, or a callable `init(rng) -> solutions`:

```python
res = search.run(init, max_eval=10_000, seed=0)   # callable
res = search.run([x0, x1], max_iter=500)          # list of solutions
res = search.run(x0, max_iter=500)                # one solution
```

A flat list or tuple of numbers is treated as one vector solution, so `[x0, x1]` means two solutions only when `x0` and `x1` are themselves arrays or lists. Every initial solution is evaluated, up to the `max_eval` budget. At least one must be feasible, or `run` raises an error.

## What the engine does

Each iteration:

1. Selects one solution from the archive.
2. Copies it once for each objective's pool and runs one sampled move per objective from that same base.
3. Evaluates each returned candidate, skipping the ones that return `None`.
4. Rejects infeasible candidates and candidates that do not strictly improve their tagged objective.
5. Offers the rest to the archive, which evicts whatever they dominate. A descent that the archive rejects because an existing point already dominates it (or equals it) is counted as `dom`.

The descent check is relative, `f[k] < base_f[k] - eps_rel * max(|f[k]|, |base_f[k]|)`, so float noise is not treated as an improvement. A rejected non-descent is counted as `no-desc`; that usually means the move's own improvement test disagrees with `evaluate`.

## Results and statistics

`run` returns a `Result`:

```python
res.objectives        # (front size, n_obj) objective vectors
res.solutions         # the solution behind each row
res.archive           # the Archive itself
res.n_eval, res.n_iter, res.elapsed, res.stopped_by
res.stats             # {move label: MoveStats}
res.history           # [(iteration, n_eval, front size), ...] with record_every
```

`res.archive.knee()` returns the index of the point nearest the ideal after normalisation, and `res.archive.extreme(k)` returns the index of the best point on objective `k`:

```python
k = res.archive.knee()
best_0 = res.archive.extreme(0)
print(res.objectives[k], res.solutions[k])
```

`res.summary()` prints one line per move:

```
MDLS: 3388 iterations, 6000 evaluations, 0.31 s, stopped by max_eval; front holds 85
  move                     obj   calls  no-move no-desc  infeas     dom accepted  acc %    sec
  drop_add_0                 0    1721       49       0       0    1612       60    3.5   0.03
  swap                       0    1667      678       0       0     946       43    2.6   0.03
  drop_add_1                 1    3387       55       0       0    3232      100    3.0   0.06
```

- `calls`: how often the move was sampled.
- `no-move`: the move returned `None` after all retries.
- `no-desc`: the candidate did not strictly improve its tagged objective.
- `infeas`: the candidate violated a constraint.
- `dom`: the candidate improved its objective but was dominated by (or equal to) a point already in the archive.
- `accepted`: the candidate entered the archive.
- `acc %`: accepted as a share of calls.

Budgets are set on `run`. At least one is required:

```python
search.run(init, max_iter=5_000)                  # iterations
search.run(init, max_eval=20_000)                 # evaluations, including initial ones
search.run(init, time_limit=60)                   # wall-clock seconds
search.run(init, max_eval=20_000, time_limit=60)  # stop at whichever comes first
```

`max_eval` counts every call to `evaluate`, including the initial solutions. Internal retries are not counted. `time_limit` includes any numba compilation on the first call.

## Options

```python
MDLS(evaluate, moves=moves, instance=inst,
     is_feasible=fn,          # fn(g) -> bool; default is every g <= 0
     selection=fn,            # fn(archive, rng) -> index; default is uniform
     eps_rel=1e-9,            # relative tolerance for dominance and descent
     verbose=0,               # log progress every n iterations
     copy_solution=fn)        # fn(x) -> copy; default x.copy() or deepcopy
```

- `is_feasible=fn(g)` replaces the feasibility rule. For ORBITAL, pass `oos.schedule.is_feasible`.
- `selection=fn(archive, rng)` replaces uniform selection. It must return a valid index into the archive.
- `copy_solution=fn(x)` controls how an archived solution is copied before it is handed to a move. The default copies NumPy arrays with `.copy()` and deep-copies every other object, so nested mutable structures cannot leak back into the archive. Override the `copy` method instead to change the rule for a whole subclass.
- `verbose=n` logs one progress line every `n` iterations through the `mdls` logger. Enable it with `logging.basicConfig(level=logging.INFO)`; the library never writes to stdout on its own.
- To change one step of the loop, subclass `MDLS` and override `select`, `offer`, `copy` or `on_iteration`:

```python
class MyMDLS(MDLS):
    def select(self, archive, rng):
        return archive.knee()          # always search from the knee

    def on_iteration(self, iteration, n_eval, archive):
        ...                            # custom logging or bookkeeping
```

- `run(..., record_every=n)` logs `(iteration, n_eval, front size)` to `res.history`, which is handy for plotting archive growth.

## Numba

The interface stays plain Python. Performance comes from compiling the loop-heavy kernels inside your moves with `@njit`, which leaves the class and the move signatures untouched:

```python
from numba import njit

@njit
def greedy_fill(x, w, cap, order):
    load = 0.0
    for i in range(x.size):
        if x[i]:
            load += w[i]
    for t in range(order.size):
        i = order[t]
        if not x[i] and load + w[i] <= cap:
            x[i] = True
            load += w[i]
    return x
```

`examples/knapsack.py` does this for its greedy refill and its swap search, and the library compiles the archive's dominance scan in `src/mdls/core.py`. Keep these rules in mind:

- `cache=True` avoids recompiling in later processes, but it is optional and can misbehave for modules loaded dynamically; the first call in a process still compiles.
- Do not use `fastmath=True` in the archive: it can break the 1e-9 tolerance and the finite-value check.
- Keep random draws in the Python move and pass arrays into the kernel; numba's support for `np.random.Generator` methods is incomplete.
- `attempts` is a Python-level retry loop. A move that scans a batch inside a numba kernel (like `swap` with `tries`) can keep `attempts=1`; the kernel's batch is its own guard.
- A wall-clock `time_limit` includes the first compile. Warm the kernels up before a timed comparison.

## Files and commands

**Files:** `src/mdls/core.py` (the class and decorator), `examples/knapsack.py` (a bi-objective knapsack and the command line), `examples/params.json` (the parameters), `tests/` (the pytest suite), `pyproject.toml` (packaging and dependencies).

**Commands:**

```
pip install -e ".[examples]"                   # from the repository root
pytest                                          # the test suite
python examples/knapsack.py run                 # one search, writes front.csv
python examples/knapsack.py plot                # fronts for up to 4 seeds, writes front.png
python examples/knapsack.py run --config my.json             # another parameter file
python examples/knapsack.py run --set max_eval=2000 seeds=[1,2]   # override single entries
```
