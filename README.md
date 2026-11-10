# mdls

`mdls` is a small, decorator-based implementation of **multi-directional local
search** (MDLS) for multi-objective optimisation. You supply an `evaluate`
function and one move per objective; the library owns the non-dominated
archive, selection, budgets and statistics.

The method follows Tricoire, Graf and Hartl (2012). This package is a
generalised, reusable version of the MDLS engine used in
[ORBITAL](https://github.com/nsushant/ORBITAL/tree/MDLS).

## Install

```bash
pip install mdls
```

Python 3.10 or newer is required. NumPy and Numba are installed automatically.

For the example's plotting command, install the extra:

```bash
pip install "mdls[examples]"
```

## Quick start

```python
import numpy as np

from mdls import MDLS, move


def evaluate(x, inst):
    """MDLS minimises; g <= 0 means feasible."""
    return -(inst["obj"] @ x), [inst["weight"] @ x - inst["cap"]]


def _toggle(x, inst, rng, k, n_toggle):
    """One attempt: flip items, keep the copy if objective k improves."""
    y = x.copy()
    flip = rng.choice(x.size, size=n_toggle, replace=False)
    y[flip] = ~y[flip]
    if inst["weight"] @ y > inst["cap"]:
        return None
    return y if inst["obj"][k] @ y > inst["obj"][k] @ x else None


@move(improves=0, attempts=8)
def toggle_0(x, inst, rng, n_toggle=2):
    return _toggle(x, inst, rng, 0, n_toggle)


@move(improves=1, attempts=8)
def toggle_1(x, inst, rng, n_toggle=2):
    return _toggle(x, inst, rng, 1, n_toggle)


rng = np.random.default_rng(0)
n = 20
inst = {"obj": rng.random((2, n)), "weight": rng.random(n), "cap": 0.3 * n}
moves = [toggle_0(n_toggle=2), toggle_1(n_toggle=2)]
search = MDLS(evaluate, moves=moves, instance=inst)

res = search.run([np.zeros(n, dtype=bool)], max_eval=500, seed=0)
print(res.summary())
print(res.objectives)
```

## What the library provides

- **Archive** of feasible, non-dominated solutions with a relative tolerance,
  duplicate rejection and dominated-point eviction.
- **Move contract**: a move is one improving attempt, optionally retried up to
  `attempts` times by the `@move` decorator.
- **Balanced search**: every iteration applies one sampled move per objective
  from the same archived base solution.
- **Budgets**: stop on iterations (`max_iter`), evaluations (`max_eval`,
  including initial solutions) or wall-clock time (`time_limit`).
- **Statistics**: per-move counts of calls, no-moves, non-descents, infeasible
  results, dominated results, accepted results and time.
- **Extension points**: `is_feasible`, `selection`, `copy_solution`, and the
  `select`, `offer`, `copy` and `on_iteration` hooks.

Moves are ordinary functions and stay readable; Numba is optional but is used
inside the library's dominance scan and can be used inside your own moves.

## Examples

The repository includes a bi-objective 0/1 knapsack:

```bash
python examples/knapsack.py run      # one search, writes front.csv
python examples/knapsack.py plot     # fronts for up to four seeds, writes front.png
```

The full tutorial, including the move contract and every option, is in
[`user_guide.md`](https://github.com/nsushant/ORBITAL/blob/MDLS/user_guide.md).

## Testing

```bash
pip install -e ".[dev,examples]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this package in academic work, cite the MDLS paper and optionally
the software itself (see `CITATION.cff`):

> Tricoire, F., Graf, A., & Hartl, R. F. (2012). The bi-objective stochastic
> covering tour problem. *Computers & Operations Research*, 39(7), 1582–1592.
