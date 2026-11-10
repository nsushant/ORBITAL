"""MDLS: multi-directional local search with user-supplied moves.

The public interface is small:

    from mdls import MDLS, move

    @move(improves=0, attempts=4)
    def my_move(x, instance, rng): ...

    search = MDLS(evaluate, moves=[my_move()], instance=instance)
    result = search.run(init, max_eval=10_000)
"""

from .core import (
    EPS_REL,
    Archive,
    MDLS,
    MoveStats,
    Result,
    all_feasible,
    dominates,
    move,
)

__version__ = "0.1.2"

__all__ = [
    "EPS_REL",
    "Archive",
    "MDLS",
    "MoveStats",
    "Result",
    "all_feasible",
    "dominates",
    "move",
    "__version__",
]
