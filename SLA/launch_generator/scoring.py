"""Small proper-score primitives for archived launch forecasts."""

from __future__ import annotations

from collections.abc import Iterable


def brier_score(probability: float, outcome: bool) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    return (probability - float(outcome)) ** 2


def interval_coverage(intervals: Iterable[tuple[float, float]],
                      observations: Iterable[float]) -> float:
    pairs = list(zip(intervals, observations, strict=True))
    if not pairs:
        raise ValueError("at least one interval and observation are required")
    covered = sum(lo <= value <= hi for (lo, hi), value in pairs)
    return covered / len(pairs)
