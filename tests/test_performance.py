"""Performance guards for the dominance index.

Deselected by default; run with `python -m pytest -m slow`.
"""

import time

import numpy as np
import pytest

import mdls.core as core
from mdls import Archive
from mdls._dominance import _Forest

pytestmark = pytest.mark.slow


def _front(n, m, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random((n, m))
    for j in range(m):
        u[:, j] = np.sort(u[:, j]) if j % 2 == 0 else np.sort(u[:, j])[::-1]
    return u


def _tradeoffs(rows, count=100, seed=5):
    """Improve one objective and worsen another: accepted, no victims."""
    rng = np.random.default_rng(seed)
    n, m = rows.shape
    span = (rows.max(0) - rows.min(0)).max()
    out = []
    for _ in range(count):
        point = rows[int(rng.integers(n))].copy()
        a, b = rng.choice(m, 2, replace=False)
        point[a] -= 0.02 * span
        point[b] += 0.02 * span
        out.append(point)
    return out


def test_twenty_thousand_inserts(monkeypatch):
    """A growing m=6 front must stay well under a second (0.1.2 took ~10 s)."""
    m = 6
    rows = _front(20_000, m)
    monkeypatch.setattr(core, "INDEX_MIN", 0)
    archive = Archive(m, capacity=32)
    archive.add(rows[0], 0)  # warm up the Numba kernels outside the timer

    started = time.perf_counter()
    for i, row in enumerate(rows[1:], start=1):
        archive.add(row, i)
    elapsed = time.perf_counter() - started

    assert len(archive) == 20_000
    assert elapsed < 1.0, f"{elapsed:.3f} s"


def test_index_beats_scan_at_large_front(monkeypatch):
    """Indexed inserts are materially faster than the scan once the front is big."""
    m, n = 6, 32_768
    rows = _front(n, m)
    points = _tradeoffs(rows)

    def build(index_min):
        monkeypatch.setattr(core, "INDEX_MIN", index_min)
        archive = Archive(m, capacity=n + len(points) + 2)
        archive._f[:n] = rows
        archive._n = n
        archive.solutions = list(range(n))
        archive.add(points[0], -1)  # builds the index outside the timer
        return archive

    def timed(index_min):
        archive = build(index_min)
        started = time.perf_counter()
        for point in points[1:]:
            archive.add(point, -1)
        return (time.perf_counter() - started) / (len(points) - 1) * 1e6

    scan = timed(10 ** 9)
    indexed = timed(0)
    assert indexed < 0.6 * scan, f"indexed {indexed:.1f} us vs scan {scan:.1f} us"


def test_query_budget_at_one_hundred_thousand():
    """A dominance query at n=100k must stay in single-digit microseconds."""
    m, n = 6, 100_000
    rows = _front(n, m)
    forest = _Forest.build(rows)
    rng = np.random.default_rng(1)
    queries = [rows[i] * (1 + rng.normal(0, 0.01, m))
               for i in rng.choice(n, 1_000, replace=False)]
    forest.blocked(queries[0], 1e-9)  # compile

    started = time.perf_counter()
    for q in queries:
        forest.blocked(q, 1e-9)
    elapsed = (time.perf_counter() - started) / len(queries) * 1e6
    assert elapsed < 10.0, f"{elapsed:.2f} us/query"
