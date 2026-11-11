"""Dominance index tests: kernels, forest, and agreement with the linear scan."""

import numpy as np
import pytest

import mdls.core as core
from mdls import Archive, MDLS, move
from mdls._dominance import _Forest


def _no_worse(a, b, eps):
    """True when a is no worse than b in every objective, within eps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b + eps * np.maximum(np.abs(a), np.abs(b))))


def _front(n, m, seed=0, anti=True):
    """A non-dominated front (anti-correlated when `anti`), or uniform noise."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, m))
    if anti:
        for j in range(m):
            u[:, j] = np.sort(u[:, j]) if j % 2 == 0 else np.sort(u[:, j])[::-1]
    return u


def _brute_victims(rows, q, eps):
    return np.array([i for i, p in enumerate(rows)
                     if _no_worse(q, p, eps) and not _no_worse(p, q, eps)], dtype=np.int64)


def _brute_blocked(rows, q, eps):
    return any(_no_worse(p, q, eps) for p in rows)


def _queries(rows, count=16, seed=0):
    rng = np.random.default_rng(seed)
    n, m = rows.shape
    out = [rows[int(rng.integers(n))] * (1 + rng.normal(0, 0.01, m)) for _ in range(count)]
    out += [rng.random(m) for _ in range(count // 2)]
    return out


# --------------------------------------------------------------------------- kernels

@pytest.mark.parametrize("m", [1, 2, 3, 6, 8])
@pytest.mark.parametrize("eps", [0.0, 1e-9, 1e-6])
def test_forest_queries_match_brute_force(m, eps):
    for n in (1, 2, 15, 16, 17, 33, 64, 512):
        rows = _front(n, m, seed=n + m)
        forest = _Forest.build(rows)
        for q in _queries(rows, seed=n):
            assert forest.blocked(q, eps) == _brute_blocked(rows, q, eps)
            assert np.array_equal(np.sort(forest.victims(q, eps)), _brute_victims(rows, q, eps))


@pytest.mark.parametrize("m", [2, 6])
def test_incremental_forest_matches_rebuild(m):
    rows = _front(600, m, seed=3)
    forest = _Forest(m)
    for i, row in enumerate(rows):
        forest.add(row, i)
    rebuilt = _Forest.build(rows)
    for q in _queries(rows, count=40, seed=9):
        assert forest.blocked(q, 1e-9) == rebuilt.blocked(q, 1e-9)
        assert np.array_equal(np.sort(forest.victims(q, 1e-9)),
                              np.sort(rebuilt.victims(q, 1e-9)))


def test_forest_owns_its_rows():
    """A caller may reuse its buffer after building; the index must not alias it."""
    rows = _front(64, 3)
    forest = _Forest.build(rows)
    queries = rows.copy()
    before = [(forest.blocked(q, 1e-9), tuple(np.sort(forest.victims(q, 1e-9))))
              for q in queries]
    rows[:] = 0.0
    after = [(forest.blocked(q, 1e-9), tuple(np.sort(forest.victims(q, 1e-9))))
             for q in queries]
    assert before == after


# --------------------------------------------------------------------------- archive

@pytest.mark.parametrize("m", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("eps", [0.0, 1e-9, 1e-6])
@pytest.mark.parametrize("anti", [True, False])
def test_index_matches_scan_after_every_insert(m, eps, anti, monkeypatch):
    points = _front(300, m, seed=m + int(anti), anti=anti)
    scan, indexed = Archive(m, eps_rel=eps), Archive(m, eps_rel=eps)
    for i, point in enumerate(points):
        monkeypatch.setattr(core, "INDEX_MIN", 10 ** 9)
        scan_accepted = scan.add(point, i)
        monkeypatch.setattr(core, "INDEX_MIN", 0)
        indexed_accepted = indexed.add(point, i)
        assert scan_accepted == indexed_accepted
        assert np.array_equal(scan.objectives, indexed.objectives)
        assert scan.solutions == indexed.solutions


def test_threshold_crossing_with_eviction(monkeypatch):
    """The index is built from live rows, then compaction rewrites those rows."""
    m = 4
    points = _front(200, m, seed=11)
    wipe = np.full(m, -1.0)

    def run(index_min):
        monkeypatch.setattr(core, "INDEX_MIN", index_min)
        archive = Archive(m)
        for i, point in enumerate(points):
            archive.add(point, i)
            if i == 120:
                archive.add(wipe, "wipe")
        for i, point in enumerate(_front(120, m, seed=12) * 0.5):
            archive.add(point, ("post", i))
        return archive

    scan = run(10 ** 9)
    indexed = run(64)
    assert np.array_equal(scan.objectives, indexed.objectives)
    assert scan.solutions == indexed.solutions


def test_eviction_heavy_falls_back_to_scan(monkeypatch):
    """Frequent evictions disable the index and results stay identical."""
    sequence = []
    for t in range(60):
        sequence.append(([float(t), -float(t)], f"p{t}"))          # appended
        sequence.append(([float(t), -float(t) - 0.5], f"q{t}"))    # evicts p{t}

    def run(index_min):
        monkeypatch.setattr(core, "INDEX_MIN", index_min)
        archive = Archive(2)
        for point, label in sequence:
            archive.add(point, label)
        return archive

    scan = run(10 ** 9)
    indexed = run(0)
    assert np.array_equal(scan.objectives, indexed.objectives)
    assert scan.solutions == indexed.solutions
    assert indexed._index_enabled is False
    assert indexed._rebuilds > 8


def test_objectives_stays_a_view(monkeypatch):
    monkeypatch.setattr(core, "INDEX_MIN", 0)
    archive = Archive(2)
    for i in range(40):
        archive.add([float(i), -float(i)], i)
    assert np.shares_memory(archive.objectives, archive._f)


# --------------------------------------------------------------------------- engine

def _evaluate(x, inst):
    return [x[0], x[1]], None


@move(improves=0)
def _down_0(x, inst, rng):
    return [x[0] - 0.5, x[1] + rng.random()]


@move(improves=1)
def _down_1(x, inst, rng):
    return [x[0] + rng.random(), x[1] - 0.5]


@pytest.mark.parametrize("index_min", [0, 10 ** 9])
def test_engine_is_deterministic_and_agrees(index_min, monkeypatch):
    def run(threshold):
        monkeypatch.setattr(core, "INDEX_MIN", threshold)
        search = MDLS(_evaluate, moves=[_down_0(), _down_1()], instance=None)
        return search.run([5.0, 5.0], max_iter=400, seed=7)

    first = run(index_min)
    second = run(index_min)
    assert np.array_equal(first.objectives, second.objectives)
    assert first.solutions == second.solutions

    reference = run(10 ** 9)
    assert np.array_equal(first.objectives, reference.objectives)
    assert first.solutions == reference.solutions
