"""Archive and dominance tests."""

import numpy as np
import pytest

from mdls import Archive, dominates


def test_dominance_basics():
    assert dominates([1, 1], [2, 2])
    assert not dominates([1, 3], [3, 1])
    assert not dominates([1, 1], [1, 1])


def test_relative_tolerance_treats_last_bit_noise_as_equality():
    assert not dominates([1.0, 2.7e8], [1.0, 2.7e8 + 3e-8])


def test_archive_rejects_dominated_and_duplicate_points():
    archive = Archive(2)
    assert archive.add([5, 5], "a")
    assert not archive.add([6, 6], "b")
    assert not archive.add([5, 5], "c")
    assert len(archive) == 1
    assert archive.solutions == ["a"]


def test_archive_accepts_trade_off_and_evicts_dominated():
    archive = Archive(2)
    archive.add([5, 5], "a")
    assert archive.add([1, 9], "b")
    assert len(archive) == 2
    assert archive.add([1, 1], "c")
    assert archive.solutions == ["c"]
    assert np.array_equal(archive.objectives, [[1, 1]])


def test_archive_survives_churn_and_growth():
    rng = np.random.default_rng(0)
    archive = Archive(3, capacity=4)
    for i in range(500):
        archive.add(rng.random(3), i)

    f = archive.objectives
    assert len(archive.solutions) == len(archive)
    for i in range(len(archive)):
        for j in range(len(archive)):
            if i != j:
                assert not dominates(f[j], f[i])


def test_archive_knee_and_extreme():
    archive = Archive(2)
    for point in ([0.0, 10.0], [5.0, 5.0], [10.0, 0.0]):
        archive.add(point, point)
    assert archive.extreme(0) == 0
    assert archive.extreme(1) == 2
    assert archive.knee() == 1


def test_archive_validation():
    with pytest.raises(ValueError):
        Archive(0)
    with pytest.raises(ValueError):
        Archive(2, capacity=0)
    with pytest.raises(ValueError):
        Archive(2, eps_rel=-1.0)
    with pytest.raises(ValueError):
        Archive(2, eps_rel=float("nan"))

    archive = Archive(2)
    with pytest.raises(ValueError):
        archive.add([1.0], "wrong length")
    with pytest.raises(ValueError):
        archive.add([np.nan, 1.0], "non-finite")
