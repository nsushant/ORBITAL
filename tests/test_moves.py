"""Move decorator, retry loop and tagging tests."""

import numpy as np
import pytest

from mdls import MDLS, move


def evaluate(x, inst):
    return [x[0]], None


def toy_run(mv, n_iter):
    return MDLS(evaluate, moves=[mv], instance=None).run([[10.0]], max_iter=n_iter, seed=0)


def test_attempts_defaults_to_one_and_is_recorded():
    @move(improves=0)
    def once(x, inst, rng):
        return None

    assert once.attempts == 1
    assert once().attempts == 1
    assert once.improves == 0


def test_one_attempt_per_call_when_attempts_is_one():
    attempts = []

    @move(improves=0, attempts=1)
    def single_try(x, inst, rng):
        attempts.append(1)
        return None

    toy_run(single_try(), 4)
    assert len(attempts) == 4


def test_retry_stops_at_first_improvement():
    attempts = []

    @move(improves=0, attempts=5)
    def immediate(x, inst, rng):
        attempts.append(1)
        return [x[0] - 1.0]

    result = toy_run(immediate(), 3)
    assert len(attempts) == 3
    assert result.objectives[0, 0] == 7.0


def test_retry_reaches_the_first_improving_attempt():
    attempts = []

    @move(improves=0, attempts=3)
    def every_third(x, inst, rng):
        attempts.append(1)
        return None if len(attempts) % 3 else [x[0] - 1.0]

    result = toy_run(every_third(), 2)
    assert len(attempts) == 6
    assert result.objectives[0, 0] == 8.0


def test_all_attempts_failing_is_one_no_move():
    attempts = []

    @move(improves=0, attempts=3)
    def never(x, inst, rng):
        attempts.append(1)
        return None

    result = toy_run(never(), 2)
    assert result.stats["never"].no_move == 2
    assert len(attempts) == 6


def test_non_descent_is_rejected_and_counted():
    @move(improves=0)
    def sideways(x, inst, rng):
        return [x[0]]

    result = toy_run(sideways(), 2)
    assert result.stats["sideways"].not_descent == 2
    assert len(result.archive) == 1


def test_internal_attempts_are_not_evaluations():
    attempts = []

    @move(improves=0, attempts=10)
    def failing(x, inst, rng):
        attempts.append(1)
        return None

    result = toy_run(failing(), 3)
    assert result.n_eval == 1
    assert len(attempts) == 30


@pytest.mark.parametrize("attempts", [0, -1, 1.5, True, "3"])
def test_attempts_must_be_a_positive_integer(attempts):
    with pytest.raises(ValueError):
        move(improves=0, attempts=attempts)


@pytest.mark.parametrize("improves", [-1, 1.5, True, "0"])
def test_improves_must_be_a_non_negative_integer(improves):
    with pytest.raises(ValueError):
        move(improves=improves)


def test_bare_factory_is_refused():
    @move(improves=0)
    def factory(x, inst, rng):
        return None

    with pytest.raises(ValueError, match="factory"):
        MDLS(evaluate, moves=[factory], instance=None)


def test_untagged_move_is_refused():
    with pytest.raises(ValueError, match="not tagged"):
        MDLS(evaluate, moves=[lambda x, inst, rng: x], instance=None)


def test_unknown_objective_is_refused():
    @move(improves=5)
    def ghost(x, inst, rng):
        return None

    search = MDLS(evaluate, moves=[ghost()], instance=None)
    with pytest.raises(ValueError, match="objective"):
        search.run([[10.0]], max_iter=1)


def test_missing_objective_is_refused():
    @move(improves=0)
    def only_zero(x, inst, rng):
        return None

    def two_objectives(x, inst):
        return [x[0], x[1]], None

    search = MDLS(two_objectives, moves=[only_zero()], instance=None)
    with pytest.raises(ValueError, match="no move tagged"):
        search.run([[10.0, 10.0]], max_iter=1)


def test_move_labels_are_function_names_and_duplicates_are_numbered():
    @move(improves=0)
    def same(x, inst, rng):
        return None

    search = MDLS(evaluate, moves=[same(), same()], instance=None)
    assert sorted(search.pools[0][i][0] for i in range(2)) == ["same", "same#2"]
