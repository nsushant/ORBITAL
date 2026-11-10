"""Engine, budget, statistics and extension-point tests."""

import logging

import numpy as np
import pytest

from mdls import MDLS, move


def two_objectives(x, inst):
    return [x[0], x[1]], None


@move(improves=0)
def down_0(x, inst, rng):
    return [x[0] - 1.0, x[1]]


@move(improves=1)
def down_1(x, inst, rng):
    return [x[0], x[1] - 1.0]


def make_search(**kwargs):
    return MDLS(two_objectives, moves=[down_0(), down_1()], instance=None, **kwargs)


def test_evaluation_accounting():
    search = make_search()
    n_init = 3
    init = [[10.0, 10.0], [10.0, 9.0], [9.0, 10.0]]
    result = search.run(init, max_iter=300, seed=0)

    stats = list(result.stats.values())
    spent = sum(s.calls - s.no_move for s in stats) + n_init
    assert result.n_eval == spent
    assert result.n_iter == 300
    assert result.stopped_by == "max_iter"


def test_one_move_per_objective_per_iteration():
    search = make_search()
    result = search.run([[10.0, 10.0]], max_iter=25, seed=0)
    assert sum(s.calls for s in result.stats.values()) == len(search.pools) * result.n_iter


def test_pools_and_stats_cover_every_objective():
    search = make_search()
    result = search.run([[10.0, 10.0]], max_iter=5, seed=0)
    assert sorted(search.pools) == [0, 1]
    assert all(s.objective in search.pools for s in result.stats.values())
    assert set(result.stats) == {"down_0", "down_1"}


def test_max_eval_is_not_exceeded():
    search = make_search()
    result = search.run([[10.0, 10.0]], max_eval=50, seed=0)
    assert result.n_eval <= 50
    assert result.stopped_by == "max_eval"


def test_max_eval_bounds_initial_solutions():
    search = make_search()
    init = [[float(10 + i), float(10 + i)] for i in range(10)]
    result = search.run(init, max_eval=3, seed=0)
    assert result.n_eval == 3
    assert result.n_iter == 0
    assert result.stopped_by == "max_eval"


def test_same_seed_gives_the_same_front():
    search = make_search()
    first = search.run([[10.0, 10.0]], max_eval=200, seed=7)
    second = search.run([[10.0, 10.0]], max_eval=200, seed=7)
    assert np.array_equal(first.objectives, second.objectives)


def test_moves_receive_a_copy_not_the_archived_solution():
    @move(improves=0)
    def vandal(x, inst, rng):
        x[:] = 999.0
        return None

    @move(improves=1)
    def idle(x, inst, rng):
        return None

    search = MDLS(two_objectives, moves=[vandal(), idle()], instance=None)
    result = search.run([np.array([10.0, 10.0])], max_iter=3, seed=0)
    assert np.array_equal(result.solutions[0], [10.0, 10.0])


def test_dominated_candidates_are_counted():
    def pick_middle(archive, rng):
        return next(i for i, f in enumerate(archive.objectives)
                    if np.allclose(f, [5.0, 5.0]))

    @move(improves=0)
    def bad_trade(x, inst, rng):
        return [x[0] - 3.0, x[1] + 6.0]

    @move(improves=1)
    def noop(x, inst, rng):
        return None

    search = MDLS(two_objectives, moves=[bad_trade(), noop()], instance=None,
                  selection=pick_middle)
    init = [[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]]
    result = search.run(init, max_iter=2, seed=0)
    assert result.stats["bad_trade"].dominated == 2
    assert result.stats["bad_trade"].accepted == 0


def test_infeasible_candidates_are_counted():
    def evaluate(x, inst):
        return [x[0]], [x[0] - 5.0]

    @move(improves=0)
    def too_far(x, inst, rng):
        return [x[0] + 10.0]

    search = MDLS(evaluate, moves=[too_far()], instance=None)
    result = search.run([[0.0]], max_iter=2, seed=0)
    assert result.stats["too_far"].infeasible == 2
    assert result.n_eval == 3


def test_copy_solution_hook_is_used():
    copied = []

    def copy_solution(x):
        copied.append(True)
        return x.copy()

    search = make_search(copy_solution=copy_solution)
    search.run([[10.0, 10.0]], max_iter=2, seed=0)
    assert copied


def test_default_copy_falls_back_to_deepcopy():
    class Solution:
        def __init__(self, value):
            self.value = value

    def evaluate(x, inst):
        return [x.value], None

    @move(improves=0)
    def improve(x, inst, rng):
        return Solution(x.value - 1.0)

    search = MDLS(evaluate, moves=[improve()], instance=None)
    result = search.run([Solution(10.0)], max_iter=2, seed=0)
    assert result.objectives[0, 0] == 8.0


def test_scalar_objective_with_vector_constraint_is_a_clear_error():
    def evaluate(x, inst):
        return x[0], [0.0]

    search = MDLS(evaluate, moves=[down_0()], instance=None)
    with pytest.raises(ValueError, match="wrap f"):
        search.run([[10.0, 10.0]], max_iter=1)


def test_scalar_pair_is_read_as_two_objectives():
    def evaluate(x, inst):
        return x[0], x[1]

    @move(improves=0)
    def improve_0(x, inst, rng):
        return [x[0] - 1.0, x[1]]

    @move(improves=1)
    def improve_1(x, inst, rng):
        return [x[0], x[1] - 1.0]

    search = MDLS(evaluate, moves=[improve_0(), improve_1()], instance=None)
    result = search.run([[10.0, 10.0]], max_iter=1, seed=0)
    assert result.objectives.shape[1] == 2


@pytest.mark.parametrize("init", [
    [10.0, 10.0],
    [[10.0, 10.0]],
    iter([[10.0, 10.0]]),
    lambda rng: [[10.0, 10.0]],
    lambda rng: [10.0, 10.0],
])
def test_init_accepts_single_solution_lists_iterables_and_callables(init):
    result = make_search().run(init, max_iter=1, seed=0)
    assert len(result.archive) >= 1


def test_empty_init_is_refused():
    with pytest.raises(ValueError, match="no solutions"):
        make_search().run([], max_iter=1)


def test_no_initial_feasible_solution_is_refused():
    def evaluate(x, inst):
        return [x[0]], [1.0]

    @move(improves=0)
    def improve(x, inst, rng):
        return [x[0] - 1.0]

    search = MDLS(evaluate, moves=[improve()], instance=None)
    with pytest.raises(RuntimeError, match="feasible"):
        search.run([[10.0]], max_iter=1)


def test_at_least_one_budget_is_required():
    with pytest.raises(ValueError, match="at least one"):
        make_search().run([[10.0, 10.0]])


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0},
    {"max_eval": 0},
    {"time_limit": 0},
    {"time_limit": -1.0},
    {"time_limit": "soon"},
    {"time_limit": True},
    {"record_every": -1},
])
def test_invalid_budgets_are_refused(kwargs):
    with pytest.raises(ValueError):
        make_search().run([[10.0, 10.0]], **kwargs)


def test_constructor_validation():
    with pytest.raises(TypeError):
        MDLS(None, moves=[down_0()])
    with pytest.raises(TypeError):
        MDLS(two_objectives, moves=[down_0()], is_feasible=1)
    with pytest.raises(TypeError):
        MDLS(two_objectives, moves=[down_0()], selection=1)
    with pytest.raises(TypeError):
        MDLS(two_objectives, moves=[down_0()], copy_solution=1)
    with pytest.raises(ValueError):
        MDLS(two_objectives, moves=[down_0()], eps_rel=-1.0)
    with pytest.raises(ValueError):
        MDLS(two_objectives, moves=[down_0()], verbose=-1)
    with pytest.raises(ValueError):
        MDLS(two_objectives, moves=[])


def test_verbose_uses_logging(caplog):
    search = make_search(verbose=1)
    with caplog.at_level(logging.INFO, logger="mdls"):
        search.run([[10.0, 10.0]], max_iter=2, seed=0)
    assert "iter" in caplog.text
    assert "archive" in caplog.text


def test_summary_mentions_dominated_column():
    result = make_search().run([[10.0, 10.0]], max_iter=2, seed=0)
    text = result.summary()
    assert "dom" in text
    assert "down_0" in text
    assert "down_1" in text
