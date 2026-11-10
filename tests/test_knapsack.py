"""Integration tests against the knapsack example."""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "knapsack.py"

spec = importlib.util.spec_from_file_location("knapsack_example", EXAMPLE)
knapsack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knapsack)


def small_config(**overrides):
    cfg = json.loads((ROOT / "examples" / "params.json").read_text())
    cfg.update(n_items=30, n_random_starts=2, max_eval=400, seeds=[0],
               record_every=20, verbose=0)
    cfg.update(overrides)
    return cfg


def test_example_search_respects_budget_and_feasibility():
    cfg = small_config()
    inst, search, init = knapsack.build(cfg)
    result = search.run(init, max_eval=cfg["max_eval"], seed=0)

    assert result.n_eval <= cfg["max_eval"]
    assert result.stopped_by == "max_eval"
    assert len(result.archive) > 0
    assert all(inst["w"] @ x <= inst["cap"] for x in result.solutions)
    assert set(result.stats) == {"drop_add_0", "drop_add_1", "swap"}
    assert "dom" in result.summary()


def test_example_is_reproducible():
    cfg = small_config()
    _, search, init = knapsack.build(cfg)
    first = search.run(init, max_eval=300, seed=3)
    second = search.run(init, max_eval=300, seed=3)
    assert np.array_equal(first.objectives, second.objectives)


def test_example_can_disable_the_swap_move():
    cfg = small_config(use_swap=False)
    _, search, _ = knapsack.build(cfg)
    labels = {label for entries in search.pools.values() for label, _ in entries}
    assert labels == {"drop_add_0", "drop_add_1"}


def test_cmd_run_writes_the_front(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = small_config(front_csv=str(tmp_path / "front.csv"))
    knapsack.cmd_run(cfg)

    written = tmp_path / "front.csv"
    assert written.exists()
    lines = written.read_text().splitlines()
    assert lines[0] == "profit_0,profit_1"
    assert len(lines) > 1
    assert "knee:" in capsys.readouterr().out
