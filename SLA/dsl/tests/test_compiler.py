"""Compilation tests for the three SLA DSL v0.1 examples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from SLA.dsl.compiler import (
    CommodityDeliveryService,
    DSLCompileError,
    DeorbitService,
    RepairService,
    compile_document,
    load_problem,
)


EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    ("filename", "service_type"),
    [
        ("refuelling.json", CommodityDeliveryService),
        ("repair.json", RepairService),
        ("deorbit.json", DeorbitService),
    ],
)
def test_examples_compile(filename, service_type):
    problem = load_problem(EXAMPLES / filename)
    assert problem.dsl_version == "0.1"
    assert isinstance(problem.sla.service, service_type)
    assert problem.candidate_servicers()
    assert problem.sla.service_window.earliest_s <= problem.sla.service_window.latest_s
    assert problem.provider_state.observation_time_s == 0.0
    assert problem.launch_access.information_cutoff_s == 0.0


def test_compile_preserves_money_exactly():
    problem = load_problem(EXAMPLES / "refuelling.json")
    assert str(problem.sla.commercial_terms.revenue.value) == "15000000"
    assert problem.sla.commercial_terms.revenue.currency == "EUR"


def test_absolute_time_round_trip_uses_seconds():
    problem = load_problem(EXAMPLES / "refuelling.json")
    deadline = problem.at_seconds(problem.sla.service_window.latest_s)
    assert deadline == datetime(2030, 6, 30, tzinfo=timezone.utc)
    assert problem.seconds_at(deadline) == problem.sla.service_window.latest_s


def test_semantic_mismatch_is_rejected():
    data = json.loads((EXAMPLES / "repair.json").read_text(encoding="utf-8"))
    data["models"]["service"] = "deorbit-v1"
    with pytest.raises(DSLCompileError, match="repair-v1"):
        compile_document(data)


def test_unknown_fields_are_rejected():
    data = json.loads((EXAMPLES / "refuelling.json").read_text(encoding="utf-8"))
    data["sla"]["unexpected"] = True
    with pytest.raises(DSLCompileError, match="unexpected"):
        compile_document(data)
