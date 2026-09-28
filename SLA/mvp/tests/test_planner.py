import json
from pathlib import Path

from SLA.mvp.planner import evaluate_portfolio


EXAMPLE = Path(__file__).parents[1] / "example_problem.json"


def problem():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_example_protects_existing_sla_and_constructs_manifest():
    result = evaluate_portfolio(problem())
    assert result["baseline"]["reliable"]
    assert result["recommendation"] == "ACCEPT_WITH_EXPANSION"
    selected = next(x for x in result["options"]
                    if x["package_id"] == result["selected_package_id"])
    assert set(selected["item_ids"]) == {"XE-LOT-A", "XE-LOT-B", "SERVICER-2"}
    assert sum(item["mass_kg"] for item in selected["manifest"]) == 380
    assert sum(item["mass_kg"] for item in selected["manifest"]) <= 400
    assert selected["reliability"]["COMMITTED-REFUEL-01"] == 1.0
    assert selected["reliability"]["CANDIDATE-REFUEL-02"] >= 0.85


def test_rejects_candidate_when_revenue_does_not_cover_plan():
    data = problem()
    data["candidate_slas"][0]["revenue"] = 1.0
    result = evaluate_portfolio(data)
    assert result["recommendation"] == "REJECT"


def test_invalid_scenario_probability_is_rejected():
    data = problem()
    data["scenarios"][0]["probability"] = 0.5
    try:
        evaluate_portfolio(data)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("invalid probabilities were accepted")
