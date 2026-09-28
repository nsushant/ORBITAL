import json
from pathlib import Path

from SLA.mvp.planner import evaluate_portfolio


EXAMPLE = Path(__file__).parents[1] / "example_problem.json"


def test_product_result_contains_milp_and_mdls_evidence():
    problem = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    result = evaluate_portfolio(problem)
    assert result["milp"]["status"] == "optimal"
    assert sum(result["milp"]["package_binary_variables"].values()) == 1
    assert {item_id for item_id, value in result["milp"]["binary_variables"].items()
            if value == 1} == {"XE-LOT-A", "XE-LOT-B", "SERVICER-2"}
    selected = next(x for x in result["options"]
                    if x["package_id"] == result["selected_package_id"])
    assert all(outcome["mdls_evaluations"] > 0 for outcome in selected["scenario_outcomes"])
    assert any(outcome["routes"] for outcome in selected["scenario_outcomes"])
    assert all(outcome["route_model"] == "mdls-edelbaum-j2-v0.2"
               for outcome in selected["scenario_outcomes"])
