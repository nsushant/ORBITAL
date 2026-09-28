"""Validate OOS SLA DSL problem bundles against the v0.1 JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def semantic_errors(problem: dict) -> list[str]:
    errors: list[str] = []
    sla = problem["sla"]
    state = problem["provider_state"]
    launch = problem["launch_access"]
    policy = problem["decision_policy"]

    if sla["service_window"]["earliest_s"] > sla["service_window"]["latest_s"]:
        errors.append("service_window.earliest_s must not exceed service_window.latest_s")
    if state["observation_time_s"] != 0:
        errors.append("provider_state.observation_time_s must be zero in v0.1")
    if launch["information_cutoff_s"] != 0:
        errors.append("launch_access.information_cutoff_s must be zero in v0.1")
    if launch["horizon_end_s"] < sla["service_window"]["latest_s"]:
        errors.append("launch-access horizon must cover the SLA completion deadline")
    if sla["reliability"]["scenario_model"] != launch["model_id"]:
        errors.append("SLA scenario_model must reference launch_access.model_id")

    expected_plugin = {
        "commodity_delivery": "commodity-delivery-v1",
        "repair": "repair-v1",
        "deorbit": "deorbit-v1",
    }[sla["service"]["type"]]
    if problem["models"]["service"] != expected_plugin:
        errors.append(f"service type requires models.service={expected_plugin!r}")

    decision_names = set(policy["decisions"])
    for rule in policy["freeze_rules"]:
        if rule["decision"] not in decision_names:
            errors.append(f"freeze rule references undefined decision {rule['decision']!r}")

    required = set(sla["service"].get("required_capabilities", []))
    available = {cap for servicer in state["servicers"] for cap in servicer["capabilities"]}
    missing = required - available
    if missing:
        errors.append(f"no servicer provides required capabilities: {sorted(missing)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="+", type=Path)
    parser.add_argument("--schema", type=Path, default=Path(__file__).parent / "schema" / "sla-dsl-v0.1.schema.json")
    args = parser.parse_args()

    try:
        import jsonschema
    except ImportError as exc:
        raise SystemExit("Install the 'jsonschema' package to validate the DSL") from exc

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    failed = False
    for path in args.problems:
        problem = json.loads(path.read_text(encoding="utf-8"))
        errors = [e.message for e in validator.iter_errors(problem)] + semantic_errors(problem)
        if errors:
            failed = True
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {path}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
