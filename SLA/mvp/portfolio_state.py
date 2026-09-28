"""Persistent rolling-portfolio transitions for the local demonstrator."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


def load_portfolio(example: Path, state_file: Path) -> dict[str, Any]:
    source = state_file if state_file.exists() else example
    return json.loads(source.read_text(encoding="utf-8"))


def accept_contract(payload: Mapping[str, Any], state_file: Path) -> dict[str, Any]:
    problem = deepcopy(payload["problem"])
    result = payload["result"]
    if result.get("recommendation") not in {"ACCEPT", "ACCEPT_WITH_EXPANSION"}:
        raise ValueError("Only an accepted planner recommendation can be committed.")
    candidates = problem.get("candidate_slas", [])
    if len(candidates) != 1:
        raise ValueError("The MVP acceptance route requires exactly one candidate SLA.")
    selected_id = result.get("selected_package_id")
    selected = next((x for x in result.get("options", []) if x.get("package_id") == selected_id), None)
    if selected is None:
        raise ValueError("The accepted recommendation has no selected planning option.")

    accepted = candidates[0]
    achieved = selected.get("reliability", {}).get(accepted["sla_id"])
    if achieved is None:
        raise ValueError("The selected option has no reliability result for the candidate SLA.")
    if accepted.get("required_reliability_source") == "model_output":
        accepted["required_reliability"] = float(achieved)
        accepted["required_reliability_source"] = "accepted_quote"
    accepted["acceptance"] = {
        "recommendation": result["recommendation"],
        "predicted_reliability": float(achieved),
        "package_id": selected_id,
    }
    problem.setdefault("accepted_slas", []).append(accepted)

    existing = {x["item_id"] for x in problem.get("committed_manifest", [])}
    committed = [dict(x) for x in selected.get("manifest", []) if x["item_id"] not in existing]
    problem.setdefault("committed_manifest", []).extend(committed)
    committed_ids = {x["item_id"] for x in problem["committed_manifest"]}
    problem["manifest_candidates"] = [
        x for x in problem.get("manifest_candidates", []) if x["item_id"] not in committed_ids
    ]

    template = deepcopy(accepted)
    template["sla_id"] = f"CANDIDATE-{len(problem['accepted_slas']) + 1:03d}"
    template["target_asset_id"] = "TO-BE-SPECIFIED"
    template.pop("target_asset_ids", None)
    template.pop("target_orbit_description", None)
    template.pop("acceptance", None)
    template["target_count"] = 1
    template["inventory"] = {key: 0 for key in template.get("inventory", {"xenon_kg": 0})}
    template["revenue"] = 0
    template["required_reliability"] = 0.0
    template["required_reliability_source"] = "model_output"
    problem["candidate_slas"] = [template]

    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(problem, indent=2), encoding="utf-8")
    temporary.replace(state_file)
    return {
        "status": "accepted",
        "accepted_sla_id": accepted["sla_id"],
        "committed_manifest_item_ids": [x["item_id"] for x in committed],
        "problem": problem,
    }
