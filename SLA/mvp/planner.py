"""Explainable first-pass SLA portfolio and architecture evaluator.

This is intentionally an aggregate planning approximation.  It screens
architecture packages across named scenarios using service-time and commodity
balances.  Detailed orbital routing remains the responsibility of MDLS; the
stable JSON boundary and result schema here are designed so that MDLS/MILP can
replace this screening model without changing the local web application.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
from typing import Any, Mapping

from ..optimization.milp import construct_manifest_packages, select_architecture
from .routing import build_edelbaum_cost_table, run_mdls_scenario


def _nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _validate(problem: Mapping[str, Any]) -> None:
    as_of = datetime.fromisoformat(str(problem["as_of"]).replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        raise ValueError("as_of must be an accurate timezone-aware timestamp")
    scenarios = problem.get("scenarios", [])
    if not scenarios:
        raise ValueError("at least one uncertainty scenario is required")
    probability = sum(float(s["probability"]) for s in scenarios)
    if abs(probability - 1.0) > 1e-9:
        raise ValueError(f"scenario probabilities must sum to one, got {probability}")
    for group in ("accepted_slas", "candidate_slas"):
        ids = set()
        for sla in problem.get(group, []):
            if sla["sla_id"] in ids:
                raise ValueError(f"duplicate SLA id {sla['sla_id']!r} in {group}")
            ids.add(sla["sla_id"])
            _nonnegative(sla["service_duration_s"], "service_duration_s")
            _nonnegative(sla["deadline_s"], "deadline_s")
            reliability = float(sla["required_reliability"])
            if not 0 <= reliability <= 1:
                raise ValueError("required_reliability must lie in [0, 1]")


def _resource_at_deadline(
    state: Mapping[str, Any], package: Mapping[str, Any], scenario: Mapping[str, Any],
    deadline_s: float,
) -> tuple[float, dict[str, float], bool]:
    base_servicers = _nonnegative(state.get("servicer_count", 0), "servicer_count")
    availability = float(scenario.get("servicer_availability", 1.0))
    base_capacity = base_servicers * deadline_s * availability
    base_inventory = {
        name: _nonnegative(quantity, f"inventory.{name}")
        * float(scenario.get("inventory_multiplier", 1.0))
        for name, quantity in state.get("inventory", {}).items()
    }

    dependency = package.get("launch_dependency")
    deployed = dependency is None or bool(scenario.get("launch_outcomes", {}).get(dependency, False))
    if deployed and float(package.get("available_from_s", 0.0)) <= deadline_s:
        effective_s = max(0.0, deadline_s - float(package.get("available_from_s", 0.0)))
        base_capacity += float(package.get("added_servicers", 0.0)) * effective_s * availability
        for commodity, quantity in package.get("added_inventory", {}).items():
            base_inventory[commodity] = base_inventory.get(commodity, 0.0) + float(quantity)
    return base_capacity, base_inventory, deployed


def _scenario_schedule(
    state: Mapping[str, Any],
    package: Mapping[str, Any],
    scenario: Mapping[str, Any],
    slas: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Earliest-deadline aggregate schedule with explicit failure reasons."""
    ordered = sorted(slas, key=lambda x: (float(x["deadline_s"]), x["sla_id"]))
    used_service_s = 0.0
    used_inventory: dict[str, float] = defaultdict(float)
    success: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}

    for sla in ordered:
        deadline_s = float(sla["deadline_s"])
        capacity_s, inventory, deployed = _resource_at_deadline(
            state, package, scenario, deadline_s)
        required_service_s = float(sla["service_duration_s"])
        required_inventory = sla.get("inventory", {})
        failures = []
        if package.get("launch_dependency") and not deployed:
            # Only flag the dependency when base resources alone are inadequate.
            base_package = {"added_servicers": 0, "added_inventory": {}}
            base_capacity, base_inventory, _ = _resource_at_deadline(
                state, base_package, scenario, deadline_s)
            needs_package = used_service_s + required_service_s > base_capacity or any(
                used_inventory[name] + float(quantity) > base_inventory.get(name, 0.0)
                for name, quantity in required_inventory.items())
            if needs_package:
                failures.append(f"launch dependency {package['launch_dependency']} did not occur")
        if used_service_s + required_service_s > capacity_s:
            failures.append("insufficient servicer time before deadline")
        for commodity, quantity in required_inventory.items():
            if used_inventory[commodity] + float(quantity) > inventory.get(commodity, 0.0):
                failures.append(f"insufficient {commodity} inventory")
        success[sla["sla_id"]] = not failures
        reasons[sla["sla_id"]] = failures
        if not failures:
            used_service_s += required_service_s
            for commodity, quantity in required_inventory.items():
                used_inventory[commodity] += float(quantity)

    return {
        "scenario_id": scenario["scenario_id"],
        "probability": float(scenario["probability"]),
        "success": success,
        "failure_reasons": reasons,
        "used_service_s": used_service_s,
        "used_inventory": dict(used_inventory),
    }


def _evaluate_option(
    problem: Mapping[str, Any], package: Mapping[str, Any], include_candidates: bool,
    cost_table_bundle=None,
) -> dict[str, Any]:
    accepted = list(problem.get("accepted_slas", []))
    candidates = list(problem.get("candidate_slas", [])) if include_candidates else []
    slas = accepted + candidates
    outcomes = []
    for scenario in problem["scenarios"]:
        digest = hashlib.blake2b(
            f"{package['package_id']}|{scenario['scenario_id']}|{include_candidates}".encode(),
            digest_size=4).digest()
        outcomes.append(run_mdls_scenario(
            problem, package, scenario, slas,
            seed=int.from_bytes(digest, "big"),
            max_iter=int(problem.get("mdls_iterations", 60)),
            cost_table_bundle=cost_table_bundle))
    reliability = {
        sla["sla_id"]: sum(
            outcome["probability"]
            for outcome in outcomes if outcome["success"].get(sla["sla_id"], False)
        )
        for sla in slas
    }
    requirements = {sla["sla_id"]: float(sla["required_reliability"]) for sla in slas}
    # When reliability is an output rather than a contractual threshold, zero
    # must not count as feasible. At least one modeled scenario must be
    # serviceable before the optimizer may recommend acceptance.
    feasibility_requirements = {
        sla["sla_id"]: (float(sla["required_reliability"])
                        if sla.get("required_reliability_source") != "model_output"
                        else 1e-12)
        for sla in slas
    }
    reliable = all(reliability[sla_id] + 1e-15 >= required
                   for sla_id, required in feasibility_requirements.items())
    contracted_revenue = sum(float(sla.get("revenue", 0.0)) for sla in candidates)
    expected_revenue = sum(
        outcome["probability"] * sum(
            float(sla.get("revenue", 0.0))
            for sla in candidates if outcome["success"].get(sla["sla_id"], False)
        )
        for outcome in outcomes
    )
    expected_penalty = sum(
        outcome["probability"] * sum(
            float(sla.get("failure_penalty", 0.0))
            for sla in slas if not outcome["success"].get(sla["sla_id"], False)
        )
        for outcome in outcomes
    )
    package_cost = float(package.get("fixed_cost", 0.0))
    expected_operation_cost = sum(
        outcome["probability"] * outcome["routing_objectives"]["delta_v_m_s"]
        * float(problem.get("delta_v_cost_per_m_s", 1000.0))
        for outcome in outcomes
    )
    expected_net_value = expected_revenue - package_cost - expected_penalty - expected_operation_cost
    failure_drivers: dict[str, float] = defaultdict(float)
    for outcome in outcomes:
        for reason_list in outcome["failure_reasons"].values():
            for reason in reason_list:
                failure_drivers[reason] += outcome["probability"]
    return {
        "package_id": package["package_id"],
        "package_name": package.get("name", package["package_id"]),
        "item_ids": package.get("item_ids", []),
        "manifest": package.get("manifest", []),
        "include_candidates": include_candidates,
        "reliable": reliable,
        "reliability": reliability,
        "requirements": requirements,
        "candidate_revenue": contracted_revenue,
        "expected_revenue": expected_revenue,
        "package_cost": package_cost,
        "expected_penalty": expected_penalty,
        "expected_operation_cost": expected_operation_cost,
        "expected_net_value": expected_net_value,
        "failure_drivers": [
            {"reason": reason, "probability_weight": weight}
            for reason, weight in sorted(failure_drivers.items(), key=lambda x: -x[1])
        ],
        "scenario_outcomes": outcomes,
        "launch_dependency": package.get("launch_dependency"),
        "launch_dependencies": package.get("launch_dependencies", []),
    }


def evaluate_portfolio(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the accepted baseline with candidate-pool architecture options."""
    _validate(problem)
    packages = construct_manifest_packages(problem)
    if not packages:
        raise ValueError("at least one architecture package is required")
    baseline_package = next(
        (package for package in packages if package.get("is_baseline", False)), packages[0])
    accepted_slas = list(problem.get("accepted_slas", []))
    full_slas = accepted_slas + list(problem.get("candidate_slas", []))
    baseline_table = build_edelbaum_cost_table(problem, accepted_slas)
    full_table = build_edelbaum_cost_table(problem, full_slas)
    baseline = _evaluate_option(
        problem, baseline_package, include_candidates=False,
        cost_table_bundle=baseline_table)
    options = [_evaluate_option(
        problem, package, include_candidates=True,
        cost_table_bundle=full_table) for package in packages]
    protected_ids = {sla["sla_id"] for sla in problem.get("accepted_slas", [])}
    candidate_ids = {sla["sla_id"] for sla in problem.get("candidate_slas", [])}

    milp = select_architecture(options)
    selected = next(
        (option for option in options if option["package_id"] == milp["selected_package_id"]), None)
    feasible = [option for option in options if option["reliable"]]
    if not baseline["reliable"]:
        recommendation = "RECOVER_EXISTING_PORTFOLIO"
        explanation = "The current architecture does not protect all accepted SLAs."
    elif selected is not None:
        dependencies = selected.get("launch_dependencies", []) or (
            [selected["launch_dependency"]] if selected.get("launch_dependency") else [])
        recommendation = "ACCEPT_WITH_EXPANSION" if dependencies else "ACCEPT"
        explanation = (
            f"Accept with {selected['package_name']}; reliability requirements are met "
            f"and expected net value is {selected['expected_net_value']:,.0f}."
        )
        if dependencies:
            explanation += f" The plan depends on launch event(s) {', '.join(dependencies)}."
    elif feasible:
        selected = max(feasible, key=lambda x: x["expected_net_value"])
        recommendation = "REJECT"
        explanation = "A reliable plan exists, but its expected net value is not positive."
    else:
        recommendation = "DEFER_OR_REJECT"
        explanation = "No tested architecture package meets every SLA reliability requirement."

    selected_package = next(
        (package for package in packages
         if selected is not None and package["package_id"] == selected["package_id"]), None)
    deployment_calendar = []
    resupply_calendar = []
    if selected_package is not None:
        launch_lookup = {x["launch_event"]: x for x in problem.get("launch_opportunities", ())}
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in selected_package.get("manifest", []):
            launch_event = item.get("launch_event", selected_package.get("launch_dependency"))
            if launch_event:
                groups.setdefault(launch_event, []).append(item)
        for launch_event, manifest in groups.items():
            launch = launch_lookup.get(launch_event, {})
            deployment = {
                "launch_event": launch_event,
                "booking_deadline_s": launch.get("booking_deadline_s", selected_package.get("booking_deadline_s")),
                "planned_launch_s": launch.get("planned_launch_s", selected_package.get("planned_launch_s")),
                "available_from_s": launch.get("available_from_s", selected_package.get("available_from_s")),
                "manifest": manifest,
            }
            deployment_calendar.append(deployment)
        for deployment in deployment_calendar:
            for item in deployment["manifest"]:
                if item.get("payload_type") == "commodity":
                    resupply_calendar.append({
                        "launch_event": deployment["launch_event"],
                        "available_from_s": deployment["available_from_s"],
                        **item,
                    })

    return {
        "problem_id": problem.get("problem_id", "unnamed"),
        "as_of": problem["as_of"],
        "recommendation": recommendation,
        "explanation": explanation,
        "selected_package_id": None if selected is None else selected["package_id"],
        "baseline": baseline,
        "options": options,
        "milp": milp,
        "deployment_calendar": deployment_calendar,
        "resupply_calendar": resupply_calendar,
        "next_review_s": min(
            [float(x["booking_deadline_s"]) for x in deployment_calendar
             if x.get("booking_deadline_s") is not None] or
            [min(float(sla["deadline_s"]) for sla in problem.get("candidate_slas", []))]
        ) if problem.get("candidate_slas") else None,
        "protected_sla_ids": sorted(protected_ids),
        "candidate_sla_ids": sorted(candidate_ids),
        "model_scope": {
            "planner": "manifest-milp-mdls-portfolio-v0.2",
            "included": ["binary launch-manifest MILP", "Edelbaum/J2 transfer table", "MDLS orbital routing", "scenario reliability", "commodity inventory", "launch-dependent assets", "economics"],
            "deferred": ["route-level inventory pickup visits", "continuous launch-capacity pricing"],
        },
    }
