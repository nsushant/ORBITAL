"""Pareto-derived counterfactuals for rejected SLA decisions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from .planner import evaluate_portfolio


def _inventory_gap(problem: dict[str, Any]) -> float:
    slas = problem.get("accepted_slas", []) + problem.get("candidate_slas", [])
    required = sum(float(x.get("inventory", {}).get("xenon_kg", 0)) for x in slas)
    current = float(problem.get("current_state", {}).get("inventory", {}).get("xenon_kg", 0))
    listed = sum(float(x.get("quantity", 0)) for x in problem.get("manifest_candidates", [])
                 if x.get("payload_type") == "commodity" and x.get("commodity") == "xenon_kg")
    return max(0.0, required - current - listed)


def _variants(problem: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    variants: list[tuple[str, str, dict[str, Any]]] = []
    candidate = problem["candidate_slas"][0]
    count = int(candidate.get("target_count", 1))
    per_sat = (float(candidate.get("inventory", {}).get("xenon_kg", 0)) / count
               if count and candidate.get("propellant_quantity_basis") == "per_satellite" else None)
    protected = sum(float(x.get("inventory", {}).get("xenon_kg", 0))
                    for x in problem.get("accepted_slas", []))
    listed = sum(float(x.get("quantity", 0)) for x in problem.get("manifest_candidates", [])
                 if x.get("commodity") == "xenon_kg")
    current = float(problem["current_state"]["inventory"].get("xenon_kg", 0))

    if per_sat and count > 1:
        serviceable = max(1, min(count, int(max(0, current + listed - protected) // per_sat)))
        for n in sorted({serviceable, max(1, count // 2)}):
            if n >= count:
                continue
            p = deepcopy(problem); c = p["candidate_slas"][0]
            c["target_count"] = n
            c["inventory"]["xenon_kg"] = per_sat * n
            c["revenue"] = float(c.get("revenue", 0)) * n / count
            c["sla_id"] = f"{c['sla_id']}-TRANCHE-{n}"
            variants.append((f"Serve {n} satellites first", "scope", p))

    p = deepcopy(problem); gap = _inventory_gap(p)
    if gap > 0:
        launch = p["launch_opportunities"][0]
        p["manifest_candidates"].append({
            "item_id": "COUNTERFACTUAL-XE", "name": f"{gap:g} kg additional xenon",
            "payload_type": "commodity", "commodity": "xenon_kg", "quantity": gap,
            "mass_kg": gap, "destination": "DEPOT", "launch_event": launch["launch_event"],
            "available_from_s": launch["available_from_s"], "cost": gap * 50000,
        })
        launch["capacity_kg"] = max(float(launch["capacity_kg"]),
                                    sum(float(x.get("mass_kg", 0)) for x in p["manifest_candidates"]))
        variants.append(("Resource the full constellation", "architecture", p))

    p = deepcopy(problem); p["candidate_slas"][0]["deadline_s"] += 180 * 86400
    variants.append(("Extend the service window by 180 days", "deadline", p))
    return variants


def _point(label: str, kind: str, problem: dict[str, Any], result: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    selected_id = result.get("selected_package_id")
    option = next((x for x in result.get("options", []) if x["package_id"] == selected_id), None)
    if option is None and result.get("options"):
        option = max(result["options"], key=lambda x: (x["expected_net_value"], sum(x["reliability"].values())))
    candidate_ids = {x["sla_id"] for x in problem.get("candidate_slas", [])}
    values = [v for k, v in (option or {}).get("reliability", {}).items() if k in candidate_ids]
    reliability = min(values) if values else 0.0
    scope = int(problem["candidate_slas"][0].get("target_count", 1))
    original_deadline = int(original["candidate_slas"][0]["deadline_s"])
    deadline = int(problem["candidate_slas"][0]["deadline_s"])
    return {
        "label": label, "kind": kind, "scope": scope,
        "deadline_extension_s": max(0, deadline - original_deadline),
        "reliability": reliability,
        "investment": float((option or {}).get("package_cost", 0)),
        "expected_net_value": float((option or {}).get("expected_net_value", float("-inf"))),
        "recommendation": result.get("recommendation"), "problem": problem,
    }


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # More scope, reliability and value are preferred; less deadline concession and investment are preferred.
    av = (a["scope"], a["reliability"], a["expected_net_value"], -a["deadline_extension_s"], -a["investment"])
    bv = (b["scope"], b["reliability"], b["expected_net_value"], -b["deadline_extension_s"], -b["investment"])
    return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))


def pareto_counterfactuals(payload: dict[str, Any]) -> dict[str, Any]:
    problem = payload.get("problem", payload)
    base_result = payload.get("result", {})
    archives = [
        outcome.get("search_archive", {})
        for option in base_result.get("options", [])
        for outcome in option.get("scenario_outcomes", [])
        if outcome.get("search_archive")
    ]
    retained = sum(int(x.get("retained", 0)) for x in archives)
    evaluated = sum(int(x.get("evaluated", 0)) for x in archives)
    candidate_ids = {x["sla_id"] for x in problem.get("candidate_slas", [])}
    archive_hits = sum(
        1 for archive in archives for solution in archive.get("solutions", [])
        if solution.get("feasible") and candidate_ids.issubset(set(solution.get("served", [])))
    )
    variants = _variants(problem)
    with ThreadPoolExecutor(max_workers=min(3, len(variants) or 1)) as pool:
        results = list(pool.map(lambda item: evaluate_portfolio(item[2]), variants))
    points = [_point(label, kind, p, result, problem)
              for (label, kind, p), result in zip(variants, results)]
    frontier = [p for p in points if not any(_dominates(q, p) for q in points if q is not p)]
    frontier.sort(key=lambda x: (-x["reliability"], -x["scope"], -x["expected_net_value"]))
    return {
        "method": "archive-first-pareto-counterfactual-v0.2",
        "archive": {"runs": len(archives), "evaluated": evaluated,
                    "retained": retained, "candidate_route_hits": archive_hits},
        "repair_problems_evaluated": len(points),
        "points": frontier,
    }
