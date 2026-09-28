"""MDLS routing adapter for the product MVP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from ..edelbaum import j2_raan_rate, transfer_cost
from ..optimization.greedy import greedy_schedule
from ..optimization.mdls import mdls
from ..schedule import CostTable, Demands


DAY = 86_400.0


def _angle_distance(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def build_edelbaum_cost_table(problem: Mapping[str, Any], slas: list[Mapping[str, Any]]) -> tuple[CostTable, dict[str, int]]:
    """Build the compact, time-dependent Edelbaum/J2 table used by MDLS."""
    depot_orbit = problem["current_state"].get(
        "depot_orbit", {"semimajor_axis_km": 6978.137, "inclination_rad": 0.925, "raan_rad": 0.0})
    names = ["DEPOT"] + [sla.get("target_asset_id", sla["sla_id"]) for sla in slas]
    orbits = [depot_orbit] + [sla.get("target_orbit", depot_orbit) for sla in slas]
    node_index = {sla["sla_id"]: index + 1 for index, sla in enumerate(slas)}
    horizon_s = max(float(sla["deadline_s"]) for sla in slas) + 30 * DAY
    dep_s = np.arange(0.0, horizon_s + 30 * DAY, 30 * DAY)
    tof_s = np.asarray([30, 60, 90, 120, 180, 240], dtype=float) * DAY
    n = len(names)
    dv = np.full((n, n, len(dep_s), len(tof_s)), np.nan, dtype=float)
    phasing = np.zeros_like(dv)
    for i, origin in enumerate(orbits):
        for j, target in enumerate(orbits):
            if i == j:
                dv[i, j, :, :] = 0.0
                continue
            state1 = np.asarray([
                float(origin["semimajor_axis_km"]), float(origin["inclination_rad"]),
                float(origin.get("raan_rad", 0.0)), 335.0, 2800.0, 1e-5])
            state2 = np.asarray([
                float(target["semimajor_axis_km"]), float(target["inclination_rad"]),
                float(target.get("raan_rad", 0.0)), 335.0, 2800.0, 1e-5])
            tof_flat = np.tile(tof_s, len(dep_s))
            departure_flat = np.repeat(dep_s, len(tof_s))
            source_rate = j2_raan_rate(state1[0], state1[1])
            target_rate = j2_raan_rate(state2[0], state2[1])
            required_raan = (
                state2[2] + target_rate * (departure_flat + tof_flat)
                - state1[2] - source_rate * departure_flat)
            costs_km_s = transfer_cost(
                state1, state2, tof_flat, required_raan,
                n_scan=int(problem.get("edelbaum_n_scan", 32)),
                n_growth=int(problem.get("edelbaum_n_growth", 20)),
                growth_rtol=float(problem.get("edelbaum_growth_rtol", 5e-3)))
            dv[i, j] = costs_km_s.reshape(len(dep_s), len(tof_s)) * 1000.0
    return CostTable(
        dv=dv,
        phasing=phasing,
        names=names,
        dep_s=dep_s,
        tof_s=tof_s,
        plane_of_node=np.arange(n, dtype=np.int64),
        dv_budget=float(problem["current_state"].get("dv_budget_m_s", 2500.0)),
    ), node_index


def run_mdls_scenario(
    problem: Mapping[str, Any],
    package: Mapping[str, Any],
    scenario: Mapping[str, Any],
    slas: list[Mapping[str, Any]],
    *,
    seed: int,
    max_iter: int = 60,
    cost_table_bundle: tuple[CostTable, dict[str, int]] | None = None,
) -> dict[str, Any]:
    manifest = list(package.get("manifest", ()))
    successful_manifest = [
        item for item in manifest
        if bool(scenario.get("launch_outcomes", {}).get(item.get("launch_event"), True))]
    base_vehicles = int(problem["current_state"].get("servicer_count", 0))
    added_vehicles = sum(int(item.get("quantity", 1)) for item in successful_manifest
                         if item.get("payload_type") == "servicer")
    max_vehicles = max(1, base_vehicles + added_vehicles)
    availability = max(1e-6, float(scenario.get("servicer_availability", 1.0)))

    cost_table, node_index = cost_table_bundle or build_edelbaum_cost_table(problem, slas)
    protected = {sla["sla_id"] for sla in problem.get("accepted_slas", [])}
    demands = Demands(
        node=np.asarray([node_index[sla["sla_id"]] for sla in slas], dtype=np.int64),
        release=np.asarray([float(sla.get("release_s", 0.0)) for sla in slas]),
        deadline=np.asarray([float(sla["deadline_s"]) for sla in slas]),
        service=np.asarray([float(sla["service_duration_s"]) / availability for sla in slas]),
        value=np.asarray([
            (1.0e12 + float(sla.get("failure_penalty", 0.0)))
            if sla["sla_id"] in protected
            else float(sla.get("revenue", 0.0)) + float(sla.get("failure_penalty", 0.0))
            for sla in slas
        ]),
    )
    initial, _ = greedy_schedule(
        cost_table, demands, 0, max_vehicles, 0.5 * DAY,
        dv_budget=cost_table.dv_budget)
    if not len(initial.node):
        return {
            "scenario_id": scenario["scenario_id"], "probability": float(scenario["probability"]),
            "success": {sla["sla_id"]: False for sla in slas},
            "failure_reasons": {sla["sla_id"]: ["no feasible orbital route"] for sla in slas},
            "used_service_s": 0.0, "used_inventory": {}, "routes": [],
            "routing_objectives": {"delta_v_m_s": 0.0, "vehicles": 0, "unrecovered_value": float(demands.value.sum())},
            "mdls_evaluations": 0, "route_model": "mdls-edelbaum-j2-v0.2",
        }
    archive, evaluations, search_archive = mdls(
        cost_table, demands, 0, max_vehicles=max_vehicles,
        refuel_time=0.5 * DAY, max_iter=max_iter, seed=seed,
        shift=7 * DAY, dv_budget=cost_table.dv_budget, init=[initial],
        return_search_archive=True,
        search_archive_capacity=int(problem.get("search_archive_capacity", 512)))
    objectives = archive.objectives
    chosen = int(np.lexsort((objectives[:, 0], objectives[:, 1], objectives[:, 2]))[0])
    schedule = archive.schedules[chosen]
    routed = set(int(uid) for uid in schedule.served())

    inventory = {
        name: float(quantity) * float(scenario.get("inventory_multiplier", 1.0))
        for name, quantity in problem["current_state"].get("inventory", {}).items()
    }
    used_inventory: dict[str, float] = defaultdict(float)
    success: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}
    allocation_order = sorted(
        range(len(slas)), key=lambda uid: (slas[uid]["sla_id"] not in protected, float(slas[uid]["deadline_s"])))
    for uid in allocation_order:
        sla = slas[uid]
        failures = []
        if uid not in routed:
            failures.append("no route selected by MDLS")
        for commodity, required in sla.get("inventory", {}).items():
            supply = inventory.get(commodity, 0.0)
            for item in successful_manifest:
                if (item.get("payload_type") == "commodity"
                        and item.get("commodity") == commodity
                        and float(sla["deadline_s"]) >= float(item.get("available_from_s", 0.0))):
                    supply += float(item.get("quantity", 0.0))
            if used_inventory[commodity] + float(required) > supply:
                failures.append(f"insufficient {commodity} inventory")
        success[sla["sla_id"]] = not failures
        reasons[sla["sla_id"]] = failures
        if not failures:
            for commodity, required in sla.get("inventory", {}).items():
                used_inventory[commodity] += float(required)

    routes = []
    for vehicle in range(schedule.n_vehicles):
        visits = []
        lo, hi = int(schedule.veh_start[vehicle]), int(schedule.veh_start[vehicle + 1])
        for k in range(lo, hi):
            uid = int(schedule.uid[k])
            visits.append({
                "target": "DEPOT" if uid < 0 else slas[uid]["sla_id"],
                "arrival_s": float(schedule.arrival[k]),
                "departure_s": float(schedule.departure[k]),
                "leg_delta_v_m_s": float(schedule.cost[k]),
            })
        routes.append({"vehicle": vehicle + 1, "visits": visits})
    f = objectives[chosen]
    return {
        "scenario_id": scenario["scenario_id"],
        "probability": float(scenario["probability"]),
        "success": success,
        "failure_reasons": reasons,
        "used_service_s": float(sum(demands.service[uid] for uid in routed)),
        "used_inventory": dict(used_inventory),
        "routes": routes,
        "routing_objectives": {"delta_v_m_s": float(f[0]), "vehicles": int(round(f[1])), "unrecovered_value": float(f[2])},
        "mdls_evaluations": evaluations,
        "search_archive": {
            "evaluated": evaluations,
            "retained": len(search_archive.records),
            "solutions": search_archive.summaries([sla["sla_id"] for sla in slas]),
        },
        "route_model": "mdls-edelbaum-j2-v0.2",
    }
