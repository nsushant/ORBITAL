"""Exact solver for the MVP's one-hot architecture-package MILP.

The small product demonstrator exposes a finite menu of explainable packages.
The associated binary MILP has one variable per package, ``sum(x) <= 1``, SLA
reliability constraints and a linear expected-value objective. Enumerating its
binary vertices is exact and dependency-free; a future large model can replace
this function with HiGHS or Gurobi without changing the planner contract.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping


def construct_manifest_packages(problem: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate feasible vertices of the small manifest-design MILP.

    Each candidate item is a binary decision. Launch mass capacities and item
    prerequisites are linear constraints. Enumeration is exact for the MVP's
    deliberately small candidate menu and keeps the installation dependency-free.
    """
    items = list(problem.get("manifest_candidates", ()))
    committed = list(problem.get("committed_manifest", ()))
    if not items and not committed:
        return list(problem.get("architecture_packages", ()))
    if len(items) + len(committed) > 16:
        raise ValueError("the dependency-free manifest MILP supports at most 16 candidate items")
    launches = {x["launch_event"]: x for x in problem.get("launch_opportunities", ())}
    packages = []
    for size in range(len(items) + 1):
        for chosen_tuple in combinations(items, size):
            new_items = list(chosen_tuple)
            chosen = committed + new_items
            chosen_ids = {x["item_id"] for x in chosen}
            if any(not set(x.get("requires", ())).issubset(chosen_ids) for x in chosen):
                continue
            mass_by_launch: dict[str, float] = {}
            valid = True
            for item in chosen:
                launch_id = item["launch_event"]
                if launch_id not in launches:
                    raise ValueError(f"manifest item references unknown launch {launch_id!r}")
                mass_by_launch[launch_id] = mass_by_launch.get(launch_id, 0.0) + float(item["mass_kg"])
                if mass_by_launch[launch_id] > float(launches[launch_id]["capacity_kg"]):
                    valid = False
                    break
            if not valid:
                continue
            manifest = []
            added_inventory: dict[str, float] = {}
            added_servicers = 0
            for item in chosen:
                entry = dict(item)
                manifest.append(entry)
                if item["payload_type"] == "commodity":
                    name = item["commodity"]
                    added_inventory[name] = added_inventory.get(name, 0.0) + float(item["quantity"])
                elif item["payload_type"] == "servicer":
                    added_servicers += int(item.get("quantity", 1))
            if new_items:
                label = " + ".join(x.get("name", x["item_id"]) for x in new_items)
            elif committed:
                label = "Committed deployment"
            else:
                label = "Current architecture"
            packages.append({
                "package_id": "CURRENT" if not new_items else "MANIFEST__" + "__".join(sorted(chosen_ids)),
                "name": label,
                "is_baseline": not new_items,
                "fixed_cost": sum(float(x.get("cost", 0.0)) for x in new_items),
                "added_servicers": added_servicers,
                "added_inventory": added_inventory,
                "manifest": manifest,
                "item_ids": sorted(chosen_ids),
                "new_item_ids": sorted(x["item_id"] for x in new_items),
                "launch_dependencies": sorted({x["launch_event"] for x in chosen}),
            })
    return packages


def select_architecture(options: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    options = list(options)
    feasible = [option for option in options if option["reliable"]]
    profitable = [option for option in feasible if option["expected_net_value"] > 0]
    selected = max(profitable, key=lambda option: option["expected_net_value"], default=None)
    return {
        "solver": "exact-enumerated-binary-milp-v0.1",
        "status": "optimal" if selected is not None else "no-profitable-feasible-package",
        "selected_package_id": None if selected is None else selected["package_id"],
        "objective_value": None if selected is None else selected["expected_net_value"],
        "binary_variables": {
            item_id: int(selected is not None and item_id in selected.get("item_ids", ()))
            for item_id in sorted({item for option in options for item in option.get("item_ids", ())})
        },
        "package_binary_variables": {
            option["package_id"]: int(selected is option) for option in options
        },
        "feasible_package_ids": [option["package_id"] for option in feasible],
    }
