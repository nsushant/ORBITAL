"""Compiler for the `commodity-delivery-v1` service plugin."""

from __future__ import annotations

from dataclasses import dataclass

from ..dsl.compiler import CommodityDeliveryService, Depot, SLAProblem, Servicer


class ServiceCompileError(ValueError):
    pass


@dataclass(frozen=True)
class DepotInventoryCandidate:
    depot: Depot
    available_quantity: float
    unit: str


@dataclass(frozen=True)
class CommodityDeliveryRequirements:
    sla_id: str
    commodity: str
    quantity: float
    unit: str
    earliest_start_s: float
    completion_deadline_s: float
    service_duration_s: float
    required_capabilities: tuple[str, ...]
    candidate_servicers: tuple[Servicer, ...]
    inventory_candidates: tuple[DepotInventoryCandidate, ...]
    inventory_shortfall: float

    @property
    def needs_resupply(self) -> bool:
        return self.inventory_shortfall > 0.0


def _available_at_depot(depot: Depot, commodity: str, unit: str) -> float:
    available = 0.0
    for item in depot.inventory:
        if item.commodity != commodity:
            continue
        if item.quantity.unit != unit:
            raise ServiceCompileError(
                f"depot {depot.depot_id!r} stores {commodity!r} in "
                f"{item.quantity.unit!r}, expected {unit!r}"
            )
        reserved = 0.0 if item.reserved is None else item.reserved.value
        if item.reserved is not None and item.reserved.unit != unit:
            raise ServiceCompileError(
                f"reserved inventory unit mismatch at depot {depot.depot_id!r}"
            )
        available += max(0.0, item.quantity.value - reserved)
    return available


def compile_commodity_delivery(problem: SLAProblem) -> CommodityDeliveryRequirements:
    service = problem.sla.service
    if not isinstance(service, CommodityDeliveryService):
        raise ServiceCompileError(
            f"commodity-delivery-v1 cannot compile service type {service.type!r}"
        )

    required = service.quantity.value
    unit = service.quantity.unit
    servicers = tuple(
        s for s in problem.candidate_servicers()
        if unit != "kg" or s.maximum_payload_kg >= required
    )
    if not servicers:
        raise ServiceCompileError(
            "no compatible servicer has the required capabilities and payload capacity"
        )

    candidates = tuple(
        DepotInventoryCandidate(d, q, unit)
        for d in problem.provider_state.depots
        if (q := _available_at_depot(d, service.commodity, unit)) > 0.0
    )
    total_available = sum(x.available_quantity for x in candidates)
    return CommodityDeliveryRequirements(
        sla_id=problem.sla.sla_id,
        commodity=service.commodity,
        quantity=required,
        unit=unit,
        earliest_start_s=problem.sla.service_window.earliest_s,
        completion_deadline_s=problem.sla.service_window.latest_s,
        service_duration_s=service.nominal_service_duration_s,
        required_capabilities=service.required_capabilities,
        candidate_servicers=servicers,
        inventory_candidates=candidates,
        inventory_shortfall=max(0.0, required - total_available),
    )
