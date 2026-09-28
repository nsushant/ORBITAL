"""Compile OOS SLA DSL v0.1 JSON into typed, immutable Python objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from .validate_dsl import semantic_errors


SCHEMA_PATH = Path(__file__).parent / "schema" / "sla-dsl-v0.1.schema.json"


class DSLCompileError(ValueError):
    """A problem bundle is not a valid OOS SLA DSL v0.1 document."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid SLA DSL document:\n  - " + "\n  - ".join(errors))


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str


@dataclass(frozen=True)
class Money:
    value: Decimal
    currency: str


@dataclass(frozen=True)
class Orbit:
    reference: str
    semimajor_axis_km: float
    inclination_rad: float
    raan_rad: float
    epoch_s: float = 0.0


@dataclass(frozen=True)
class ComparisonRule:
    metric: str
    operator: str
    value: float | str | bool
    unit: str | None = None


@dataclass(frozen=True)
class AllRule:
    rules: tuple["Rule", ...]


@dataclass(frozen=True)
class AnyRule:
    rules: tuple["Rule", ...]


@dataclass(frozen=True)
class NotRule:
    rule: "Rule"


Rule: TypeAlias = ComparisonRule | AllRule | AnyRule | NotRule


@dataclass(frozen=True)
class CommodityDeliveryService:
    commodity: str
    quantity: Quantity
    nominal_service_duration_s: float
    required_capabilities: tuple[str, ...]
    type: str = "commodity_delivery"


@dataclass(frozen=True)
class PartRequirement:
    commodity: str
    quantity: Quantity


@dataclass(frozen=True)
class RepairService:
    task_code: str
    parts: tuple[PartRequirement, ...]
    required_capabilities: tuple[str, ...]
    nominal_service_duration_s: float | None
    type: str = "repair"


@dataclass(frozen=True)
class DeorbitService:
    maximum_terminal_perigee_altitude_km: float
    required_capabilities: tuple[str, ...]
    nominal_service_duration_s: float | None
    type: str = "deorbit"


Service: TypeAlias = CommodityDeliveryService | RepairService | DeorbitService


@dataclass(frozen=True)
class Client:
    client_id: str
    target_orbit: Orbit


@dataclass(frozen=True)
class ServiceWindow:
    earliest_s: float
    latest_s: float


@dataclass(frozen=True)
class ReliabilityRequirement:
    measure: str
    minimum: float
    scenario_model: str


@dataclass(frozen=True)
class CommercialTerms:
    revenue: Money
    failure_penalty: Money
    late_penalty_per_second: Money | None
    cancellation_penalty: Money | None


@dataclass(frozen=True)
class SLA:
    sla_id: str
    customer_id: str
    service: Service
    client: Client
    service_window: ServiceWindow
    success: Rule
    reliability: ReliabilityRequirement
    commercial_terms: CommercialTerms


@dataclass(frozen=True)
class Propulsion:
    type: str
    isp_s: float
    thrust_n: float


@dataclass(frozen=True)
class Servicer:
    servicer_id: str
    orbit: Orbit
    dry_mass_kg: float
    propellant_kg: float
    maximum_payload_kg: float
    capabilities: tuple[str, ...]
    available_from_s: float
    propulsion: Propulsion


@dataclass(frozen=True)
class InventoryItem:
    commodity: str
    quantity: Quantity
    reserved: Quantity | None


@dataclass(frozen=True)
class Depot:
    depot_id: str
    orbit: Orbit
    commissioned_from_s: float
    inventory: tuple[InventoryItem, ...]


@dataclass(frozen=True)
class ProviderState:
    state_id: str
    observation_time_s: float
    servicers: tuple[Servicer, ...]
    depots: tuple[Depot, ...]
    existing_commitments: tuple[str, ...]


@dataclass(frozen=True)
class LaunchAccess:
    model_id: str
    model_type: str
    information_cutoff_s: float
    horizon_end_s: float
    generated_scenarios: int
    marks: tuple[str, ...]
    capacity_evidence_allowed: tuple[str, ...]
    random_seed: int


@dataclass(frozen=True)
class Decision:
    stage: str
    required: bool = False


@dataclass(frozen=True)
class FreezeRule:
    decision: str
    relative_to: str
    offset_s: float


@dataclass(frozen=True)
class DecisionPolicy:
    policy_id: str
    decisions: Mapping[str, Decision]
    freeze_rules: tuple[FreezeRule, ...]


@dataclass(frozen=True)
class ModelReferences:
    transfer: str
    planning: str
    service: str


@dataclass(frozen=True)
class Metadata:
    title: str | None
    description: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SLAProblem:
    dsl_version: str
    problem_id: str
    as_of: datetime
    sla: SLA
    provider_state: ProviderState
    launch_access: LaunchAccess
    decision_policy: DecisionPolicy
    models: ModelReferences
    metadata: Metadata
    source: Path | None = None

    def candidate_servicers(self) -> tuple[Servicer, ...]:
        required = set(self.sla.service.required_capabilities)
        return tuple(s for s in self.provider_state.servicers
                     if required.issubset(s.capabilities))

    def at_seconds(self, seconds: float) -> datetime:
        """Convert seconds relative to `as_of` into an absolute UTC time."""
        return self.as_of + timedelta(seconds=float(seconds))

    def seconds_at(self, timestamp: datetime) -> float:
        """Convert an aware timestamp into seconds relative to `as_of`."""
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return (timestamp.astimezone(timezone.utc) - self.as_of).total_seconds()


def _quantity(data: Mapping[str, Any] | None) -> Quantity | None:
    return None if data is None else Quantity(float(data["value"]), data["unit"])


def _money(data: Mapping[str, Any] | None) -> Money | None:
    return None if data is None else Money(Decimal(str(data["value"])), data["currency"])


def _orbit(data: Mapping[str, Any]) -> Orbit:
    return Orbit(data["reference"], float(data["semimajor_axis_km"]),
                 float(data["inclination_rad"]), float(data["raan_rad"]),
                 float(data.get("epoch_s", 0.0)))


def _rule(data: Mapping[str, Any]) -> Rule:
    if "all" in data:
        return AllRule(tuple(_rule(r) for r in data["all"]))
    if "any" in data:
        return AnyRule(tuple(_rule(r) for r in data["any"]))
    if "not" in data:
        return NotRule(_rule(data["not"]))
    return ComparisonRule(data["metric"], data["operator"], data["value"], data.get("unit"))


def _service(data: Mapping[str, Any]) -> Service:
    required = tuple(data.get("required_capabilities", ()))
    if data["type"] == "commodity_delivery":
        return CommodityDeliveryService(data["commodity"], _quantity(data["quantity"]),
                                        float(data["nominal_service_duration_s"]), required)  # type: ignore[arg-type]
    if data["type"] == "repair":
        parts = tuple(PartRequirement(p["commodity"], _quantity(p["quantity"])) for p in data["parts"])  # type: ignore[arg-type]
        duration = data.get("nominal_service_duration_s")
        return RepairService(data["task_code"], parts, required,
                             None if duration is None else float(duration))
    if data["type"] == "deorbit":
        duration = data.get("nominal_service_duration_s")
        return DeorbitService(float(data["maximum_terminal_perigee_altitude_km"]), required,
                              None if duration is None else float(duration))
    raise DSLCompileError([f"unsupported service type {data['type']!r}"])


def _compile(data: Mapping[str, Any], source: Path | None) -> SLAProblem:
    s = data["sla"]
    terms = s["commercial_terms"]
    state = data["provider_state"]
    launch = data["launch_access"]
    policy = data["decision_policy"]
    meta = data.get("metadata", {})

    servicers = tuple(Servicer(
        x["servicer_id"], _orbit(x["orbit"]), float(x["dry_mass_kg"]),
        float(x["propellant_kg"]), float(x["maximum_payload_kg"]),
        tuple(x["capabilities"]), float(x["available_from_s"]),
        Propulsion(x["propulsion"]["type"], float(x["propulsion"]["isp_s"]),
                   float(x["propulsion"]["thrust_n"]))
    ) for x in state["servicers"])
    depots = tuple(Depot(
        x["depot_id"], _orbit(x["orbit"]), float(x["commissioned_from_s"]),
        tuple(InventoryItem(i["commodity"], _quantity(i["quantity"]),
                            _quantity(i.get("reserved"))) for i in x["inventory"])
    ) for x in state["depots"])

    return SLAProblem(
        data["dsl_version"], data["problem_id"], _datetime(data["as_of"]),
        SLA(
            s["sla_id"], s["customer_id"], _service(s["service"]),
            Client(s["client"]["client_id"], _orbit(s["client"]["target_orbit"])),
            ServiceWindow(float(s["service_window"]["earliest_s"]),
                          float(s["service_window"]["latest_s"])),
            _rule(s["success"]),
            ReliabilityRequirement(s["reliability"]["measure"],
                                   float(s["reliability"]["minimum"]),
                                   s["reliability"]["scenario_model"]),
            CommercialTerms(_money(terms["revenue"]), _money(terms["failure_penalty"]),
                            _money(terms.get("late_penalty_per_second")),
                            _money(terms.get("cancellation_penalty")))  # type: ignore[arg-type]
        ),
        ProviderState(state["state_id"], float(state["observation_time_s"]),
                      servicers, depots, tuple(state["existing_commitments"])),
        LaunchAccess(launch["model_id"], launch["model_type"],
                     float(launch["information_cutoff_s"]), float(launch["horizon_end_s"]),
                     int(launch["generated_scenarios"]), tuple(launch["marks"]),
                     tuple(launch["capacity_evidence_allowed"]), int(launch["random_seed"])),
        DecisionPolicy(policy["policy_id"],
                       {name: Decision(x["stage"], x.get("required", False))
                        for name, x in policy["decisions"].items()},
                       tuple(FreezeRule(x["decision"], x["relative_to"],
                                        float(x["offset_s"])) for x in policy["freeze_rules"])),
        ModelReferences(**data["models"]),
        Metadata(meta.get("title"), meta.get("description"), tuple(meta.get("tags", ()))),
        source,
    )


def validate_document(data: Mapping[str, Any], schema_path: Path = SCHEMA_PATH) -> None:
    """Run structural and semantic validation, raising one aggregated error."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("Install jsonschema>=4.18 to compile SLA DSL documents") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    structural = [f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                  for e in validator.iter_errors(data)]
    semantic = semantic_errors(dict(data)) if not structural else []
    errors = structural + semantic
    if errors:
        raise DSLCompileError(errors)


def compile_document(data: Mapping[str, Any], *, validate: bool = True,
                     schema_path: Path = SCHEMA_PATH, source: Path | None = None) -> SLAProblem:
    if validate:
        validate_document(data, schema_path)
    return _compile(data, source)


def load_problem(path: str | Path, *, validate: bool = True,
                 schema_path: Path = SCHEMA_PATH) -> SLAProblem:
    source = Path(path).resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DSLCompileError([f"cannot read {source}: {exc}"]) from exc
    return compile_document(data, validate=validate, schema_path=schema_path, source=source)
