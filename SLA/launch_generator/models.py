"""Versioned launch-access data objects using seconds relative to `as_of`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    if utc.utcoffset() != timedelta(0):
        raise ValueError("as_of must resolve to UTC")
    return utc


@dataclass(frozen=True)
class EvidenceAssertion:
    evidence_id: str
    source_url: str
    source_type: str
    published_s: float
    subject_id: str
    claim_type: str
    fields: Mapping[str, Any]
    confidence: float
    status: str = "extracted"

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must lie in [0, 1]")
        if self.status not in {"extracted", "human_verified", "rejected"}:
            raise ValueError(f"invalid evidence status {self.status!r}")


@dataclass(frozen=True)
class LaunchOpportunity:
    opportunity_id: str
    launch_time_s: float
    site: str
    provider: str
    vehicle: str
    orbit_class: str
    semimajor_axis_km: float | None
    inclination_rad: float | None
    projected_purchasable_capacity_kg: float
    price: float | None
    currency: str | None
    booking_deadline_s: float
    evidence_class: str
    occurrence_probability: float
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.launch_time_s < self.booking_deadline_s:
            raise ValueError("booking deadline must not follow launch time")
        if self.projected_purchasable_capacity_kg < 0:
            raise ValueError("projected capacity must be non-negative")
        if not 0.0 <= self.occurrence_probability <= 1.0:
            raise ValueError("occurrence probability must lie in [0, 1]")
        if self.evidence_class not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid capacity evidence class {self.evidence_class!r}")
        if (self.price is None) != (self.currency is None):
            raise ValueError("price and currency must be present or absent together")


@dataclass(frozen=True)
class LaunchScenario:
    scenario_id: str
    probability: float
    opportunities: tuple[LaunchOpportunity, ...]

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("scenario probability must lie in [0, 1]")
        times = [x.launch_time_s for x in self.opportunities]
        if times != sorted(times):
            raise ValueError("scenario opportunities must be sorted by launch_time_s")


@dataclass(frozen=True)
class ForecastSnapshot:
    forecast_id: str
    as_of: datetime
    horizon_s: float
    model_version: str
    evidence_snapshot_id: str
    random_seed: int
    scenarios: tuple[LaunchScenario, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "as_of", require_utc(self.as_of))
        if self.horizon_s <= 0:
            raise ValueError("forecast horizon_s must be positive")
        total = sum(x.probability for x in self.scenarios)
        if self.scenarios and abs(total - 1.0) > 1e-9:
            raise ValueError(f"scenario probabilities must sum to one, got {total}")
        for scenario in self.scenarios:
            if any(x.launch_time_s > self.horizon_s for x in scenario.opportunities):
                raise ValueError("scenario contains a launch beyond the forecast horizon")

    def at_seconds(self, seconds: float) -> datetime:
        return self.as_of + timedelta(seconds=float(seconds))

    def seconds_at(self, timestamp: datetime) -> float:
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return (timestamp.astimezone(timezone.utc) - self.as_of).total_seconds()


@dataclass(frozen=True)
class RealizedLaunch:
    realization_id: str
    matched_opportunity_id: str | None
    occurred: bool
    actual_launch_time_s: float | None
    actual_site: str | None
    actual_provider: str | None
    actual_vehicle: str | None
    actual_orbit_class: str | None
    observed_purchasable_capacity_kg: float | None
    observation_quality: str

    def __post_init__(self):
        if self.occurred and self.actual_launch_time_s is None:
            raise ValueError("an occurred launch requires actual_launch_time_s")
        if self.observed_purchasable_capacity_kg is not None and self.observed_purchasable_capacity_kg < 0:
            raise ValueError("observed capacity must be non-negative")
        if self.observation_quality not in {"observed", "partially_observed", "unknown"}:
            raise ValueError(f"invalid observation quality {self.observation_quality!r}")
