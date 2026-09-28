"""Typed inputs and outputs for the SLA demand forecast.

All times are seconds relative to the forecast snapshot's timezone-aware UTC
``as_of`` timestamp.  These objects deliberately contain only the marks needed
by the SLA acceptance demonstrator; they do not prescribe an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from ..launch_generator.models import LaunchScenario, require_utc


@dataclass(frozen=True)
class SatelliteAsset:
    asset_id: str
    orbit_class: str
    semimajor_axis_km: float
    inclination_rad: float
    age_at_as_of_s: float
    activation_s: float = 0.0
    design_life_s: float | None = None
    health_multiplier: float = 1.0
    operator: str | None = None

    def __post_init__(self) -> None:
        if self.activation_s < 0 or self.age_at_as_of_s < 0:
            raise ValueError("activation and age must be non-negative seconds")
        if self.design_life_s is not None and self.design_life_s <= 0:
            raise ValueError("design_life_s must be positive")
        if self.health_multiplier < 0:
            raise ValueError("health_multiplier must be non-negative")

    def age_at(self, time_s: float) -> float:
        return max(0.0, self.age_at_as_of_s + time_s - self.activation_s)


@dataclass(frozen=True)
class DemandRequest:
    request_id: str
    target_asset_id: str
    orbit_class: str
    semimajor_axis_km: float
    inclination_rad: float
    service_type: str
    release_s: float
    deadline_s: float
    service_duration_s: float
    value: float
    inventory: Mapping[str, float] = field(default_factory=dict)
    priority: int = 0
    source: str = "generated"

    def __post_init__(self) -> None:
        if min(self.release_s, self.service_duration_s, self.value) < 0:
            raise ValueError("request times, duration and value must be non-negative")
        if self.deadline_s < self.release_s:
            raise ValueError("deadline_s must not precede release_s")
        if any(quantity < 0 for quantity in self.inventory.values()):
            raise ValueError("inventory requirements must be non-negative")


@dataclass(frozen=True)
class PipelineOrder:
    request: DemandRequest
    conversion_probability: float

    def __post_init__(self) -> None:
        if not 0 <= self.conversion_probability <= 1:
            raise ValueError("conversion_probability must lie in [0, 1]")


@dataclass(frozen=True)
class PlannedDeployment:
    """Assets activated only when the linked launch occurs in a scenario."""

    launch_opportunity_id: str
    assets: tuple[SatelliteAsset, ...]
    activation_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.activation_delay_s < 0:
            raise ValueError("activation_delay_s must be non-negative")


@dataclass(frozen=True)
class DemandScenario:
    scenario_id: str
    probability: float
    requests: tuple[DemandRequest, ...]
    active_asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("scenario probability must lie in [0, 1]")
        releases = [request.release_s for request in self.requests]
        if releases != sorted(releases):
            raise ValueError("requests must be sorted by release_s")


@dataclass(frozen=True)
class DemandForecast:
    forecast_id: str
    as_of: datetime
    horizon_s: float
    model_version: str
    random_seed: int
    scenarios: tuple[DemandScenario, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", require_utc(self.as_of))
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        total = sum(s.probability for s in self.scenarios)
        if self.scenarios and abs(total - 1.0) > 1e-9:
            raise ValueError(f"scenario probabilities must sum to one, got {total}")


@dataclass(frozen=True)
class JointPlanningScenario:
    """A coupled launch-calendar and future-demand realization."""

    scenario_id: str
    probability: float
    launch: LaunchScenario
    demand: DemandScenario

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("scenario probability must lie in [0, 1]")
