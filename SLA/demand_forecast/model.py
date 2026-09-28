"""Interpretable Bayesian demand generator for the research MVP.

The model is a marked, state-dependent mixed Poisson process.  One latent rate
is drawn from each Gamma posterior per scenario.  Satellite age, health and
time in service scale its exposure; request marks are drawn from an explicit
``DemandProfile``.  Gamma-Poisson conjugacy makes learning transparent and
works with sparse event histories.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np

from ..launch_generator.models import ForecastSnapshot, LaunchScenario
from .models import (
    DemandForecast,
    DemandRequest,
    DemandScenario,
    JointPlanningScenario,
    PipelineOrder,
    PlannedDeployment,
    SatelliteAsset,
)


@dataclass(frozen=True)
class RatePosterior:
    """Gamma(shape, rate) posterior for events per weighted satellite-second."""

    shape: float
    rate_s: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.rate_s <= 0:
            raise ValueError("Gamma shape and rate_s must be positive")

    @property
    def mean_per_satellite_second(self) -> float:
        return self.shape / self.rate_s

    def updated(self, observed_events: int, weighted_exposure_s: float) -> "RatePosterior":
        if observed_events < 0 or weighted_exposure_s < 0:
            raise ValueError("events and exposure must be non-negative")
        return RatePosterior(self.shape + observed_events, self.rate_s + weighted_exposure_s)


@dataclass(frozen=True)
class DemandProfile:
    service_type: str
    rate: RatePosterior
    deadline_window_s: tuple[float, float]
    service_duration_s: tuple[float, float]
    value: tuple[float, float]
    inventory: Mapping[str, tuple[float, float]]
    age_exponent: float = 1.0
    infant_fraction_of_life: float = 0.05
    infant_multiplier: float = 1.0
    priority: int = 0

    def __post_init__(self) -> None:
        for name, bounds in (
            ("deadline_window_s", self.deadline_window_s),
            ("service_duration_s", self.service_duration_s),
            ("value", self.value),
        ):
            if bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError(f"invalid {name} bounds")
        if self.age_exponent < 0 or self.infant_multiplier < 0:
            raise ValueError("age_exponent and infant_multiplier must be non-negative")
        if not 0 <= self.infant_fraction_of_life <= 1:
            raise ValueError("infant_fraction_of_life must lie in [0, 1]")

    def exposure_multiplier(self, asset: SatelliteAsset, time_s: float) -> float:
        age_s = asset.age_at(time_s)
        if asset.design_life_s is None:
            life_fraction = age_s / (age_s + 31_557_600.0)
        else:
            life_fraction = min(2.0, age_s / asset.design_life_s)
        ageing = 1.0 + life_fraction ** self.age_exponent
        infant = self.infant_multiplier if life_fraction <= self.infant_fraction_of_life else 1.0
        return asset.health_multiplier * ageing * infant


class BayesianDemandModel:
    """Generate and update sparse future SLA demand scenarios."""

    model_version = "bayesian-demand-v0.1"

    def __init__(self, profiles: Iterable[DemandProfile], *, bin_s: float = 2_592_000.0):
        self.profiles = {profile.service_type: profile for profile in profiles}
        if not self.profiles:
            raise ValueError("at least one demand profile is required")
        if bin_s <= 0:
            raise ValueError("bin_s must be positive")
        self.bin_s = float(bin_s)

    def update(self, observations: Mapping[str, tuple[int, float]]) -> "BayesianDemandModel":
        """Return a new model after ``service_type -> (events, exposure_s)`` data."""
        profiles = []
        unknown = set(observations) - set(self.profiles)
        if unknown:
            raise KeyError(f"unknown service types: {sorted(unknown)}")
        for name, profile in self.profiles.items():
            events, exposure = observations.get(name, (0, 0.0))
            profiles.append(replace(profile, rate=profile.rate.updated(events, exposure)))
        return BayesianDemandModel(profiles, bin_s=self.bin_s)

    @staticmethod
    def _activated_assets(
        installed_base: Iterable[SatelliteAsset],
        launch: LaunchScenario,
        deployments: Iterable[PlannedDeployment],
    ) -> list[SatelliteAsset]:
        assets = list(installed_base)
        launch_times = {item.opportunity_id: item.launch_time_s for item in launch.opportunities}
        for deployment in deployments:
            launch_s = launch_times.get(deployment.launch_opportunity_id)
            if launch_s is None:
                continue
            for asset in deployment.assets:
                assets.append(replace(
                    asset,
                    activation_s=max(asset.activation_s, launch_s + deployment.activation_delay_s),
                    age_at_as_of_s=0.0,
                ))
        ids = [asset.asset_id for asset in assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset_id values must be unique after applying deployments")
        return assets

    def _generate_one(
        self,
        scenario_id: str,
        probability: float,
        horizon_s: float,
        assets: list[SatelliteAsset],
        confirmed_orders: tuple[DemandRequest, ...],
        pipeline_orders: tuple[PipelineOrder, ...],
        rng: np.random.Generator,
    ) -> DemandScenario:
        requests = [r for r in confirmed_orders if r.release_s <= horizon_s]
        requests.extend(
            order.request for order in pipeline_orders
            if order.request.release_s <= horizon_s and rng.random() < order.conversion_probability
        )
        edges = np.arange(0.0, horizon_s, self.bin_s)
        for profile in self.profiles.values():
            latent_rate = rng.gamma(profile.rate.shape, 1.0 / profile.rate.rate_s)
            event_number = 0
            for start_s in edges:
                stop_s = min(start_s + self.bin_s, horizon_s)
                midpoint_s = 0.5 * (start_s + stop_s)
                active = [asset for asset in assets if asset.activation_s <= midpoint_s]
                if not active:
                    continue
                weights = np.asarray(
                    [profile.exposure_multiplier(asset, midpoint_s) for asset in active],
                    dtype=float,
                )
                total_exposure_s = weights.sum() * (stop_s - start_s)
                count = int(rng.poisson(latent_rate * total_exposure_s))
                if count == 0:
                    continue
                probabilities = weights / weights.sum()
                for _ in range(count):
                    asset = active[int(rng.choice(len(active), p=probabilities))]
                    release_s = float(rng.uniform(start_s, stop_s))
                    window_s = float(rng.uniform(*profile.deadline_window_s))
                    duration_s = float(rng.uniform(*profile.service_duration_s))
                    value = float(rng.uniform(*profile.value))
                    inventory = {
                        name: float(rng.uniform(*bounds))
                        for name, bounds in profile.inventory.items()
                    }
                    event_number += 1
                    requests.append(DemandRequest(
                        request_id=f"{scenario_id}-{profile.service_type}-{event_number}",
                        target_asset_id=asset.asset_id,
                        orbit_class=asset.orbit_class,
                        semimajor_axis_km=asset.semimajor_axis_km,
                        inclination_rad=asset.inclination_rad,
                        service_type=profile.service_type,
                        release_s=release_s,
                        deadline_s=release_s + window_s,
                        service_duration_s=duration_s,
                        value=value,
                        inventory=inventory,
                        priority=profile.priority,
                    ))
        requests.sort(key=lambda request: request.release_s)
        return DemandScenario(
            scenario_id=scenario_id,
            probability=probability,
            requests=tuple(requests),
            active_asset_ids=tuple(asset.asset_id for asset in assets),
        )

    def forecast(
        self,
        *,
        forecast_id: str,
        as_of: datetime,
        horizon_s: float,
        installed_base: Iterable[SatelliteAsset],
        confirmed_orders: Iterable[DemandRequest] = (),
        pipeline_orders: Iterable[PipelineOrder] = (),
        scenario_count: int = 100,
        random_seed: int = 0,
    ) -> DemandForecast:
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        assets = list(installed_base)
        rng = np.random.default_rng(random_seed)
        probability = 1.0 / scenario_count
        scenarios = tuple(self._generate_one(
            f"{forecast_id}-{index:04d}", probability, horizon_s, assets,
            tuple(confirmed_orders), tuple(pipeline_orders), rng,
        ) for index in range(scenario_count))
        return DemandForecast(
            forecast_id, as_of, horizon_s, self.model_version, random_seed, scenarios
        )

    def joint_forecast(
        self,
        launch_forecast: ForecastSnapshot,
        *,
        installed_base: Iterable[SatelliteAsset],
        deployments: Iterable[PlannedDeployment] = (),
        confirmed_orders: Iterable[DemandRequest] = (),
        pipeline_orders: Iterable[PipelineOrder] = (),
        demand_draws_per_launch: int = 1,
        random_seed: int | None = None,
    ) -> tuple[JointPlanningScenario, ...]:
        """Sample demand conditional on every supplied launch realization."""
        if demand_draws_per_launch <= 0:
            raise ValueError("demand_draws_per_launch must be positive")
        seed = launch_forecast.random_seed if random_seed is None else random_seed
        rng = np.random.default_rng(seed)
        installed = tuple(installed_base)
        deployments = tuple(deployments)
        confirmed = tuple(confirmed_orders)
        pipeline = tuple(pipeline_orders)
        joint = []
        for launch in launch_forecast.scenarios:
            assets = self._activated_assets(installed, launch, deployments)
            probability = launch.probability / demand_draws_per_launch
            for draw in range(demand_draws_per_launch):
                scenario_id = f"{launch.scenario_id}-demand-{draw:03d}"
                demand = self._generate_one(
                    scenario_id, probability, launch_forecast.horizon_s,
                    assets, confirmed, pipeline, rng,
                )
                joint.append(JointPlanningScenario(
                    scenario_id=scenario_id,
                    probability=probability,
                    launch=launch,
                    demand=demand,
                ))
        total = sum(item.probability for item in joint)
        if joint and abs(total - 1.0) > 1e-9:
            raise ValueError(f"joint scenario probabilities must sum to one, got {total}")
        return tuple(joint)


def to_scheduler_demands(requests: Iterable[DemandRequest], node_index: Mapping[str, int]):
    """Adapt generated requests to the existing optimization ``Demands`` type."""
    from ..schedule import Demands

    ordered = tuple(requests)
    missing = sorted({r.target_asset_id for r in ordered} - set(node_index))
    if missing:
        raise KeyError(f"targets absent from the cost table: {missing}")
    return Demands(
        node=np.asarray([node_index[r.target_asset_id] for r in ordered], dtype=np.int64),
        release=np.asarray([r.release_s for r in ordered], dtype=float),
        deadline=np.asarray([r.deadline_s for r in ordered], dtype=float),
        service=np.asarray([r.service_duration_s for r in ordered], dtype=float),
        value=np.asarray([r.value for r in ordered], dtype=float),
    )
