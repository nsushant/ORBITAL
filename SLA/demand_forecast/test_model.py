from datetime import datetime, timezone

import numpy as np

from SLA.demand_forecast import (
    BayesianDemandModel,
    DemandProfile,
    DemandRequest,
    PipelineOrder,
    PlannedDeployment,
    RatePosterior,
    SatelliteAsset,
)
from SLA.demand_forecast.model import to_scheduler_demands
from SLA.launch_generator.models import ForecastSnapshot, LaunchOpportunity, LaunchScenario


DAY = 86_400.0
YEAR = 365.25 * DAY


def asset(asset_id="sat-1", activation_s=0.0):
    return SatelliteAsset(asset_id, "LEO", 6878.0, 0.9, 5 * YEAR,
                          activation_s=activation_s, design_life_s=10 * YEAR)


def model(rate=RatePosterior(2.0, 20 * YEAR)):
    return BayesianDemandModel([DemandProfile(
        "refuel", rate, (30 * DAY, 60 * DAY), (DAY, 2 * DAY),
        (1e6, 2e6), {"propellant_kg": (10.0, 20.0)}, age_exponent=2.0,
    )], bin_s=30 * DAY)


def test_gamma_poisson_update_is_conjugate():
    updated = model().update({"refuel": (3, 10 * YEAR)})
    posterior = updated.profiles["refuel"].rate
    assert posterior.shape == 5.0
    assert posterior.rate_s == 30 * YEAR


def test_confirmed_and_pipeline_orders_are_handled_separately():
    request = DemandRequest("known", "sat-1", "LEO", 6878, 0.9, "refuel",
                            0, 10 * DAY, DAY, 1e6)
    forecast = model(RatePosterior(1.0, 1e30)).forecast(
        forecast_id="f", as_of=datetime.now(timezone.utc), horizon_s=YEAR,
        installed_base=[asset()], confirmed_orders=[request],
        pipeline_orders=[PipelineOrder(request, 0.0)], scenario_count=3, random_seed=4,
    )
    assert all([r.request_id for r in s.requests].count("known") == 1 for s in forecast.scenarios)
    assert np.isclose(sum(s.probability for s in forecast.scenarios), 1.0)


def test_future_asset_only_activates_when_linked_launch_occurs():
    opportunity = LaunchOpportunity("launch-1", 20 * DAY, "site", "provider", "vehicle",
        "LEO", 6878, 0.9, 100, None, None, 10 * DAY, "A", 1.0)
    launches = ForecastSnapshot(
        "lf", datetime.now(timezone.utc), YEAR, "v", "e", 1,
        (LaunchScenario("occurs", 0.5, (opportunity,)), LaunchScenario("misses", 0.5, ())),
    )
    deployment = PlannedDeployment("launch-1", (asset("future"),), 5 * DAY)
    joint = model(RatePosterior(1.0, 1e30)).joint_forecast(
        launches, installed_base=[asset()], deployments=[deployment], random_seed=2)
    by_id = {s.launch.scenario_id: s for s in joint}
    assert "future" in by_id["occurs"].demand.active_asset_ids
    assert "future" not in by_id["misses"].demand.active_asset_ids
    assert np.isclose(sum(s.probability for s in joint), 1.0)


def test_scheduler_adapter_preserves_seconds_and_values():
    request = DemandRequest("r", "sat-1", "LEO", 6878, 0.9, "refuel",
                            12.0, 34.0, 5.0, 6.0)
    demands = to_scheduler_demands([request], {"sat-1": 7})
    assert demands.node.tolist() == [7]
    assert demands.release.tolist() == [12.0]
    assert demands.deadline.tolist() == [34.0]
    assert demands.service.tolist() == [5.0]
    assert demands.value.tolist() == [6.0]
