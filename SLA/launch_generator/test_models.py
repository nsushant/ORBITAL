from datetime import datetime, timezone

from SLA.launch_generator import (
    ForecastSnapshot,
    LaunchOpportunity,
    LaunchScenario,
    brier_score,
    interval_coverage,
    load_snapshot,
    save_snapshot,
)


def _snapshot():
    opportunity = LaunchOpportunity(
        opportunity_id="L-1", launch_time_s=864000.0, site="VAFB",
        provider="P", vehicle="V", orbit_class="SSO",
        semimajor_axis_km=7078.137, inclination_rad=1.7104,
        projected_purchasable_capacity_kg=100.0, price=1_000_000.0,
        currency="EUR", booking_deadline_s=432000.0, evidence_class="B",
        occurrence_probability=0.8,
    )
    return ForecastSnapshot(
        forecast_id="F-1", as_of=datetime(2030, 1, 1, tzinfo=timezone.utc),
        horizon_s=31_536_000.0, model_version="baseline-v1",
        evidence_snapshot_id="E-1", random_seed=7,
        scenarios=(LaunchScenario("S-1", 1.0, (opportunity,)),),
    )


def test_snapshot_round_trip(tmp_path):
    original = _snapshot()
    loaded = load_snapshot(save_snapshot(tmp_path / "forecast.json", original))
    assert loaded == original
    assert loaded.seconds_at(loaded.at_seconds(123.5)) == 123.5


def test_scores():
    assert brier_score(0.8, True) == (0.8 - 1.0) ** 2
    assert interval_coverage([(0, 2), (0, 1)], [1, 2]) == 0.5
