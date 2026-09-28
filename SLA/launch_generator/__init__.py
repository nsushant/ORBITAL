"""Data contracts and feature builders for launch-access forecasting."""

from .models import (
    EvidenceAssertion,
    ForecastSnapshot,
    LaunchOpportunity,
    LaunchScenario,
    RealizedLaunch,
)
from .archive import load_snapshot, save_snapshot
from .scoring import brier_score, interval_coverage
from .features import (
    CapacityTrainingRow,
    CountTrainingRow,
    FeatureBundle,
    MarkTrainingRow,
    MissionTrainingRow,
    build_capacity_rows,
    build_count_rows,
    build_feature_bundle,
    build_mark_rows,
    build_mission_rows,
    chronological_split,
    provider_resolver,
    write_jsonl,
)

__all__ = [
    "EvidenceAssertion", "ForecastSnapshot", "LaunchOpportunity",
    "LaunchScenario", "RealizedLaunch", "load_snapshot", "save_snapshot",
    "brier_score", "interval_coverage", "CapacityTrainingRow",
    "CountTrainingRow", "FeatureBundle", "MarkTrainingRow",
    "MissionTrainingRow", "build_capacity_rows", "build_count_rows",
    "build_feature_bundle", "build_mark_rows", "build_mission_rows",
    "chronological_split", "provider_resolver", "write_jsonl",
]
