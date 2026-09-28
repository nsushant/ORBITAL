"""JSON archive for immutable forecast snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import ForecastSnapshot, LaunchOpportunity, LaunchScenario


def save_snapshot(path: str | Path, snapshot: ForecastSnapshot) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(snapshot)
    data["as_of"] = snapshot.as_of.isoformat().replace("+00:00", "Z")
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_snapshot(path: str | Path) -> ForecastSnapshot:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    scenarios = tuple(
        LaunchScenario(
            scenario_id=s["scenario_id"],
            probability=float(s["probability"]),
            opportunities=tuple(
                LaunchOpportunity(
                    **{
                        **x,
                        "evidence_ids": tuple(x.get("evidence_ids", ())),
                    }
                )
                for x in s["opportunities"]
            ),
        )
        for s in data["scenarios"]
    )
    return ForecastSnapshot(
        forecast_id=data["forecast_id"],
        as_of=datetime.fromisoformat(data["as_of"].replace("Z", "+00:00")),
        horizon_s=float(data["horizon_s"]),
        model_version=data["model_version"],
        evidence_snapshot_id=data["evidence_snapshot_id"],
        random_seed=int(data["random_seed"]),
        scenarios=scenarios,
        metadata=data.get("metadata", {}),
    )
