"""Leakage-safe training rows for launch-calendar forecasting.

Every feature is computed from records available on or before a row's `as_of`.
Labels may use later realized GCAT events. Cross-source entity resolution is
explicit: callers provide provider aliases and LL2-to-GCAT launch crosswalks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .ingestion.gcat import GcatImportResult, GcatLaunch
from .ingestion.ll2 import LL2LaunchState, LL2Snapshot, diff_snapshots


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _default_provider(value: str | None) -> str | None:
    return value.strip().casefold() if value and value.strip() else None


def provider_resolver(aliases: Mapping[str, str] | None = None) -> Callable[[str | None], str | None]:
    lookup = {key.strip().casefold(): value for key, value in (aliases or {}).items()}

    def resolve(value: str | None) -> str | None:
        normalized = _default_provider(value)
        return None if normalized is None else lookup.get(normalized, normalized)

    return resolve


@dataclass(frozen=True)
class CountTrainingRow:
    as_of: datetime
    provider: str
    horizon_s: float
    month: int
    month_sin: float
    month_cos: float
    launches_previous_30d: int
    launches_previous_90d: int
    launches_previous_365d: int
    announced_in_horizon: int
    announced_with_day_or_better_precision: int
    mean_announced_lead_s: float | None
    mean_prior_revision_count: float
    target_realized_launch_count: int


@dataclass(frozen=True)
class MissionTrainingRow:
    as_of: datetime
    ll2_launch_id: str
    provider: str | None
    vehicle: str | None
    site: str | None
    orbit_abbrev: str | None
    status: str | None
    time_precision: str | None
    announced_net_s: float | None
    revision_count: int
    provider_launches_previous_90d: int
    label_observed: bool
    launched: bool | None
    actual_launch_s: float | None
    schedule_error_s: float | None
    gcat_launch_id: str | None


@dataclass(frozen=True)
class MarkTrainingRow:
    as_of: datetime
    gcat_launch_id: str
    provider: str | None
    month: int
    provider_launches_previous_90d: int
    target_vehicle: str | None
    target_site: str | None
    target_orbit_class: str | None


@dataclass(frozen=True)
class CapacityTrainingRow:
    as_of: datetime
    gcat_launch_id: str
    provider: str | None
    vehicle: str | None
    site: str | None
    orbit_class: str | None
    constellation_payload_fraction: float
    target_payload_count: int
    target_manifested_mass_kg: float | None
    mass_observation_fraction: float


@dataclass(frozen=True)
class FeatureBundle:
    counts: tuple[CountTrainingRow, ...]
    missions: tuple[MissionTrainingRow, ...]
    marks: tuple[MarkTrainingRow, ...]
    capacities: tuple[CapacityTrainingRow, ...]


def _launch_time(launch: GcatLaunch) -> datetime | None:
    return launch.launch_time_utc


def _prior_count(events: Sequence[GcatLaunch], as_of: datetime, provider: str,
                 window_s: float, resolve: Callable[[str | None], str | None]) -> int:
    start = as_of - timedelta(seconds=window_s)
    return sum(
        1 for event in events
        if event.launch_time_utc is not None
        and start <= event.launch_time_utc < as_of
        and resolve(event.agency) == provider
    )


def _absolute(snapshot: LL2Snapshot, seconds: float | None) -> datetime | None:
    return snapshot.absolute_time(seconds)


def _revision_counts(snapshots: Sequence[LL2Snapshot]) -> list[dict[str, int]]:
    """Cumulative revisions visible at each snapshot, never using later states."""
    counts: dict[str, int] = {}
    result: list[dict[str, int]] = []
    previous: LL2Snapshot | None = None
    for snapshot in snapshots:
        if previous is not None:
            for revision in diff_snapshots(previous, snapshot):
                counts[revision.launch_id] = counts.get(revision.launch_id, 0) + 1
        result.append(dict(counts))
        previous = snapshot
    return result


def build_count_rows(gcat: GcatImportResult, snapshots: Sequence[LL2Snapshot],
                     horizons_s: Sequence[float], *,
                     provider_aliases: Mapping[str, str] | None = None,
                     label_cutoff: datetime | None = None) -> tuple[CountTrainingRow, ...]:
    resolve = provider_resolver(provider_aliases)
    ordered = sorted(snapshots, key=lambda x: x.as_of)
    revisions = _revision_counts(ordered)
    events = tuple(x for x in gcat.launches if x.launch_time_utc is not None)
    cutoff = _utc(label_cutoff) if label_cutoff else max(
        (x.launch_time_utc for x in events),
        default=datetime.min.replace(tzinfo=timezone.utc),
    )
    rows: list[CountTrainingRow] = []
    for index, snapshot in enumerate(ordered):
        providers = {
            provider for launch in snapshot.launches
            if (provider := resolve(launch.provider)) is not None
        }
        providers.update(
            provider for event in events if event.launch_time_utc < snapshot.as_of
            if (provider := resolve(event.agency)) is not None
        )
        for horizon_s in horizons_s:
            if horizon_s <= 0:
                raise ValueError("horizons_s must be positive")
            end = snapshot.as_of + timedelta(seconds=float(horizon_s))
            # A partially observed label horizon would look like a false zero.
            if end > cutoff:
                continue
            for provider in sorted(providers):
                announced = [
                    launch for launch in snapshot.launches
                    if resolve(launch.provider) == provider
                    and (time := _absolute(snapshot, launch.net_s)) is not None
                    and snapshot.as_of < time <= end
                ]
                leads = [launch.net_s for launch in announced if launch.net_s is not None]
                precision_count = sum(
                    1 for launch in announced
                    if (launch.time_precision or "").upper() in {"SEC", "MIN", "HR", "DAY"}
                )
                revision_values = [revisions[index].get(x.launch_id, 0) for x in announced]
                target = sum(
                    1 for event in events
                    if snapshot.as_of < event.launch_time_utc <= end
                    and resolve(event.agency) == provider
                )
                angle = 2.0 * math.pi * (snapshot.as_of.month - 1) / 12.0
                rows.append(CountTrainingRow(
                    as_of=snapshot.as_of,
                    provider=provider,
                    horizon_s=float(horizon_s),
                    month=snapshot.as_of.month,
                    month_sin=math.sin(angle),
                    month_cos=math.cos(angle),
                    launches_previous_30d=_prior_count(events, snapshot.as_of, provider, 30 * 86400, resolve),
                    launches_previous_90d=_prior_count(events, snapshot.as_of, provider, 90 * 86400, resolve),
                    launches_previous_365d=_prior_count(events, snapshot.as_of, provider, 365 * 86400, resolve),
                    announced_in_horizon=len(announced),
                    announced_with_day_or_better_precision=precision_count,
                    mean_announced_lead_s=sum(leads) / len(leads) if leads else None,
                    mean_prior_revision_count=(sum(revision_values) / len(revision_values)) if revision_values else 0.0,
                    target_realized_launch_count=target,
                ))
    return tuple(rows)


def build_mission_rows(gcat: GcatImportResult, snapshots: Sequence[LL2Snapshot], *,
                       launch_crosswalk: Mapping[str, str],
                       provider_aliases: Mapping[str, str] | None = None,
                       label_cutoff: datetime | None = None) -> tuple[MissionTrainingRow, ...]:
    resolve = provider_resolver(provider_aliases)
    ordered = sorted(snapshots, key=lambda x: x.as_of)
    revisions = _revision_counts(ordered)
    gcat_by_id = {x.launch_id: x for x in gcat.launches}
    cutoff = _utc(label_cutoff) if label_cutoff else max(
        (x.launch_time_utc for x in gcat.launches if x.launch_time_utc is not None),
        default=datetime.min.replace(tzinfo=timezone.utc),
    )
    events = tuple(x for x in gcat.launches if x.launch_time_utc is not None)
    rows: list[MissionTrainingRow] = []
    for index, snapshot in enumerate(ordered):
        for launch in snapshot.launches:
            provider = resolve(launch.provider)
            gcat_id = launch_crosswalk.get(launch.launch_id)
            realized = gcat_by_id.get(gcat_id) if gcat_id else None
            actual = realized.launch_time_utc if realized else None
            # Once the outcome is known this is no longer a forecast-time row.
            if actual is not None and actual <= snapshot.as_of:
                continue
            announced = _absolute(snapshot, launch.net_s)
            label_observed = actual is not None and actual <= cutoff
            actual_s = (actual - snapshot.as_of).total_seconds() if label_observed else None
            rows.append(MissionTrainingRow(
                as_of=snapshot.as_of,
                ll2_launch_id=launch.launch_id,
                provider=provider,
                vehicle=launch.vehicle,
                site=launch.site,
                orbit_abbrev=launch.orbit_abbrev,
                status=launch.status,
                time_precision=launch.time_precision,
                announced_net_s=launch.net_s,
                revision_count=revisions[index].get(launch.launch_id, 0),
                provider_launches_previous_90d=(
                    _prior_count(events, snapshot.as_of, provider, 90 * 86400, resolve)
                    if provider is not None else 0
                ),
                label_observed=label_observed,
                launched=True if label_observed else None,
                actual_launch_s=actual_s,
                schedule_error_s=(actual - announced).total_seconds()
                    if label_observed and announced is not None else None,
                gcat_launch_id=gcat_id,
            ))
    return tuple(rows)


def _primary_orbit(launch: GcatLaunch) -> str | None:
    classes = [x.orbit_class for x in launch.payloads if x.orbit_class]
    return sorted(set(classes), key=lambda value: (-classes.count(value), value))[0] if classes else None


def build_mark_rows(gcat: GcatImportResult, *,
                    provider_aliases: Mapping[str, str] | None = None) -> tuple[MarkTrainingRow, ...]:
    resolve = provider_resolver(provider_aliases)
    events = sorted(
        (x for x in gcat.launches if x.launch_time_utc is not None),
        key=lambda x: x.launch_time_utc,
    )
    rows = []
    for event in events:
        provider = resolve(event.agency)
        rows.append(MarkTrainingRow(
            as_of=event.launch_time_utc,
            gcat_launch_id=event.launch_id,
            provider=provider,
            month=event.launch_time_utc.month,
            provider_launches_previous_90d=(
                _prior_count(events, event.launch_time_utc, provider, 90 * 86400, resolve)
                if provider is not None else 0
            ),
            target_vehicle=event.vehicle,
            target_site=event.site,
            target_orbit_class=_primary_orbit(event),
        ))
    return tuple(rows)


def build_capacity_rows(gcat: GcatImportResult, *,
                        provider_aliases: Mapping[str, str] | None = None) -> tuple[CapacityTrainingRow, ...]:
    resolve = provider_resolver(provider_aliases)
    rows = []
    for event in gcat.launches:
        if event.launch_time_utc is None or not event.payloads:
            continue
        observed = [x for x in event.payloads if x.best_mass_kg is not None]
        programs = [x.program.casefold() for x in event.payloads if x.program]
        constellation = sum(1 for x in programs if "constell" in x)
        rows.append(CapacityTrainingRow(
            as_of=event.launch_time_utc,
            gcat_launch_id=event.launch_id,
            provider=resolve(event.agency),
            vehicle=event.vehicle,
            site=event.site,
            orbit_class=_primary_orbit(event),
            constellation_payload_fraction=constellation / len(event.payloads),
            target_payload_count=len(event.payloads),
            target_manifested_mass_kg=(sum(x.best_mass_kg for x in observed) if observed else None),
            mass_observation_fraction=len(observed) / len(event.payloads),
        ))
    return tuple(rows)


def build_feature_bundle(gcat: GcatImportResult, snapshots: Sequence[LL2Snapshot],
                         horizons_s: Sequence[float], *,
                         provider_aliases: Mapping[str, str] | None = None,
                         launch_crosswalk: Mapping[str, str] | None = None,
                         label_cutoff: datetime | None = None) -> FeatureBundle:
    return FeatureBundle(
        counts=build_count_rows(
            gcat, snapshots, horizons_s, provider_aliases=provider_aliases,
            label_cutoff=label_cutoff,
        ),
        missions=build_mission_rows(
            gcat, snapshots, launch_crosswalk=launch_crosswalk or {},
            provider_aliases=provider_aliases, label_cutoff=label_cutoff,
        ),
        marks=build_mark_rows(gcat, provider_aliases=provider_aliases),
        capacities=build_capacity_rows(gcat, provider_aliases=provider_aliases),
    )


T = TypeVar("T")


def chronological_split(rows: Sequence[T], train_before: datetime,
                        calibration_before: datetime) -> tuple[tuple[T, ...], tuple[T, ...], tuple[T, ...]]:
    train_before = _utc(train_before)
    calibration_before = _utc(calibration_before)
    if calibration_before <= train_before:
        raise ValueError("calibration_before must follow train_before")
    train, calibration, test = [], [], []
    for row in rows:
        as_of = _utc(getattr(row, "as_of"))
        if as_of < train_before:
            train.append(row)
        elif as_of < calibration_before:
            calibration.append(row)
        else:
            test.append(row)
    return tuple(train), tuple(calibration), tuple(test)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            data = asdict(row)
            if isinstance(data.get("as_of"), datetime):
                data["as_of"] = data["as_of"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            stream.write(json.dumps(data, sort_keys=True) + "\n")
    return target
