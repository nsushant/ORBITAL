"""Launch Library 2 collection, normalization, and revision detection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

LL2_BASE_URL = "https://ll.thespacedevs.com/2.3.0"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _seconds(as_of: datetime, value: str | None) -> float | None:
    timestamp = _parse_time(value)
    return None if timestamp is None else (timestamp - as_of).total_seconds()


def _nested(data: Mapping[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


@dataclass(frozen=True)
class LL2LaunchState:
    launch_id: str
    name: str
    status: str | None
    net_s: float | None
    window_start_s: float | None
    window_end_s: float | None
    time_precision: str | None
    last_updated_s: float | None
    provider: str | None
    vehicle: str | None
    site: str | None
    pad: str | None
    mission_type: str | None
    orbit_name: str | None
    orbit_abbrev: str | None


@dataclass(frozen=True)
class LL2Snapshot:
    as_of: datetime
    request_uri: str
    launches: tuple[LL2LaunchState, ...]
    source_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _utc(self.as_of))

    def absolute_time(self, seconds: float | None) -> datetime | None:
        return None if seconds is None else self.as_of + timedelta(seconds=seconds)


@dataclass(frozen=True)
class LL2Revision:
    launch_id: str
    field: str
    old_value: Any
    new_value: Any


def _normalize_launch(item: Mapping[str, Any], as_of: datetime) -> LL2LaunchState:
    launch_id = str(item.get("id") or "").strip()
    if not launch_id:
        raise ValueError("LL2 launch is missing id")
    pad = item.get("pad") or {}
    location = pad.get("location") if isinstance(pad, Mapping) else {}
    mission = item.get("mission") or {}
    orbit = mission.get("orbit") if isinstance(mission, Mapping) else {}
    configuration = _nested(item, "rocket", "configuration") or {}
    return LL2LaunchState(
        launch_id=launch_id,
        name=str(item.get("name") or "").strip(),
        status=_nested(item, "status", "abbrev") or _nested(item, "status", "name"),
        net_s=_seconds(as_of, item.get("net")),
        window_start_s=_seconds(as_of, item.get("window_start")),
        window_end_s=_seconds(as_of, item.get("window_end")),
        time_precision=_nested(item, "net_precision", "abbrev"),
        last_updated_s=_seconds(as_of, item.get("last_updated")),
        provider=_nested(item, "launch_service_provider", "name"),
        vehicle=(configuration.get("full_name") or configuration.get("name"))
            if isinstance(configuration, Mapping) else None,
        site=location.get("name") if isinstance(location, Mapping) else None,
        pad=pad.get("name") if isinstance(pad, Mapping) else None,
        mission_type=mission.get("type") if isinstance(mission, Mapping) else None,
        orbit_name=orbit.get("name") if isinstance(orbit, Mapping) else None,
        orbit_abbrev=orbit.get("abbrev") if isinstance(orbit, Mapping) else None,
    )


def normalize_response(payload: bytes | str | Mapping[str, Any], as_of: datetime,
                       request_uri: str) -> LL2Snapshot:
    as_of = _utc(as_of)
    if isinstance(payload, bytes):
        data = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("LL2 response does not contain a results list")
    launches = tuple(sorted(
        (_normalize_launch(item, as_of) for item in results),
        key=lambda item: (item.net_s is None, item.net_s or 0.0, item.launch_id),
    ))
    return LL2Snapshot(as_of, request_uri, launches, int(data.get("count", len(results))))


_TIME_FIELDS = {"net_s", "window_start_s", "window_end_s", "last_updated_s"}


def diff_snapshots(old: LL2Snapshot, new: LL2Snapshot) -> tuple[LL2Revision, ...]:
    """Return field revisions, comparing time values in absolute UTC."""
    old_by_id = {x.launch_id: x for x in old.launches}
    new_by_id = {x.launch_id: x for x in new.launches}
    revisions: list[LL2Revision] = []
    for launch_id in sorted(old_by_id.keys() | new_by_id.keys()):
        before = old_by_id.get(launch_id)
        after = new_by_id.get(launch_id)
        if before is None:
            revisions.append(LL2Revision(launch_id, "record", None, "added"))
            continue
        if after is None:
            revisions.append(LL2Revision(launch_id, "record", "present", "removed"))
            continue
        for field in LL2LaunchState.__dataclass_fields__:
            if field in {"launch_id", "last_updated_s"}:
                continue
            old_value = getattr(before, field)
            new_value = getattr(after, field)
            if field in _TIME_FIELDS:
                old_value = old.absolute_time(old_value)
                new_value = new.absolute_time(new_value)
            if old_value != new_value:
                revisions.append(LL2Revision(launch_id, field, old_value, new_value))
    return tuple(revisions)


class LL2Client:
    """Small standard-library client with bounded pagination and retry."""

    def __init__(self, base_url: str = LL2_BASE_URL, user_agent: str = "ORBITAL-SLA/0.1",
                 timeout_s: float = 30.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.retries = retries

    def upcoming_url(self, *, limit: int = 100, mode: str = "normal") -> str:
        if not 1 <= limit <= 100:
            raise ValueError("LL2 page limit must lie in [1, 100]")
        return f"{self.base_url}/launches/upcoming/?{urlencode({'limit': limit, 'mode': mode})}"

    def _get(self, url: str) -> bytes:
        expected_host = urlparse(self.base_url).netloc
        if urlparse(url).netloc != expected_host:
            raise ValueError("refusing LL2 pagination URL on another host")
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    return response.read()
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == self.retries:
                    raise
                delay = float(exc.headers.get("Retry-After", "1"))
                time.sleep(min(delay, 30.0))
            except (TimeoutError, URLError):
                if attempt == self.retries:
                    raise
                time.sleep(min(2.0 ** attempt, 30.0))
        raise RuntimeError("unreachable")

    def fetch_upcoming(self, *, as_of: datetime | None = None, limit: int = 100,
                       max_pages: int = 20,
                       page_sink: Callable[[str, bytes, datetime], None] | None = None) -> LL2Snapshot:
        as_of = _utc(as_of or datetime.now(timezone.utc))
        first_url = self.upcoming_url(limit=limit)
        url: str | None = first_url
        all_results: list[Mapping[str, Any]] = []
        source_count = 0
        page = 0
        while url is not None:
            if page >= max_pages:
                raise RuntimeError(f"LL2 response exceeded max_pages={max_pages}")
            raw = self._get(url)
            if page_sink is not None:
                page_sink(url, raw, as_of)
            data = json.loads(raw.decode("utf-8"))
            if page == 0:
                source_count = int(data.get("count", 0))
            results = data.get("results")
            if not isinstance(results, list):
                raise ValueError("LL2 response does not contain a results list")
            all_results.extend(results)
            url = data.get("next")
            page += 1
        combined = {"count": source_count, "results": all_results}
        return normalize_response(combined, as_of, first_url)
