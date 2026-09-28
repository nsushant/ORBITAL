"""Repeatable Launch Library 2 snapshot collection command."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from .ingestion.ll2 import LL2Client, diff_snapshots
from ..storage import LaunchCalendarRepository, SnapshotRepository, SourceSnapshot


@dataclass(frozen=True)
class LL2CollectionReport:
    as_of: datetime
    manifest_snapshot_id: UUID
    previous_snapshot_id: UUID | None
    page_count: int
    announced_launch_count: int
    revision_count: int
    revisions_by_field: dict[str, int]
    raw_storage_bytes: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _manifest_bytes(as_of: datetime, entries: Sequence[dict[str, Any]],
                    previous_snapshot_id: UUID | None) -> bytes:
    return json.dumps({
        "as_of": as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "previous_snapshot_id": str(previous_snapshot_id) if previous_snapshot_id else None,
        "entries": list(entries),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def collect_ll2(connection: Any, *, client: LL2Client | None = None,
                as_of: datetime | None = None, limit: int = 100,
                max_pages: int = 20) -> LL2CollectionReport:
    """Collect, archive, normalize, compare, and persist one LL2 calendar."""
    as_of = as_of or _utc_now()
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(timezone.utc)
    snapshots = SnapshotRepository(connection)
    calendar = LaunchCalendarRepository(connection)
    previous = calendar.latest_ll2_snapshot()
    previous_id = previous[0] if previous else None
    page_entries: list[dict[str, Any]] = []

    def archive_page(url: str, content: bytes, retrieved_at: datetime) -> None:
        stored = snapshots.add(SourceSnapshot(
            source="ll2-upcoming-page",
            retrieved_at=retrieved_at,
            request_uri=url,
            content=content,
            media_type="application/json",
            source_version="2.3.0",
            licence_status="unresolved",
        ))
        page_entries.append({
            "request_uri": url,
            "snapshot_id": str(stored.snapshot_id),
            "content_hash": stored.content_hash,
            "size_bytes": len(content),
        })

    client = client or LL2Client()
    current = client.fetch_upcoming(
        as_of=as_of,
        limit=limit,
        max_pages=max_pages,
        page_sink=archive_page,
    )
    revisions = diff_snapshots(previous[1], current) if previous else ()
    with connection.transaction():
        manifest = snapshots.add(SourceSnapshot(
            source="ll2-upcoming-manifest",
            retrieved_at=as_of,
            request_uri=current.request_uri,
            content=_manifest_bytes(as_of, page_entries, previous_id),
            media_type="application/vnd.orbital.collection-manifest+json",
            source_version="2.3.0",
            licence_status="unresolved",
        ))
        announced_count = calendar.add_ll2_snapshot(current, manifest.snapshot_id)
        calendar.add_revisions(revisions, previous_id, manifest.snapshot_id)
    return LL2CollectionReport(
        as_of=as_of,
        manifest_snapshot_id=manifest.snapshot_id,
        previous_snapshot_id=previous_id,
        page_count=len(page_entries),
        announced_launch_count=announced_count,
        revision_count=len(revisions),
        revisions_by_field=dict(sorted(Counter(x.field for x in revisions).items())),
        raw_storage_bytes=snapshots.storage_usage_bytes(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive LL2 upcoming-launch pages and persist schedule revisions"
    )
    parser.add_argument("--database-url", default=os.environ.get("SLA_DATABASE_URL"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("set SLA_DATABASE_URL or pass --database-url")
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before collecting LL2") from exc
    with psycopg.connect(args.database_url) as connection:
        report = collect_ll2(
            connection,
            client=LL2Client(timeout_s=args.timeout_s, retries=args.retries),
            limit=args.limit,
            max_pages=args.max_pages,
        )
    output = asdict(report)
    output["as_of"] = report.as_of.isoformat().replace("+00:00", "Z")
    output["manifest_snapshot_id"] = str(report.manifest_snapshot_id)
    output["previous_snapshot_id"] = (
        str(report.previous_snapshot_id) if report.previous_snapshot_id else None
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
