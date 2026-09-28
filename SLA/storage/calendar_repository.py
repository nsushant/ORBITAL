"""Row mapping and PostgreSQL writes for the canonical launch calendar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import UUID

from ..launch_generator.ingestion.gcat import GcatImportResult
from ..launch_generator.ingestion.ll2 import LL2LaunchState, LL2Revision, LL2Snapshot


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return value.astimezone(timezone.utc)


def gcat_launch_rows(result: GcatImportResult, as_of: datetime, snapshot_id: UUID) -> tuple[list[tuple], list[tuple]]:
    """Map GCAT objects to database rows using seconds relative to `as_of`."""
    as_of = _utc(as_of)
    launches: list[tuple] = []
    payloads: list[tuple] = []
    for launch in result.launches:
        actual_launch_s = None
        if launch.launch_time_utc is not None:
            actual_launch_s = (launch.launch_time_utc - as_of).total_seconds()
        launches.append((
            "gcat", launch.launch_id, snapshot_id, actual_launch_s,
            launch.launch_time_precision, launch.launch_time_uncertain,
            launch.raw_launch_date, launch.vehicle, launch.flight_id,
            launch.platform, launch.site, launch.pad, launch.ascent_site,
            launch.ascent_pad, launch.agency, launch.provider_state,
            launch.launch_code, launch.outcome, launch.citation,
        ))
        for payload in launch.payloads:
            payloads.append((
                "gcat", launch.launch_id, payload.jcat, payload.piece,
                payload.name, payload.payload_name, payload.owner, payload.state,
                payload.mass_kg, payload.dry_mass_kg, payload.total_mass_kg,
                payload.perigee_km, payload.apogee_km, payload.inclination_deg,
                payload.orbit_code, payload.orbit_class, payload.program,
                payload.plane, payload.mission_class, payload.category,
                payload.payload_result,
            ))
    return launches, payloads


def ll2_state_rows(snapshot: LL2Snapshot, snapshot_id: UUID) -> list[tuple]:
    return [(
        snapshot_id, launch.launch_id, launch.name, launch.status,
        launch.net_s, launch.window_start_s, launch.window_end_s,
        launch.time_precision, launch.last_updated_s, launch.provider,
        launch.vehicle, launch.site, launch.pad, launch.mission_type,
        launch.orbit_name, launch.orbit_abbrev,
    ) for launch in snapshot.launches]


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return json.dumps(value)


def revision_rows(revisions: Iterable[LL2Revision], old_snapshot_id: UUID | None,
                  new_snapshot_id: UUID) -> list[tuple]:
    return [(
        old_snapshot_id, new_snapshot_id, item.launch_id, item.field,
        _json_value(item.old_value), _json_value(item.new_value),
    ) for item in revisions]


class LaunchCalendarRepository:
    def __init__(self, connection: Any):
        self.connection = connection

    def upsert_gcat(self, result: GcatImportResult, as_of: datetime, snapshot_id: UUID) -> tuple[int, int]:
        launch_rows, payload_rows = gcat_launch_rows(result, as_of, snapshot_id)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO launch_event (
                        source, external_launch_id, source_snapshot_id,
                        actual_launch_s, time_precision, time_uncertain,
                        raw_launch_date, vehicle, flight_id, platform, site, pad,
                        ascent_site, ascent_pad, agency, provider_state,
                        launch_code, outcome, citation
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, external_launch_id) DO UPDATE SET
                        source_snapshot_id=EXCLUDED.source_snapshot_id,
                        actual_launch_s=EXCLUDED.actual_launch_s,
                        time_precision=EXCLUDED.time_precision,
                        time_uncertain=EXCLUDED.time_uncertain,
                        raw_launch_date=EXCLUDED.raw_launch_date,
                        vehicle=EXCLUDED.vehicle, flight_id=EXCLUDED.flight_id,
                        platform=EXCLUDED.platform, site=EXCLUDED.site,
                        pad=EXCLUDED.pad, ascent_site=EXCLUDED.ascent_site,
                        ascent_pad=EXCLUDED.ascent_pad, agency=EXCLUDED.agency,
                        provider_state=EXCLUDED.provider_state,
                        launch_code=EXCLUDED.launch_code, outcome=EXCLUDED.outcome,
                        citation=EXCLUDED.citation, updated_at=now()
                    """,
                    launch_rows,
                )
                cursor.executemany(
                    """
                    INSERT INTO launch_payload (
                        source, external_launch_id, payload_id, piece, name,
                        payload_name, owner, owner_state, mass_kg, dry_mass_kg,
                        total_mass_kg, perigee_km, apogee_km, inclination_deg,
                        orbit_code, orbit_class, program, plane, mission_class,
                        category, payload_result
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, external_launch_id, payload_id) DO UPDATE SET
                        piece=EXCLUDED.piece, name=EXCLUDED.name,
                        payload_name=EXCLUDED.payload_name, owner=EXCLUDED.owner,
                        owner_state=EXCLUDED.owner_state, mass_kg=EXCLUDED.mass_kg,
                        dry_mass_kg=EXCLUDED.dry_mass_kg,
                        total_mass_kg=EXCLUDED.total_mass_kg,
                        perigee_km=EXCLUDED.perigee_km,
                        apogee_km=EXCLUDED.apogee_km,
                        inclination_deg=EXCLUDED.inclination_deg,
                        orbit_code=EXCLUDED.orbit_code,
                        orbit_class=EXCLUDED.orbit_class,
                        program=EXCLUDED.program, plane=EXCLUDED.plane,
                        mission_class=EXCLUDED.mission_class,
                        category=EXCLUDED.category,
                        payload_result=EXCLUDED.payload_result
                    """,
                    payload_rows,
                )
        return len(launch_rows), len(payload_rows)

    def add_ll2_snapshot(self, snapshot: LL2Snapshot, snapshot_id: UUID) -> int:
        rows = ll2_state_rows(snapshot, snapshot_id)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO announced_launch_state (
                        source_snapshot_id, external_launch_id, name, status,
                        net_s, window_start_s, window_end_s, time_precision,
                        source_last_updated_s, provider, vehicle, site, pad,
                        mission_type, orbit_name, orbit_abbrev
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_snapshot_id, external_launch_id) DO NOTHING
                    """,
                    rows,
                )
        return len(rows)

    def latest_ll2_snapshot(self) -> tuple[UUID, LL2Snapshot] | None:
        """Load the latest completed LL2 collection from canonical rows."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT s.snapshot_id, s.retrieved_at, s.request_uri
                FROM source_snapshot s
                WHERE s.source = 'll2-upcoming-manifest'
                  AND EXISTS (
                      SELECT 1 FROM announced_launch_state a
                      WHERE a.source_snapshot_id = s.snapshot_id
                  )
                ORDER BY s.retrieved_at DESC, s.created_at DESC
                LIMIT 1
                """
            )
            header = cursor.fetchone()
            if header is None:
                return None
            snapshot_id, as_of, request_uri = header
            cursor.execute(
                """
                SELECT external_launch_id, name, status, net_s,
                       window_start_s, window_end_s, time_precision,
                       source_last_updated_s, provider, vehicle, site, pad,
                       mission_type, orbit_name, orbit_abbrev
                FROM announced_launch_state
                WHERE source_snapshot_id = %s
                ORDER BY external_launch_id
                """,
                (snapshot_id,),
            )
            launches = tuple(LL2LaunchState(*row) for row in cursor.fetchall())
        return snapshot_id, LL2Snapshot(as_of, request_uri, launches, len(launches))

    def add_revisions(self, revisions: Iterable[LL2Revision],
                      old_snapshot_id: UUID | None, new_snapshot_id: UUID) -> int:
        rows = revision_rows(revisions, old_snapshot_id, new_snapshot_id)
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO launch_revision (
                        old_snapshot_id, new_snapshot_id, external_launch_id,
                        field_name, old_value, new_value
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (old_snapshot_id, new_snapshot_id,
                                 external_launch_id, field_name) DO NOTHING
                    """,
                    rows,
                )
        return len(rows)
