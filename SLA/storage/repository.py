"""PostgreSQL repository for content-addressed source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .models import SourceSnapshot


@dataclass(frozen=True)
class StoredSnapshot:
    snapshot_id: UUID
    object_id: UUID
    content_hash: str
    object_created: bool


class SnapshotRepository:
    """Persist retrievals while storing identical response bytes only once."""

    def __init__(self, connection: Any):
        self.connection = connection

    def add(self, snapshot: SourceSnapshot) -> StoredSnapshot:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO source_object (content_hash, media_type, content, size_bytes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    RETURNING object_id
                    """,
                    (
                        snapshot.content_hash,
                        snapshot.media_type,
                        snapshot.content,
                        snapshot.size_bytes,
                    ),
                )
                row = cursor.fetchone()
                object_created = row is not None
                if row is None:
                    cursor.execute(
                        "SELECT object_id FROM source_object WHERE content_hash = %s",
                        (snapshot.content_hash,),
                    )
                    row = cursor.fetchone()
                object_id = row[0]

                cursor.execute(
                    """
                    INSERT INTO source_snapshot (
                        source, retrieved_at, request_uri, object_id,
                        source_version, licence_status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING snapshot_id
                    """,
                    (
                        snapshot.source,
                        snapshot.retrieved_at,
                        snapshot.request_uri,
                        object_id,
                        snapshot.source_version,
                        snapshot.licence_status,
                    ),
                )
                snapshot_id = cursor.fetchone()[0]

        return StoredSnapshot(
            snapshot_id=snapshot_id,
            object_id=object_id,
            content_hash=snapshot.content_hash,
            object_created=object_created,
        )

    def storage_usage_bytes(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM source_object")
            return int(cursor.fetchone()[0])
