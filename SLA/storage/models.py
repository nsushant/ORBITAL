"""Storage-layer records independent of the PostgreSQL driver."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256


@dataclass(frozen=True)
class SourceSnapshot:
    source: str
    retrieved_at: datetime
    request_uri: str
    content: bytes
    media_type: str = "application/octet-stream"
    source_version: str | None = None
    licence_status: str = "unresolved"

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        object.__setattr__(self, "retrieved_at", self.retrieved_at.astimezone(timezone.utc))
        if not self.source:
            raise ValueError("source must not be empty")
        if not self.request_uri:
            raise ValueError("request_uri must not be empty")
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")

    @property
    def content_hash(self) -> str:
        return sha256(self.content).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.content)
