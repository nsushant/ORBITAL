from datetime import datetime, timedelta, timezone

import pytest

from SLA.storage.models import SourceSnapshot


def test_snapshot_hash_and_utc_normalisation():
    snapshot = SourceSnapshot(
        source="ll2",
        retrieved_at=datetime(2030, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))),
        request_uri="https://example.test/launches",
        content=b"same response",
    )
    assert snapshot.retrieved_at == datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert snapshot.size_bytes == 13
    assert len(snapshot.content_hash) == 64


def test_snapshot_rejects_naive_time():
    with pytest.raises(ValueError):
        SourceSnapshot("ll2", datetime(2030, 1, 1), "https://example.test", b"x")
