from datetime import datetime, timezone
from uuid import UUID

from SLA.launch_generator.collect_ll2 import _manifest_bytes


def test_manifest_is_deterministic_and_links_previous_collection():
    as_of = datetime(2030, 1, 1, tzinfo=timezone.utc)
    previous = UUID("00000000-0000-0000-0000-000000000001")
    entries = [{"snapshot_id": "page-1", "content_hash": "abc"}]
    first = _manifest_bytes(as_of, entries, previous)
    second = _manifest_bytes(as_of, entries, previous)
    assert first == second
    assert b'"previous_snapshot_id":"00000000-0000-0000-0000-000000000001"' in first
