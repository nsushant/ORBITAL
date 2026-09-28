import json
from datetime import datetime, timezone

from SLA.launch_generator.ingestion.ll2 import diff_snapshots, normalize_response


def _payload(net: str, status: str = "Go"):
    return {
        "count": 1,
        "results": [{
            "id": "launch-1", "name": "Mission One", "net": net,
            "window_start": net, "window_end": net,
            "last_updated": "2030-01-01T00:00:00Z",
            "status": {"abbrev": status},
            "net_precision": {"abbrev": "MIN"},
            "launch_service_provider": {"name": "Provider"},
            "rocket": {"configuration": {"full_name": "Vehicle"}},
            "pad": {"name": "Pad", "location": {"name": "Site"}},
            "mission": {"type": "Communications", "orbit": {"name": "Sun-Synchronous Orbit", "abbrev": "SSO"}},
        }],
    }


def test_normalization_uses_seconds_from_as_of():
    as_of = datetime(2030, 1, 1, tzinfo=timezone.utc)
    snapshot = normalize_response(_payload("2030-01-02T00:00:00Z"), as_of, "https://example")
    launch = snapshot.launches[0]
    assert launch.net_s == 86400.0
    assert launch.provider == "Provider"
    assert launch.orbit_abbrev == "SSO"


def test_diff_compares_absolute_time_across_as_of_epochs():
    old = normalize_response(_payload("2030-01-02T00:00:00Z"), datetime(2030, 1, 1, tzinfo=timezone.utc), "old")
    new = normalize_response(_payload("2030-01-03T00:00:00Z"), datetime(2030, 1, 2, tzinfo=timezone.utc), "new")
    revisions = diff_snapshots(old, new)
    assert [(x.field, x.old_value.isoformat(), x.new_value.isoformat()) for x in revisions] == [
        ("net_s", "2030-01-02T00:00:00+00:00", "2030-01-03T00:00:00+00:00"),
        ("window_start_s", "2030-01-02T00:00:00+00:00", "2030-01-03T00:00:00+00:00"),
        ("window_end_s", "2030-01-02T00:00:00+00:00", "2030-01-03T00:00:00+00:00"),
    ]
