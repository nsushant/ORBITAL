from datetime import datetime, timezone
from io import StringIO

from SLA.launch_generator.features import (
    build_capacity_rows,
    build_count_rows,
    build_mission_rows,
    chronological_split,
)
from SLA.launch_generator.ingestion.gcat import import_gcat
from SLA.launch_generator.ingestion.ll2 import normalize_response


LAUNCHES = """#Launch_Tag\tLaunch_Date\tPiece\tType\tName\tPLName\tJCAT\tSatOwner\tSatState\tLV_Type\tFlight_ID\tPlatform\tLaunch_Site\tLaunch_Pad\tAscent_Site\tAscent_Pad\tAgency\tLVState\tLaunch_Code\tLTCite
2029-001A\t2029 Dec 20 0000:00\t2029-001A\tP\tOld\tOld\tS1\tO\tUS\tV\tF1\t-\tSITE\tP\t-\t-\tAG\tUS\tOS\tR
2030-001A\t2030 Jan 10 0000:00\t2030-001A\tP\tFuture\tFuture\tS2\tO\tUS\tV\tF2\t-\tSITE\tP\t-\t-\tAG\tUS\tOS\tR
2030-002A\t2030 Mar 10 0000:00\t2030-002A\tP\tLater\tLater\tS3\tO\tUS\tV\tF3\t-\tSITE\tP\t-\t-\tAG\tUS\tOS\tR
"""

OBJECTS = """#JCAT\tMass\tDryMass\tTotMass\tPerigee\tApogee\tInc\tOpOrbit
S1\t10\t9\t10\t500\t510\t50\tLEO
S2\t20\t18\t20\t500\t510\t98\tLEO
S3\t30\t27\t30\t500\t510\t50\tLEO
"""


def _snapshot(as_of, net):
    payload = {"count": 1, "results": [{
        "id": "ll2-1", "name": "Future", "net": net,
        "status": {"abbrev": "Go"}, "net_precision": {"abbrev": "DAY"},
        "launch_service_provider": {"name": "Agency"},
        "rocket": {"configuration": {"full_name": "V"}},
        "pad": {"name": "P", "location": {"name": "SITE"}},
        "mission": {"orbit": {"abbrev": "SSO"}},
    }]}
    return normalize_response(payload, as_of, "fixture")


def test_count_features_do_not_use_future_launches():
    gcat = import_gcat(StringIO(LAUNCHES), StringIO(OBJECTS))
    snapshot = _snapshot(datetime(2030, 1, 1, tzinfo=timezone.utc), "2030-01-08T00:00:00Z")
    rows = build_count_rows(gcat, [snapshot], [30 * 86400], provider_aliases={"AG": "provider", "Agency": "provider"})
    row = next(x for x in rows if x.provider == "provider")
    assert row.launches_previous_30d == 1
    assert row.target_realized_launch_count == 1
    assert row.announced_in_horizon == 1


def test_incomplete_count_horizon_is_not_labeled_as_zero():
    gcat = import_gcat(StringIO(LAUNCHES), StringIO(OBJECTS))
    snapshot = _snapshot(datetime(2030, 3, 1, tzinfo=timezone.utc), "2030-03-08T00:00:00Z")
    rows = build_count_rows(gcat, [snapshot], [30 * 86400])
    assert rows == ()


def test_later_snapshot_does_not_change_earlier_revision_feature():
    gcat = import_gcat(StringIO(LAUNCHES), StringIO(OBJECTS))
    first = _snapshot(datetime(2030, 1, 1, tzinfo=timezone.utc), "2030-01-08T00:00:00Z")
    second = _snapshot(datetime(2030, 1, 2, tzinfo=timezone.utc), "2030-01-09T00:00:00Z")
    rows = build_mission_rows(
        gcat, [first, second], launch_crosswalk={"ll2-1": "2030-001A"},
        provider_aliases={"AG": "provider", "Agency": "provider"},
    )
    assert rows[0].revision_count == 0
    assert rows[1].revision_count == 1
    assert rows[0].schedule_error_s == 2 * 86400


def test_capacity_target_and_chronological_split():
    gcat = import_gcat(StringIO(LAUNCHES), StringIO(OBJECTS))
    rows = build_capacity_rows(gcat, provider_aliases={"AG": "provider"})
    assert rows[1].target_manifested_mass_kg == 20.0
    train, calibration, test = chronological_split(
        rows,
        datetime(2030, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 2, 1, tzinfo=timezone.utc),
    )
    assert [len(train), len(calibration), len(test)] == [1, 1, 1]
