from datetime import datetime, timezone
from io import StringIO
from uuid import UUID

from SLA.launch_generator.ingestion.gcat import import_gcat
from SLA.launch_generator.ingestion.ll2 import normalize_response
from SLA.storage.calendar_repository import gcat_launch_rows, ll2_state_rows


SNAPSHOT_ID = UUID("00000000-0000-0000-0000-000000000001")


def test_gcat_time_is_seconds_from_snapshot():
    text = """#Launch_Tag\tLaunch_Date\tPiece\tType\tName\tPLName\tJCAT\tSatOwner\tSatState\tLV_Type\tFlight_ID\tPlatform\tLaunch_Site\tLaunch_Pad\tAscent_Site\tAscent_Pad\tAgency\tLVState\tLaunch_Code\tLTCite
2030-001A\t2030 Jan  2 0000:00\t2030-001A\tP\tPayload\tP\tS1\tO\tUS\tVehicle\tF1\t-\tSite\tPad\t-\t-\tAgency\tUS\tOS\tRef
"""
    result = import_gcat(StringIO(text))
    launches, payloads = gcat_launch_rows(result, datetime(2030, 1, 3, tzinfo=timezone.utc), SNAPSHOT_ID)
    assert launches[0][3] == -86400.0
    assert payloads[0][2] == "S1"


def test_ll2_rows_keep_relative_seconds():
    payload = {"count": 1, "results": [{"id": "L1", "name": "M", "net": "2030-01-02T00:00:00Z"}]}
    snapshot = normalize_response(payload, datetime(2030, 1, 1, tzinfo=timezone.utc), "u")
    assert ll2_state_rows(snapshot, SNAPSHOT_ID)[0][4] == 86400.0
