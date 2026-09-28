from io import StringIO

from SLA.launch_generator.ingestion.gcat import import_gcat, parse_gcat_datetime


LAUNCHLOG = """#Launch_Tag\tLaunch_Date\tPiece\tType\tName\tPLName\tJCAT\tSatOwner\tSatState\tLV_Type\tFlight_ID\tPlatform\tLaunch_Site\tLaunch_Pad\tAscent_Site\tAscent_Pad\tAgency\tLVState\tLaunch_Code\tLTCite
# Updated fixture
2026-001A\t2026 Jan  2 0304:05\t2026-001A\tP\tPayload A\tPA\tS00001\tOWNER\tUS\tFalcon 9\tF1\t-\tCC\tLC40\t-\t-\tSPX\tUS\tOS\tREF
2026-001A\t2026 Jan  2 0304:05\t2026-001B\tP\tPayload B\tPB\tS00002\tOWNER\tUS\tFalcon 9\tF1\t-\tCC\tLC40\t-\t-\tSPX\tUS\tOS\tREF
"""

OBJECTS = """#JCAT\tMass\tDryMass\tTotMass\tPerigee\tApogee\tInc\tOpOrbit
S00001\t100\t90\t110\t500\t510\t97.4\tLEO
S00002\t200\t180\t-\t500\t510\t97.4\tLEO
"""

PAYLOADS = """#JCAT\tProgram\tPlane\tClass\tCategory\tResult
S00001\tConstellation\t1\tB\tCOM\tS
S00002\tConstellation\t2\tB\tCOM\tS
"""


def test_import_and_enrichment():
    result = import_gcat(StringIO(LAUNCHLOG), StringIO(OBJECTS), StringIO(PAYLOADS))
    assert result.licence == "CC-BY-4.0"
    assert len(result.launches) == 1
    launch = result.launches[0]
    assert launch.vehicle == "Falcon 9"
    assert launch.outcome == "success"
    assert launch.launch_time_utc.isoformat() == "2026-01-02T03:04:05+00:00"
    assert launch.manifested_payload_mass_kg == 310.0
    assert launch.payloads[0].orbit_class == "SSO_POLAR"
    assert launch.payloads[1].program == "Constellation"


def test_vague_date_is_retained():
    value = parse_gcat_datetime("2026 Jan  2?")
    assert value.timestamp_utc.isoformat() == "2026-01-02T00:00:00+00:00"
    assert value.precision == "day"
    assert value.uncertain
