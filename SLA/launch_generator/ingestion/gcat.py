"""Normalize GCAT launchlog, object, and payload TSV files.

GCAT data are CC BY 4.0. Product exports using these records must retain the
GCAT/J. McDowell attribution recorded on :class:`GcatImportResult`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, TextIO

GCAT_SOURCE_URL = "https://planet4589.org/space/gcat/"
GCAT_ATTRIBUTION = "Data from GCAT (J. McDowell, planet4589.org/space/gcat)"


@dataclass(frozen=True)
class ParsedDate:
    timestamp_utc: datetime | None
    precision: str
    uncertain: bool
    raw: str


@dataclass(frozen=True)
class GcatPayload:
    jcat: str
    piece: str | None
    name: str | None
    payload_name: str | None
    owner: str | None
    state: str | None
    mass_kg: float | None
    dry_mass_kg: float | None
    total_mass_kg: float | None
    perigee_km: float | None
    apogee_km: float | None
    inclination_deg: float | None
    orbit_code: str | None
    orbit_class: str | None
    program: str | None
    plane: str | None
    mission_class: str | None
    category: str | None
    payload_result: str | None

    @property
    def best_mass_kg(self) -> float | None:
        return self.total_mass_kg if self.total_mass_kg is not None else self.mass_kg


@dataclass(frozen=True)
class GcatLaunch:
    launch_id: str
    launch_time_utc: datetime | None
    launch_time_precision: str
    launch_time_uncertain: bool
    raw_launch_date: str
    vehicle: str | None
    flight_id: str | None
    platform: str | None
    site: str | None
    pad: str | None
    ascent_site: str | None
    ascent_pad: str | None
    agency: str | None
    provider_state: str | None
    launch_code: str | None
    outcome: str
    citation: str | None
    payloads: tuple[GcatPayload, ...]

    @property
    def manifested_payload_mass_kg(self) -> float | None:
        masses = [x.best_mass_kg for x in self.payloads if x.best_mass_kg is not None]
        return sum(masses) if masses else None


@dataclass(frozen=True)
class GcatImportResult:
    launches: tuple[GcatLaunch, ...]
    source_updated_text: str | None
    source_url: str = GCAT_SOURCE_URL
    licence: str = "CC-BY-4.0"
    attribution: str = GCAT_ATTRIBUTION


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return None if cleaned in {"", "-", "*"} else cleaned


def _number(value: str | None) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    text = text.rstrip("?~>").strip()
    try:
        return float(text)
    except ValueError:
        return None


_DATE_RE = re.compile(
    r"^(?P<year>\d{4})\s+(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2})"
    r"(?:\s+(?P<time>\d{4}:\d{2}))?"
)


def parse_gcat_datetime(value: str) -> ParsedDate:
    raw = value.rstrip("\r\n")
    text = raw.strip()
    uncertain = "?" in text or "~" in text
    match = _DATE_RE.match(text)
    if not match:
        return ParsedDate(None, "unknown", uncertain, raw)
    base = f"{match.group('year')} {match.group('month')} {match.group('day')}"
    time_text = match.group("time")
    try:
        if time_text:
            timestamp = datetime.strptime(f"{base} {time_text}", "%Y %b %d %H%M:%S")
            precision = "second"
        else:
            timestamp = datetime.strptime(base, "%Y %b %d")
            precision = "day"
    except ValueError:
        return ParsedDate(None, "unknown", uncertain, raw)
    return ParsedDate(timestamp.replace(tzinfo=timezone.utc), precision, uncertain, raw)


def _iter_tsv(source: str | Path | TextIO) -> tuple[list[dict[str, str]], str | None]:
    close = False
    if hasattr(source, "read"):
        stream = source
    else:
        stream = Path(source).open("r", encoding="utf-8-sig", newline="")
        close = True
    try:
        lines = iter(stream)
        header_line = next(lines, None)
        if header_line is None:
            raise ValueError("GCAT TSV is empty")
        header = header_line.rstrip("\r\n")
        if header.startswith("#"):
            header = header[1:]
        fieldnames = next(csv.reader([header], delimiter="\t"))
        rows: list[dict[str, str]] = []
        updated: str | None = None
        for line in lines:
            if line.startswith("#"):
                if line.lower().startswith("# updated"):
                    updated = line[1:].strip()
                continue
            if not line.strip():
                continue
            values = next(csv.reader([line.rstrip("\r\n")], delimiter="\t"))
            if len(values) < len(fieldnames):
                values.extend([""] * (len(fieldnames) - len(values)))
            rows.append(dict(zip(fieldnames, values, strict=False)))
        return rows, updated
    finally:
        if close:
            stream.close()


def _outcome(launch_code: str | None) -> str:
    code = (launch_code or "").replace(" ", "")
    if code.startswith("OS"):
        return "success"
    if code.startswith("OF") or code.startswith("OE"):
        return "failure"
    if code.startswith("OU"):
        return "unknown"
    return "other"


def classify_orbit(perigee_km: float | None, apogee_km: float | None,
                   inclination_deg: float | None, orbit_code: str | None) -> str | None:
    code = (orbit_code or "").upper()
    if "GEO" in code or "GSO" in code:
        return "GEO"
    if "GTO" in code:
        return "GTO"
    if "MEO" in code:
        return "MEO"
    if perigee_km is None or apogee_km is None:
        return None
    if apogee_km >= 30_000 and perigee_km < 5_000:
        return "GTO"
    mean_altitude = (perigee_km + apogee_km) / 2.0
    if mean_altitude >= 30_000:
        return "GEO"
    if mean_altitude >= 2_000:
        return "MEO"
    if inclination_deg is None:
        return "LEO"
    if 96.0 <= inclination_deg <= 102.0:
        return "SSO_POLAR"
    if inclination_deg >= 80.0:
        return "POLAR_LEO"
    if inclination_deg < 30.0:
        return "LOW_INCLINATION_LEO"
    return "MID_INCLINATION_LEO"


def _payload(jcat: str, launch_row: Mapping[str, str],
             object_row: Mapping[str, str] | None,
             payload_row: Mapping[str, str] | None) -> GcatPayload:
    obj = object_row or {}
    pay = payload_row or {}
    perigee = _number(obj.get("Perigee"))
    apogee = _number(obj.get("Apogee"))
    inclination = _number(obj.get("Inc"))
    orbit_code = _clean(obj.get("OpOrbit"))
    return GcatPayload(
        jcat=jcat,
        piece=_clean(launch_row.get("Piece")),
        name=_clean(launch_row.get("Name")),
        payload_name=_clean(launch_row.get("PLName")),
        owner=_clean(launch_row.get("SatOwner")),
        state=_clean(launch_row.get("SatState")),
        mass_kg=_number(obj.get("Mass")),
        dry_mass_kg=_number(obj.get("DryMass")),
        total_mass_kg=_number(obj.get("TotMass")),
        perigee_km=perigee,
        apogee_km=apogee,
        inclination_deg=inclination,
        orbit_code=orbit_code,
        orbit_class=classify_orbit(perigee, apogee, inclination, orbit_code),
        program=_clean(pay.get("Program")),
        plane=_clean(pay.get("Plane")),
        mission_class=_clean(pay.get("Class")),
        category=_clean(pay.get("Category")),
        payload_result=_clean(pay.get("Result")),
    )


def _index(rows: Iterable[Mapping[str, str]], field: str) -> dict[str, Mapping[str, str]]:
    return {
        key: row
        for row in rows
        if (key := (_clean(row.get(field)) or ""))
    }


def import_gcat(launchlog: str | Path | TextIO,
                object_catalog: str | Path | TextIO | None = None,
                payload_catalog: str | Path | TextIO | None = None) -> GcatImportResult:
    """Import the GCAT orbital launch log with optional object enrichment."""
    launch_rows, updated = _iter_tsv(launchlog)
    object_rows = _iter_tsv(object_catalog)[0] if object_catalog is not None else []
    payload_rows = _iter_tsv(payload_catalog)[0] if payload_catalog is not None else []
    objects = _index(object_rows, "JCAT")
    payload_info = _index(payload_rows, "JCAT")

    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in launch_rows:
        launch_id = _clean(row.get("Launch_Tag"))
        if launch_id is None:
            continue
        grouped.setdefault(launch_id, []).append(row)

    launches: list[GcatLaunch] = []
    for launch_id, rows in grouped.items():
        first = rows[0]
        date = parse_gcat_datetime(first.get("Launch_Date", ""))
        payloads = []
        for row in rows:
            if not (row.get("Type") or "").strip().startswith("P"):
                continue
            jcat = _clean(row.get("JCAT"))
            if jcat is None:
                continue
            payloads.append(_payload(jcat, row, objects.get(jcat), payload_info.get(jcat)))
        launches.append(GcatLaunch(
            launch_id=launch_id,
            launch_time_utc=date.timestamp_utc,
            launch_time_precision=date.precision,
            launch_time_uncertain=date.uncertain,
            raw_launch_date=date.raw,
            vehicle=_clean(first.get("LV_Type")),
            flight_id=_clean(first.get("Flight_ID")),
            platform=_clean(first.get("Platform")),
            site=_clean(first.get("Launch_Site")),
            pad=_clean(first.get("Launch_Pad")),
            ascent_site=_clean(first.get("Ascent_Site")),
            ascent_pad=_clean(first.get("Ascent_Pad")),
            agency=_clean(first.get("Agency")),
            provider_state=_clean(first.get("LVState")),
            launch_code=_clean(first.get("Launch_Code")),
            outcome=_outcome(first.get("Launch_Code")),
            citation=_clean(first.get("LTCite")),
            payloads=tuple(payloads),
        ))

    launches.sort(key=lambda x: (x.launch_time_utc or datetime.min.replace(tzinfo=timezone.utc), x.launch_id))
    return GcatImportResult(tuple(launches), updated)
