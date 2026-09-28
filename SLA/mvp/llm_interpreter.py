"""Optional OpenAI language normalization for the SLA intake."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .nl_interpreter import interpret_request as interpret_deterministically

API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "satellite_count": {"type": ["integer", "null"], "minimum": 1},
        "norad_ids": {"type": "array", "items": {"type": "string"}},
        "orbit_altitude_or_semimajor_axis_km": {"type": ["number", "null"]},
        "inclination_deg": {"type": ["number", "null"]},
        "raan_deg": {"type": ["number", "null"]},
        "orbit_description": {"type": ["string", "null"]},
        "propellant_kg": {"type": ["number", "null"], "minimum": 0},
        "propellant_basis": {"type": "string", "enum": ["unknown", "per_satellite", "total"]},
        "deadline_days": {"type": ["number", "null"], "minimum": 0},
        "deadline_date": {"type": ["string", "null"]},
        "contract_value_usd": {"type": ["number", "null"], "minimum": 0},
        "required_reliability": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    },
    "required": [
        "satellite_count", "norad_ids", "orbit_altitude_or_semimajor_axis_km",
        "inclination_deg", "raan_deg", "orbit_description", "propellant_kg",
        "propellant_basis", "deadline_days", "deadline_date",
        "contract_value_usd", "required_reliability",
    ],
}

INSTRUCTIONS = """Extract an on-orbit refuelling SLA from the complete conversation.
Do not guess missing facts. Return null, an empty list, or 'unknown' when absent.
Interpret terse labelled replies in context: 'Propellant basis: each' means
per_satellite and 'total' means total. For a relative deadline return
deadline_days (1 year = 365 days; 1 month = 30 days); preserve an explicit date
as YYYY-MM-DD. Preserve a=700 km as 700. Only extract required_reliability when
the user explicitly states a contract threshold; planner reliability is output.
"""


def _output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"]
    raise ValueError("The language model returned no structured intake.")


def _extract(text: str, as_of: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": os.getenv("OPENAI_SLA_MODEL", DEFAULT_MODEL),
        "store": False,
        "instructions": INSTRUCTIONS,
        "input": f"Planning as-of timestamp: {as_of}\n\nConversation:\n{text}",
        "text": {"format": {"type": "json_schema", "name": "sla_intake", "strict": True, "schema": SCHEMA}},
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=25) as response:
            body = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"OpenAI intake request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise ValueError(f"OpenAI intake request could not connect: {exc.reason}") from exc
    return json.loads(_output_text(body))


def _canonical_text(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    count = fields.get("satellite_count")
    if count:
        parts.append(f"Refuelling for {int(count)} satellites")
    ids = [str(x) for x in fields.get("norad_ids", []) if str(x).strip()]
    if ids:
        parts.append("NORAD IDs: " + ", ".join(ids))
    a, inc, raan = (fields.get("orbit_altitude_or_semimajor_axis_km"), fields.get("inclination_deg"), fields.get("raan_deg"))
    if a is not None and inc is not None and raan is not None:
        parts.append(f"a={a:g} km, i={inc:g} deg, RAAN={raan:g} deg")
    elif fields.get("orbit_description"):
        parts.append("Orbit: " + fields["orbit_description"])
    propellant = fields.get("propellant_kg")
    if propellant is not None:
        parts.append(f"Propellant requirement: {propellant:g} kg xenon")
    basis = fields.get("propellant_basis")
    if basis == "per_satellite":
        parts.append("Propellant basis: per satellite")
    elif basis == "total":
        parts.append("Propellant basis: total")
    if fields.get("deadline_date"):
        parts.append("Deadline: " + fields["deadline_date"])
    elif fields.get("deadline_days") is not None:
        parts.append(f"Deadline: within {fields['deadline_days']:g} days")
    if fields.get("contract_value_usd") is not None:
        parts.append(f"Contract value: ${fields['contract_value_usd']:g}")
    if fields.get("required_reliability") is not None:
        parts.append(f"Required reliability: {100 * fields['required_reliability']:g}%")
    return "\n".join(parts)


def interpret_request(text: str, problem: dict[str, Any]) -> dict[str, Any]:
    """Extract with OpenAI when configured, then validate deterministically."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        result = interpret_deterministically(text, problem)
        result["engine"] = "guided-sla-intake-v0.3-offline"
        return result
    fields = _extract(text, problem["as_of"], api_key)
    result = interpret_deterministically(_canonical_text(fields), problem)
    result["engine"] = f"openai-{os.getenv('OPENAI_SLA_MODEL', DEFAULT_MODEL)}-structured-intake"
    return result
