"""Guided natural-language intake for candidate SLA specifications."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from math import radians
from typing import Any


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _count(text: str) -> int:
    if text.isdigit():
        return int(text)
    normalized = text.lower().replace("-", " ").strip()
    units = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
             "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19}
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    if normalized in units:
        return units[normalized]
    if normalized in tens:
        return tens[normalized]
    parts = normalized.split()
    if len(parts) == 2 and parts[0] in tens and parts[1] in units:
        return tens[parts[0]] + units[parts[1]]
    raise ValueError(f"Unsupported written count: {text}")


def _money_match(text: str):
    return (re.search(r"[$€£]\s*([\d,.]+)\s*(million|m|billion|bn|b)?\b", text)
            or re.search(r"([\d,.]+)\s*(million|m|billion|bn|b)\s+(?:contract|deal|revenue|value)", text)
            or re.search(r"(?:worth|value|revenue|contract\s+value|contract value:)[^\d]{0,12}([\d,.]+)\s*(million|m|billion|bn|b)?\b", text))


def _deadline_seconds(text: str, as_of: str) -> int | None:
    duration = re.search(r"(?:deadline:?|within|by)\s+(\d+(?:\.\d+)?)\s*(days?|months?|years?)", text)
    if duration:
        amount, unit = float(duration.group(1)), duration.group(2)
        factor = 86400 if unit.startswith("day") else 30 * 86400 if unit.startswith("month") else 365 * 86400
        return round(amount * factor)
    date_match = re.search(r"(?:deadline:?|by|before|no later than)\s+(\d{4}-\d{2}-\d{2})", text)
    if date_match:
        target = datetime.fromisoformat(date_match.group(1)).replace(tzinfo=timezone.utc)
        origin = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        seconds = round((target - origin).total_seconds())
        if seconds <= 0:
            raise ValueError("The service deadline must be after the planning as-of date.")
        return seconds
    return None


def interpret_request(text: str, problem: dict[str, Any]) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Describe the service you need.")
    out = deepcopy(problem)
    lower = text.lower()

    money = _money_match(lower)
    deadline_s = _deadline_seconds(lower, out["as_of"])
    norad = re.search(r"norad(?:\s+ids?)?\s*[:#-]?\s*([\d][\d,\s]{3,})", lower)
    orbit = re.search(r"(?:orbit|orbital elements?)\s*:\s*([^\n;]+)", lower)
    a_element = re.search(r"(?:^|[\s,;])a\s*=\s*([\d.]+)\s*km", lower)
    i_element = re.search(r"(?:^|[\s,;])i\s*=\s*([\d.]+)\s*(?:deg(?:ree)?s?|°)", lower)
    raan_element = re.search(r"raan\s*=\s*([\d.]+)\s*(?:deg(?:ree)?s?|°)", lower)
    compact_orbit = a_element and i_element and raan_element
    count_token = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|thirty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|forty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|fifty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|sixty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|seventy(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|eighty(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|ninety(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
    satellite_count_match = (re.search(rf"refuell?ing\s+for\s+({count_token})\s+(?:of\s+my\s+)?satellites?", lower)
                             or re.search(rf"({count_token})\s+(?:of\s+my\s+)?satellites?", lower))
    requested_satellite_count = _count(satellite_count_match.group(1)) if satellite_count_match else 1
    satellite_count = requested_satellite_count
    cadence_match = re.search(r"(?:service\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+satellites?\s+per\s+year", lower)
    horizon_match = re.search(r"over\s+(\d+(?:\.\d+)?)\s+years?", lower)
    phased_program = satellite_count > 1 and cadence_match and horizon_match
    cadence_count = _count(cadence_match.group(1)) if cadence_match else None
    first_tranche = bool(re.search(r"program\s+scope\s*:\s*(?:first(?:[- ]year)?\s+tranche|first\s+year)", lower))
    aggregate_program = bool(re.search(r"program\s+scope\s*:\s*(?:aggregate|full|five[- ]year)", lower))
    if phased_program and first_tranche:
        satellite_count = cadence_count
    elif phased_program and aggregate_program:
        deadline_s = round(float(horizon_match.group(1)) * 365 * 86400)
    # The guided chat labels follow-up answers as ``Propellant basis: ...``.
    # Accept concise conversational replies as well as full phrases so an answer
    # such as "each" does not send the user back to the same question.
    per_satellite_basis = re.search(
        r"(?:propellant\s+basis\s*:\s*)?(?:each|per\s+(?:satellite|sat)|for\s+each(?:\s+satellite)?)\b",
        lower,
    )
    total_basis = re.search(
        r"(?:propellant\s+basis\s*:\s*)?(?:total|combined|across\s+(?:all|the\s+(?:fleet|satellites?)))\b",
        lower,
    )
    quantity_basis = per_satellite_basis or total_basis
    propellant = (re.search(r"([\d,.]+)\s*kg(?:\s+of)?\s*(?:xenon|xe|propellant)", lower)
                  or re.search(r"(?:xenon|xe|propellant)(?:\s+requirement)?\s*[:=]?\s*([\d,.]+)\s*kg", lower))

    questions = []
    if not (norad or orbit or compact_orbit):
        target_question = (f"I understand that this programme covers {requested_satellite_count} satellites. "
                           "Please give their NORAD IDs or a representative constellation orbit."
                           if satellite_count_match else "Which satellites need service?")
        questions.append({"key": "targets", "question": target_question, "placeholder": "NORAD IDs, or a=700 km, i=56 deg, RAAN=40 deg"})
    if phased_program and not (first_tranche or aggregate_program):
        questions.append({
            "key": "program_scope",
            "question": (f"I read this as {requested_satellite_count} satellites served at "
                         f"{cadence_count} per year over {horizon_match.group(1)} years. "
                         "This MVP plans one tranche at a time. Should I test the first-year "
                         f"tranche of {cadence_count} satellites, or an aggregate "
                         f"{horizon_match.group(1)}-year commitment?"),
            "placeholder": "First-year tranche, or aggregate commitment",
        })
    if not propellant:
        questions.append({"key": "propellant", "question": "How much propellant must be delivered?", "placeholder": "For example: 100 kg xenon in total"})
    if satellite_count > 1 and propellant and not quantity_basis:
        amount_text = propellant.group(1)
        questions.append({"key": "basis", "question": f"Is {amount_text} kg required for each satellite, or across all {satellite_count}?", "placeholder": "Per satellite, or total"})
    if deadline_s is None:
        questions.append({"key": "deadline", "question": "When must the service be completed?", "placeholder": "For example: 2030-12-31 or within 12 months"})
    if not money:
        questions.append({"key": "value", "question": "What is the contract value?", "placeholder": "For example: $24 million"})
    if questions:
        return {
            "status": "needs_information",
            "questions": questions,
            "message": "I understand the service request. I need a few details before creating the candidate SLA.",
            "engine": "guided-sla-intake-v0.2",
        }

    changes: list[dict[str, Any]] = []
    def set_change(label: str, path: str, old: Any, new: Any) -> None:
        changes.append({"label": label, "path": path, "old": old, "new": new})

    value = _number(money.group(1))
    suffix = money.group(2) or ""
    value *= 1_000_000_000 if suffix in {"billion", "bn", "b"} else 1_000_000 if suffix in {"million", "m"} else 1
    candidate = out["candidate_slas"][0]
    set_change("Contract value", "candidate_slas[0].revenue", candidate["revenue"], value)
    candidate["revenue"] = value

    set_change("Service deadline", "candidate_slas[0].deadline_s", candidate["deadline_s"], deadline_s)
    candidate["deadline_s"] = deadline_s

    stated_amount = _number(propellant.group(1))
    per_satellite = satellite_count > 1 and bool(per_satellite_basis)
    amount = stated_amount * satellite_count if per_satellite else stated_amount
    candidate["target_count"] = satellite_count
    if phased_program:
        candidate["service_program"] = {
            "total_target_count": requested_satellite_count,
            "services_per_year": cadence_count,
            "horizon_s": round(float(horizon_match.group(1)) * 365 * 86400),
            "planning_view": "first_tranche" if first_tranche else "aggregate",
        }
    candidate["propellant_quantity_basis"] = "per_satellite" if per_satellite else "total"
    old_amount = candidate.get("inventory", {}).get("xenon_kg", 0)
    candidate.setdefault("inventory", {})["xenon_kg"] = amount
    set_change("Satellite fleet", "candidate_slas[0].target_count", 1, satellite_count)
    delivery_label = f"{amount:g} kg xenon total" + (f" ({stated_amount:g} kg × {satellite_count})" if per_satellite else "")
    set_change("Propellant delivery", "candidate_slas[0].inventory.xenon_kg", old_amount, delivery_label)

    if norad:
        ids = re.findall(r"\d{4,8}", norad.group(1))
        candidate["target_asset_ids"] = [f"NORAD-{x}" for x in ids]
        candidate["target_asset_id"] = candidate["target_asset_ids"][0]
        set_change("Target satellites", "candidate_slas[0].target_asset_ids", "", ", ".join(candidate["target_asset_ids"]))
    elif compact_orbit:
        entered_a_km = float(a_element.group(1))
        # Values below Earth radius are understood as altitude above mean equator.
        semimajor_axis_km = entered_a_km + 6378.137 if entered_a_km < 6378.137 else entered_a_km
        candidate["target_orbit"] = {
            "semimajor_axis_km": semimajor_axis_km,
            "inclination_rad": radians(float(i_element.group(1))),
            "raan_rad": radians(float(raan_element.group(1))),
        }
        candidate["target_asset_id"] = f"ORBIT-{entered_a_km:g}KM-{float(i_element.group(1)):g}DEG"
        interpretation = (f"{semimajor_axis_km:.3f} km semimajor axis, "
                          f"{float(i_element.group(1)):g}° inclination, "
                          f"{float(raan_element.group(1)):g}° RAAN")
        if entered_a_km < 6378.137:
            interpretation += f" (interpreting a={entered_a_km:g} km as altitude)"
        set_change("Target orbit", "candidate_slas[0].target_orbit", "", interpretation)
    else:
        candidate["target_orbit_description"] = orbit.group(1).strip()
        set_change("Target orbit", "candidate_slas[0].target_orbit_description", "", candidate["target_orbit_description"])

    # Reliability is predicted from launch and operational scenarios. A threshold is
    # applied only when the commercial contract explicitly supplies one.
    reliability = (re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:required\s+)?(?:reliability|service level)", lower)
                   or re.search(r"(?:required\s+reliability|service level)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", lower))
    if reliability:
        required = float(reliability.group(1)) / 100
        candidate["required_reliability"] = required
        candidate["required_reliability_source"] = "contract"
        set_change("Contract reliability threshold", "candidate_slas[0].required_reliability", "not specified", required)
    else:
        candidate["required_reliability"] = 0.0
        candidate["required_reliability_source"] = "model_output"
        set_change("Reliability", "candidate_slas[0].reliability_mode", "", "Estimated by the planner")

    return {"status": "ready", "problem": out, "changes": changes, "engine": "guided-sla-intake-v0.2"}
