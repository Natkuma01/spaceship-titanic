"""Check the boarding form before it is sent, and explain API errors in plain words.

The rules mirror spaceship_titanic.schemas.PredictionRequest. They are copied
here (not imported) because the Streamlit image only ships the app folder;
tests/test_app_validation.py checks that the two stay in step.
"""

import re


PASSENGER_ID_PATTERN = r"^\d{4}_\d{2}$"
CABIN_PATTERN = r"^[A-GT]/\d+/(P|S)$"
HOME_PLANETS = ("Earth", "Europa", "Mars")
DESTINATIONS = ("TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22")
AGE_RANGE = (0, 120)
SPEND_FIELDS = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")

# the API needs these; everything else may be left out
REQUIRED_FIELDS = ("model", "PassengerId")

# on-screen label for each API field
FIELD_LABELS = {
    "model": "Oracle",
    "PassengerId": "Boarding pass ID",
    "HomePlanet": "Departing from",
    "CryoSleep": "Cryo-pod sleep",
    "Cabin": "Cabin (deck, number, side)",
    "Destination": "Bound for",
    "Age": "Age",
    "VIP": "VIP status",
    "RoomService": "Room service",
    "FoodCourt": "Food court",
    "ShoppingMall": "Promenade shops",
    "Spa": "Zero-G spa",
    "VRDeck": "VR deck",
}

# what a valid value looks like, in plain words
FIELD_HINTS = {
    "model": "pick one of the oracles",
    "PassengerId": "use the format 0001_01 (4 digits, underscore, 2 digits)",
    "HomePlanet": "pick Earth, Europa or Mars",
    "CryoSleep": "turn it on or off",
    "Cabin": "use the format deck/number/side, like B/12/P",
    "Destination": "pick one of the listed planets",
    "Age": f"use an age from {AGE_RANGE[0]} to {AGE_RANGE[1]}",
    "VIP": "turn it on or off",
    **{field: "use 0 or more credits" for field in SPEND_FIELDS},
}


def label(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _field_problem(field: str, value) -> bool:
    """True when a filled-in value would be rejected by the API."""
    if field == "model":
        return not isinstance(value, str)
    if field == "PassengerId":
        return not isinstance(value, str) or not re.match(PASSENGER_ID_PATTERN, value)
    if field == "Cabin":
        return not isinstance(value, str) or not re.match(CABIN_PATTERN, value)
    if field == "HomePlanet":
        return value not in HOME_PLANETS
    if field == "Destination":
        return value not in DESTINATIONS
    if field in ("CryoSleep", "VIP"):
        return not isinstance(value, bool)
    if field == "Age":
        return not _is_number(value) or not AGE_RANGE[0] <= value <= AGE_RANGE[1]
    if field in SPEND_FIELDS:
        return not _is_number(value) or value < 0
    return False


def validate_payload(payload: dict) -> dict:
    """Return {"missing": [field, ...], "invalid": [field, ...]} in form order."""
    missing, invalid = [], []
    for field in FIELD_LABELS:
        value = payload.get(field)
        if _is_blank(value):
            if field in REQUIRED_FIELDS:
                missing.append(field)
            continue
        if _field_problem(field, value):
            invalid.append(field)
    return {"missing": missing, "invalid": invalid}


def form_error_message(problems: dict) -> str | None:
    """One friendly message for the form, or None when it can be sent."""
    lines = []
    if problems["missing"]:
        names = ", ".join(label(field) for field in problems["missing"])
        lines.append(f"Please fill in: {names}.")
    if problems["invalid"]:
        fixes = "; ".join(
            f"{label(field)}: {FIELD_HINTS.get(field, 'check this value')}"
            for field in problems["invalid"]
        )
        lines.append(f"Please fix: {fixes}.")
    return "\n\n".join(lines) or None


CONNECTION_MESSAGE = "The oracle can't be reached. Is the API running?"
TIMEOUT_MESSAGE = "The oracle took too long to answer. Please try again in a moment."
BAD_REPLY_MESSAGE = "The oracle sent back a reply we couldn't read. Please try again."


def rejected_fields(body) -> list[str]:
    """API field names named in a 422 response body, in order, without repeats."""
    detail = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, list):
        return []
    fields = []
    for item in detail:
        loc = item.get("loc") if isinstance(item, dict) else None
        if not loc:
            continue
        field = str(loc[-1])
        if field == "body" or field in fields:
            continue
        fields.append(field)
    return fields


def api_error_message(status_code: int, body=None) -> str:
    """Plain-words message for a failed /predict response."""
    if status_code == 422:
        fields = rejected_fields(body)
        if not fields:
            return "The gate scanner rejected your boarding pass. Please check your details."
        fixes = "; ".join(
            f"{label(field)}: {FIELD_HINTS.get(field, 'not accepted')}"
            for field in fields
        )
        return f"The gate scanner rejected your boarding pass. Please fix: {fixes}."
    if status_code >= 500:
        return "The oracle stumbled while reading your fate. Please try again in a moment."
    return f"The oracle couldn't take your request (error {status_code}). Please try again."
