"""Tests for the boarding form checks and the friendly API error messages."""

import typing
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import validation
from spaceship_titanic.schemas import PredictionRequest
from validation import (
    CONNECTION_MESSAGE,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    api_error_message,
    form_error_message,
    rejected_fields,
    validate_payload,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

FULL_PAYLOAD = {
    "PassengerId": "0001_01",
    "HomePlanet": "Europa",
    "CryoSleep": False,
    "Cabin": "B/12/P",
    "Destination": "TRAPPIST-1e",
    "Age": 39,
    "VIP": False,
    "RoomService": 0,
    "FoodCourt": 0,
    "ShoppingMall": 0,
    "Spa": 0,
    "VRDeck": 0,
    "model": "random_forest",
}


# --- the copied rules still match the API schema ---------------------------

def _pattern(field):
    for meta in PredictionRequest.model_fields[field].metadata:
        if getattr(meta, "pattern", None):
            return meta.pattern
    raise AssertionError(f"no pattern on {field}")


def _choices(field):
    annotation = PredictionRequest.model_fields[field].annotation
    literal = next(a for a in typing.get_args(annotation) if a is not type(None))
    return typing.get_args(literal)


def test_rules_match_the_api_schema():
    assert validation.PASSENGER_ID_PATTERN == _pattern("PassengerId")
    assert validation.CABIN_PATTERN == _pattern("Cabin")
    assert validation.HOME_PLANETS == _choices("HomePlanet")
    assert validation.DESTINATIONS == _choices("Destination")


def test_labels_cover_every_api_field():
    assert set(FIELD_LABELS) == set(PredictionRequest.model_fields)


def test_required_fields_match_the_api_schema():
    required = {
        name for name, field in PredictionRequest.model_fields.items()
        if field.is_required()
    }
    assert set(REQUIRED_FIELDS) == required


# --- validate_payload / form_error_message ----------------------------------

def test_full_form_passes():
    problems = validate_payload(FULL_PAYLOAD)
    assert problems == {"missing": [], "invalid": []}
    assert form_error_message(problems) is None
    # and the API agrees
    PredictionRequest(**FULL_PAYLOAD)


def test_empty_form_lists_every_required_field():
    problems = validate_payload({})
    assert problems == {"missing": ["model", "PassengerId"], "invalid": []}
    assert form_error_message(problems) == "Please fill in: Oracle, Boarding pass ID."


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_boarding_pass_id_is_missing(blank):
    problems = validate_payload({**FULL_PAYLOAD, "PassengerId": blank})
    assert problems["missing"] == ["PassengerId"]
    assert form_error_message(problems) == "Please fill in: Boarding pass ID."


@pytest.mark.parametrize("bad_id", ["1", "0001-01", "001_01", "0001_1", "abcd_ef", "0001_01x"])
def test_badly_formed_boarding_pass_id_is_invalid(bad_id):
    problems = validate_payload({**FULL_PAYLOAD, "PassengerId": bad_id})
    assert problems == {"missing": [], "invalid": ["PassengerId"]}
    assert form_error_message(problems) == (
        "Please fix: Boarding pass ID: "
        "use the format 0001_01 (4 digits, underscore, 2 digits)."
    )


def test_optional_fields_may_be_left_out():
    payload = {"PassengerId": "0001_01", "model": "random_forest"}
    assert validate_payload(payload) == {"missing": [], "invalid": []}


@pytest.mark.parametrize("field", validation.SPEND_FIELDS)
@pytest.mark.parametrize("blank", [None, ""])
def test_blank_spending_is_fine(field, blank):
    problems = validate_payload({**FULL_PAYLOAD, field: blank})
    assert problems == {"missing": [], "invalid": []}
    assert form_error_message(problems) is None


def test_all_spending_blank_is_accepted_by_the_api():
    payload = {**FULL_PAYLOAD, **{field: None for field in validation.SPEND_FIELDS}}
    assert validate_payload(payload) == {"missing": [], "invalid": []}
    PredictionRequest(**payload)


@pytest.mark.parametrize("field", validation.SPEND_FIELDS)
def test_negative_spending_is_still_rejected(field):
    problems = validate_payload({**FULL_PAYLOAD, field: -1})
    assert problems == {"missing": [], "invalid": [field]}
    assert "use 0 or more credits" in form_error_message(problems)


@pytest.mark.parametrize(
    "field, value",
    [
        ("Cabin", "Z/1/P"),
        ("Cabin", "B/12"),
        ("HomePlanet", "Pluto"),
        ("Destination", "Narnia"),
        ("Age", -1),
        ("Age", 121),
        ("Age", "old"),
        ("CryoSleep", "yes"),
        ("VIP", 1),
        ("Spa", -50),
        ("VRDeck", True),
    ],
)
def test_bad_optional_values_are_invalid(field, value):
    problems = validate_payload({**FULL_PAYLOAD, field: value})
    assert problems == {"missing": [], "invalid": [field]}
    assert FIELD_LABELS[field] in form_error_message(problems)


def test_missing_and_invalid_are_reported_together():
    problems = validate_payload({**FULL_PAYLOAD, "model": None, "Age": 200})
    message = form_error_message(problems)
    assert message.startswith("Please fill in: Oracle.")
    assert "Please fix: Age: use an age from 0 to 120." in message


# --- API errors -------------------------------------------------------------

def test_422_names_the_rejected_fields_by_their_labels():
    body = {
        "detail": [
            {"loc": ["body", "PassengerId"], "msg": "String should match pattern"},
            {"loc": ["body", "Cabin"], "msg": "String should match pattern"},
            {"loc": ["body", "PassengerId"], "msg": "duplicate"},
        ]
    }
    assert rejected_fields(body) == ["PassengerId", "Cabin"]
    message = api_error_message(422, body)
    assert message.startswith("The gate scanner rejected your boarding pass. Please fix: ")
    assert "Boarding pass ID: use the format 0001_01" in message
    assert "Cabin (deck, number, side): use the format deck/number/side" in message


@pytest.mark.parametrize("body", [None, {}, {"detail": "nope"}, [1, 2], {"detail": [{"loc": ["body"]}]}])
def test_422_without_field_details_still_reads_well(body):
    assert api_error_message(422, body) == (
        "The gate scanner rejected your boarding pass. Please check your details."
    )


def test_422_from_the_real_api_is_explained():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json={**FULL_PAYLOAD, "PassengerId": "bad"})
    assert response.status_code == 422
    assert "Boarding pass ID" in api_error_message(422, response.json())


def test_server_and_other_errors_are_short_and_friendly():
    assert api_error_message(500) == (
        "The oracle stumbled while reading your fate. Please try again in a moment."
    )
    assert api_error_message(503).startswith("The oracle stumbled")
    assert api_error_message(404) == (
        "The oracle couldn't take your request (error 404). Please try again."
    )


# --- in the app -------------------------------------------------------------

@pytest.fixture
def app(monkeypatch):
    # closed port: the app uses its built-in model list and the API is "down"
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _pick_model(at):
    picker = next(r for r in at.radio if "Which oracle" in r.label)
    picker.set_value("random_forest").run()


def _id_input(at):
    return next(t for t in at.text_input if t.label == "Boarding pass ID")


def _no_api_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(a) or None)
    return calls


def test_empty_boarding_pass_id_blocks_boarding(app, monkeypatch):
    calls = _no_api_calls(monkeypatch)
    _pick_model(app)
    _id_input(app).set_value("")
    app.button[0].click().run()

    assert not app.exception
    assert [e.value for e in app.error] == ["Please fill in: Boarding pass ID."]
    assert any("⚠ Required" in md.value for md in app.markdown)
    assert calls == []


def test_badly_formed_boarding_pass_id_blocks_boarding(app, monkeypatch):
    calls = _no_api_calls(monkeypatch)
    _pick_model(app)
    _id_input(app).set_value("12-34")
    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value.startswith("Please fix: Boarding pass ID")
    assert any("⚠ Use the format 0001_01" in md.value for md in app.markdown)
    assert calls == []


def test_blank_spending_is_sent_as_unknown(app, monkeypatch):
    sent = []

    def fake_post(url, json=None, **kwargs):
        sent.append(json)
        raise requests.exceptions.ConnectionError

    monkeypatch.setattr(requests, "post", fake_post)
    _pick_model(app)
    app.button[0].click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [CONNECTION_MESSAGE]
    assert [sent[0][field] for field in validation.SPEND_FIELDS] == [None] * 5


def test_api_down_shows_friendly_message(app):
    _pick_model(app)
    app.button[0].click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [CONNECTION_MESSAGE]


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.mark.parametrize(
    "status_code, body, expected",
    [
        (422, {"detail": [{"loc": ["body", "Cabin"], "msg": "bad"}]},
         "The gate scanner rejected your boarding pass. Please fix: "
         "Cabin (deck, number, side): use the format deck/number/side, like B/12/P."),
        (500, {"detail": "Prediction could not be completed"},
         "The oracle stumbled while reading your fate. Please try again in a moment."),
        (502, ValueError("not json"),
         "The oracle stumbled while reading your fate. Please try again in a moment."),
        (200, {"unexpected": True}, validation.BAD_REPLY_MESSAGE),
    ],
)
def test_api_errors_show_friendly_messages(app, monkeypatch, status_code, body, expected):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _FakeResponse(status_code, body)
    )
    _pick_model(app)
    app.button[0].click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [expected]
