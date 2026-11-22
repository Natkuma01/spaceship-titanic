"""UI tests for the model picker in the Streamlit app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from model_metrics import FALLBACK_MODELS


APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP_PATH = APP_DIR / "streamlit_app.py"
PICKER_LABEL = "Which oracle should read your fate?"


@pytest.fixture
def app(monkeypatch):
    # Point at a closed port so the app uses its built-in model list.
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def picker(at):
    return next(radio for radio in at.radio if PICKER_LABEL in radio.label)


def test_picker_lists_every_model_with_its_description(app):
    radio = picker(app)

    # laid out side by side so the CSS can draw each option as a card
    assert radio.proto.horizontal

    assert radio.options == [m["display_name"] for m in FALLBACK_MODELS.values()]
    captions = list(radio.proto.captions)
    assert captions == [m["description"] for m in FALLBACK_MODELS.values()]


def test_no_model_is_picked_at_start_and_boarding_is_locked(app):
    assert picker(app).value is None
    assert app.button[0].proto.disabled
    assert any("Choose a model above" in c.value for c in app.caption)


def test_picking_a_model_unlocks_boarding_and_shows_its_accuracy(app):
    picker(app).set_value("random_forest").run()

    assert not app.exception
    assert not app.button[0].proto.disabled
    html = "".join(md.value for md in app.markdown)
    assert "Random Forest holdout accuracy" in html
    assert '<tr class="picked"><td>▸ Random Forest</td>' in html


@pytest.mark.parametrize("model_key", list(FALLBACK_MODELS))
def test_each_card_can_be_picked(app, model_key):
    picker(app).set_value(model_key).run()

    assert not app.exception
    assert picker(app).value == model_key
    assert not app.button[0].proto.disabled
    html = "".join(md.value for md in app.markdown)
    name = FALLBACK_MODELS[model_key]["display_name"]
    assert f"{name} holdout accuracy" in html


def test_picker_cards_are_styled(app):
    css = app.markdown[0].value
    assert '.st-key-oracle_picker [role="radiogroup"]' in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in css
    assert ':has(input:checked)' in css
    assert "@media (max-width: 640px)" in css


def test_track_record_has_one_short_explanation_line(app):
    html = "".join(md.value for md in app.markdown)
    assert (
        '<div class="track-note">How often each oracle guessed right on '
        "passengers it never saw.</div>"
    ) in html
    assert not any("cross-validation" in c.value for c in app.caption)


def test_track_record_legend_explains_each_number(app):
    html = "".join(md.value for md in app.markdown)
    assert '<div class="track-legend">' in html
    for label, text in [
        ("CV accuracy", "average score from 5-fold cross validation on the training data."),
        ("Holdout accuracy", "share of correct guesses on 20% of passengers held back from training."),
        ("Holdout ROC-AUC", "how well the model ranks transported above not transported (1.0 is perfect)."),
    ]:
        assert (
            f'<div class="track-note"><span class="track-label">{label}:</span> {text}</div>'
        ) in html
    css = app.markdown[0].value
    assert ".track-legend .track-label" in css
