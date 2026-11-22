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
    return next(radio for radio in at.radio if radio.label == PICKER_LABEL)


def test_picker_lists_every_model_with_its_description(app):
    radio = picker(app)

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
