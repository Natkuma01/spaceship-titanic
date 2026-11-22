"""Unit tests for the Streamlit app's model metrics helpers."""

import json

import pytest

from api.main import metadata_path
from model_metrics import (
    FALLBACK_MODELS,
    format_cv_accuracy,
    format_percent,
    holdout_chart_data,
    metrics_table_html,
    metrics_table_rows,
    models_from_api,
)
from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY


SAMPLE_MODELS = {
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Trees",
        "metrics": {
            "cv_accuracy_mean": 0.80212962,
            "cv_accuracy_std": 0.00642891,
            "holdout_accuracy": 0.81196089,
            "holdout_roc_auc": 0.89469740,
        },
    },
    "logistic_regression": {
        "display_name": "Logistic <Regression>",
        "description": "Lines",
        "metrics": {},
    },
}


@pytest.mark.parametrize(
    "value, expected",
    [(0.81196089, "81.2%"), (1.0, "100.0%"), (0.0, "0.0%"), (None, "—")],
)
def test_format_percent(value, expected):
    assert format_percent(value) == expected


def test_format_cv_accuracy():
    assert format_cv_accuracy(SAMPLE_MODELS["random_forest"]["metrics"]) == "80.2% ± 0.6%"
    assert format_cv_accuracy({"cv_accuracy_mean": 0.8}) == "80.0%"
    assert format_cv_accuracy({}) == "—"


def test_metrics_table_rows():
    rows = metrics_table_rows(SAMPLE_MODELS)

    assert rows[0] == {
        "key": "random_forest",
        "Model": "Random Forest",
        "CV accuracy": "80.2% ± 0.6%",
        "Holdout accuracy": "81.2%",
        "Holdout ROC-AUC": "89.5%",
    }
    assert rows[1]["Holdout accuracy"] == "—"


def test_holdout_chart_data_marks_picked_and_skips_missing():
    data = holdout_chart_data(SAMPLE_MODELS, "random_forest")

    assert data == [
        {"Model": "Random Forest", "Holdout accuracy": 81.2, "Picked": True}
    ]
    assert holdout_chart_data(SAMPLE_MODELS)[0]["Picked"] is False


def test_metrics_table_html_highlights_pick_and_escapes():
    html = metrics_table_html(metrics_table_rows(SAMPLE_MODELS), "random_forest")

    assert html.count('class="picked"') == 1
    assert '<tr class="picked"><td>▸ Random Forest</td>' in html
    assert "Logistic &lt;Regression&gt;" in html
    assert "<Regression>" not in html


def test_metrics_table_html_without_pick():
    html = metrics_table_html(metrics_table_rows(SAMPLE_MODELS))

    assert 'class="picked"' not in html


def test_models_from_api():
    body = {
        "models": [
            {
                "key": "random_forest",
                "display_name": "Random Forest",
                "description": "Trees",
                "metrics": {"holdout_accuracy": 0.81},
            }
        ]
    }

    assert models_from_api(body) == {
        "random_forest": {
            "display_name": "Random Forest",
            "description": "Trees",
            "metrics": {"holdout_accuracy": 0.81},
        }
    }


def test_fallback_models_match_registry_and_saved_metrics():
    assert list(FALLBACK_MODELS) == list(MODEL_NAMES)

    for model_name, model in FALLBACK_MODELS.items():
        saved_metrics = json.loads(metadata_path(model_name).read_text())["metrics"]

        assert model["display_name"] == MODEL_REGISTRY[model_name]["display_name"]
        assert model["description"] == MODEL_REGISTRY[model_name]["description"]
        assert model["metrics"] == pytest.approx(saved_metrics)
