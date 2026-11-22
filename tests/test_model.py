"""Integration test for the complete model pipeline."""

import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

from spaceship_titanic.model import MODEL_NAMES, build_pipeline, model_pipeline

# Create a small dataset for pipeline integration tests
def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    passengers = pd.DataFrame(
        {
            "PassengerId": [
                "0001_01",
                "0002_01",
                "0003_01",
                "0004_01"
            ],
            "HomePlanet": ["Europa","Earth","Mars","Earth"],
            "CryoSleep" : [False, True, False, True],
            "Cabin": ["B/0/P", "F/1/S", None, "G/2/S"],
            "Destination": [
                "TRAPPIST-1e",
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22"
            ],
            "Age": [39.0, 24.0, None, 30.0],
            "VIP": [False, False, True, False],
            "RoomService": [0.0, 0.0, 300.0, 0.0],
            "FoodCourt": [1000.0, 0.0, 50.0, 0.0],
            "ShoppingMall": [0.0, 0.0, 25.0, 0.0],
            "Spa": [50.0, 0.0, None, 0.0],
            "VRDeck": [100.0, 0.0, 10.0, 0.0]
        }
    )
    transported = pd.Series([True, True, False, False], name="Transported")
    return passengers, transported


def test_saved_pipeline_preserves_probabilities(
    tmp_path: Path,
) -> None:
    """Reloading a saved pipeline preserves its predictions."""
    passengers, transported = make_training_data()

    fitted_pipeline = clone(model_pipeline)
    fitted_pipeline.fit(passengers, transported)

    expected_probabilities = fitted_pipeline.predict_proba(passengers)

    artifact_path = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact_path)

    loaded_pipeline = joblib.load(artifact_path)
    actual_probabilities = loaded_pipeline.predict_proba(passengers)

    np.testing.assert_allclose(
        actual_probabilities,
        expected_probabilities,
    )

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_each_pipeline_fits_and_predicts(model_name: str) -> None:
    """Every registered model fits and predicts on a small sample."""
    passengers, transported = make_training_data()

    pipeline = build_pipeline(model_name)
    pipeline.fit(passengers, transported)

    predictions = pipeline.predict(passengers)
    probabilities = pipeline.predict_proba(passengers)

    assert len(predictions) == len(passengers)
    assert probabilities.shape == (len(passengers), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_registry_has_three_models() -> None:
    assert set(MODEL_NAMES) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_build_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        build_pipeline("not_a_model")


def test_model_pipeline_is_hist_gradient_boosting() -> None:
    classifier = model_pipeline.named_steps["classifier"]
    assert isinstance(classifier, HistGradientBoostingClassifier)
    assert classifier.max_iter == 592
