"""Checks that every model has a saved artifact and matching metadata."""

import json

import pytest

from api.main import ARTIFACT_VERSION, ARTIFACTS_DIR, artifact_path, metadata_path
from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_artifact_files_exist(model_name):
    model_file = ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}.joblib"
    metadata_file = (
        ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}_metadata.json"
    )

    assert artifact_path(model_name) == model_file
    assert metadata_path(model_name) == metadata_file
    assert model_file.is_file()
    assert metadata_file.is_file()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metadata_matches_filename(model_name):
    metadata = json.loads(metadata_path(model_name).read_text())

    assert metadata["model_name"] == model_name
    assert metadata["display_name"] == MODEL_REGISTRY[model_name]["display_name"]
    assert metadata["artifact_version"] == ARTIFACT_VERSION
    assert set(metadata["metrics"]) == {
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "holdout_accuracy",
        "holdout_roc_auc",
    }
