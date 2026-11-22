import pytest
from fastapi.testclient import TestClient
from api.main import app 

import api.main as api_main
from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY


EXPECTED_DESCRIPTIONS = {
    "logistic_regression": "Straightforward method to predict whether a passenger was transported",
    "random_forest": "Combine many decision trees, learn patterns in passenger details",
    "hist_gradient_boosting": "Build trees in stages, each tree learning from earlier mistakes",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert set(app.state.models) == set(MODEL_NAMES)


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert set(body) == set(MODEL_NAMES)
    for model_name, metadata in body.items():
        assert metadata["model_name"] == model_name
        assert metadata["display_name"] == MODEL_REGISTRY[model_name]["display_name"]
        assert metadata["artifact_version"] == "v1"
        assert "metrics" in metadata


def test_list_models(client):
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()["models"]

    assert [model["key"] for model in models] == list(MODEL_NAMES)
    for model in models:
        assert model["description"] == EXPECTED_DESCRIPTIONS[model["key"]]
        assert model["display_name"]
        assert 0 <= model["metrics"]["cv_accuracy_mean"] <= 1
        assert 0 <= model["metrics"]["holdout_accuracy"] <= 1


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_predict(client, model_name):
    response = client.post(
        "/predict",
        json={"PassengerId": "0001_01", "FoodCourt": 0, "model": model_name}
    )

    body = response.json()

    assert response.status_code == 200
    assert isinstance(body["transported"], bool)
    assert 0 <= body["probability"] <= 1
    assert body["model_name"] == model_name
    assert body["model_version"] == "v1"
    assert "request_id" in body


def test_predict_requires_model_choice(client):
    response = client.post(
        "/predict",
        json={"PassengerId": "0001_01", "FoodCourt": 0}
    )
    assert response.status_code == 422


def test_predict_rejects_unknown_model(client):
    response = client.post(
        "/predict",
        json={"PassengerId": "0001_01", "FoodCourt": 0, "model": "neural_network"}
    )
    assert response.status_code == 422


def test_predict_rejects_unexpected_field(client):
    response = client.post(
        "/predict",
        json={
            "PassengerId": "0001_01",
            "Foodccourt": 0,        # test misspelled for FoodCourt
            "model": "hist_gradient_boosting",
        }
    )
    assert response.status_code == 422

def test_predict_hides_internal_error_details(monkeypatch):
    sensitive_message = (
        "Could not load /Users/example/private/artifacts/model.joblib"
    )

    def raise_prediction_error(model, passenger):
        raise RuntimeError(sensitive_message)

    monkeypatch.setattr(
        api_main,
        "predict_probability",
        raise_prediction_error
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            json={"PassengerId": "0001_01", "FoodCourt": 0, "model": "random_forest"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be completed"}
    assert sensitive_message not in response.text


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_predict_batch(client, model_name):
    passengers = [
        {"PassengerId": "0001_01", "FoodCourt": 0},
        {"PassengerId": "0002_01", "FoodCourt": 100}
    ]
    response = client.post(
        "/predict-batch",
        json={"model": model_name, "passengers": passengers}
    )
    body = response.json()

    assert response.status_code == 200
    assert len(body["predictions"]) == 2

    assert body["predictions"][0]["passenger_id"] == "0001_01"
    assert body["predictions"][1]["passenger_id"] == "0002_01"

    for prediction in body["predictions"]:
        assert isinstance(prediction["transported"], bool)
        assert 0 <= prediction["probability"] <= 1

    assert body["model_name"] == model_name
    assert body["model_version"] == "v1"
    assert "request_id" in body


def test_predict_batch_requires_model_choice(client):
    response = client.post(
        "/predict-batch",
        json={"passengers": [{"PassengerId": "0001_01"}]}
    )
    assert response.status_code == 422


def test_predict_batch_rejects_unknown_model(client):
    response = client.post(
        "/predict-batch",
        json={"model": "xgboost", "passengers": [{"PassengerId": "0001_01"}]}
    )
    assert response.status_code == 422
