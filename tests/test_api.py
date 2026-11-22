import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app 

import api.main as api_main
from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY
from spaceship_titanic.schemas import PassengerRequest


EXPECTED_DESCRIPTIONS = {
    "logistic_regression": "Straightforward method to predict whether a passenger was transported",
    "random_forest": "Combine many decision trees, learn patterns in passenger details",
    "hist_gradient_boosting": "Build trees in stages, each tree learning from earlier mistakes",
}

METRIC_KEYS = {
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "holdout_accuracy",
    "holdout_roc_auc",
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
        assert set(model["metrics"]) == METRIC_KEYS
        for value in model["metrics"].values():
            assert 0 <= value <= 1


def test_list_models_metrics_match_model_info(client):
    models = client.get("/models").json()["models"]
    model_info = client.get("/model-info").json()

    for model in models:
        assert model["metrics"] == model_info[model["key"]]["metrics"]


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


@pytest.mark.parametrize("endpoint", ["/predict", "/predict-batch"])
@pytest.mark.parametrize("bad_model", [None, "", "Random Forest", "RANDOM_FOREST"])
def test_rejects_null_empty_or_misspelled_model(client, endpoint, bad_model):
    passenger = {"PassengerId": "0001_01", "FoodCourt": 0}
    if endpoint == "/predict":
        body = {**passenger, "model": bad_model}
    else:
        body = {"model": bad_model, "passengers": [passenger]}

    response = client.post(endpoint, json=body)
    assert response.status_code == 422


SAMPLE_PASSENGER = {
    "PassengerId": "0013_01",
    "HomePlanet": "Europa",
    "CryoSleep": False,
    "Cabin": "B/12/S",
    "Destination": "55 Cancri e",
    "Age": 34,
    "VIP": False,
    "RoomService": 0,
    "FoodCourt": 1800,
    "ShoppingMall": 40,
    "Spa": 650,
    "VRDeck": 300,
}


def expected_probability(model_name, passenger=SAMPLE_PASSENGER):
    """What that model's loaded pipeline says, called directly."""
    # fill in the optional columns the same way the request schema does
    frame = pd.DataFrame([PassengerRequest(**passenger).model_dump()])
    return float(app.state.models[model_name].predict_proba(frame)[0, 1])


def test_predict_uses_the_requested_model(client):
    probabilities = {}
    for model_name in MODEL_NAMES:
        response = client.post(
            "/predict", json={**SAMPLE_PASSENGER, "model": model_name}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["model_name"] == model_name
        assert body["probability"] == pytest.approx(
            expected_probability(model_name), abs=1e-12
        )
        threshold = app.state.metadata[model_name]["decision_threshold"]
        assert body["transported"] == (body["probability"] >= threshold)
        probabilities[model_name] = body["probability"]

    # the three models really are different, so a mix-up would show
    assert len({round(p, 9) for p in probabilities.values()}) >= 2


def test_predict_batch_uses_the_requested_model(client):
    second = {**SAMPLE_PASSENGER, "PassengerId": "0014_01", "CryoSleep": True,
              "FoodCourt": 0, "Spa": 0, "VRDeck": 0, "ShoppingMall": 0}
    results = {}
    for model_name in MODEL_NAMES:
        response = client.post(
            "/predict-batch",
            json={"model": model_name, "passengers": [SAMPLE_PASSENGER, second]},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["model_name"] == model_name
        got = [p["probability"] for p in body["predictions"]]
        want = [expected_probability(model_name, p) for p in (SAMPLE_PASSENGER, second)]
        assert got == pytest.approx(want, abs=1e-12)
        results[model_name] = tuple(round(p, 9) for p in got)

    assert len(set(results.values())) >= 2
