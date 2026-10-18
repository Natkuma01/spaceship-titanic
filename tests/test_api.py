from fastapi.testclient import TestClient
from api.main import app 

import api.main as api_main


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert hasattr(app.state, "model")


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")

        assert response.status_code == 200

        body = response.json()

        assert body["artifact_version"] == "v1"
        assert "metrics" in body


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"PassengerId": "0001_01", "FoodCourt": 0}
        )

        body = response.json()

        assert response.status_code == 200
        assert isinstance(body["transported"], bool)
        assert 0 <= body["probability"] <= 1
        assert body["model_version"] == "v1"
        assert "request_id" in body

def test_predict_rejects_unexpected_field():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"PassengerId": "0001_01", "Foodccourt": 0}        # test misspelled for FoodCourt
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
            json={"PassengerId": "0001_01", "FoodCourt": 0}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be completed"}
    assert sensitive_message not in response.text