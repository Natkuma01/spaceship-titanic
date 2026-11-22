from typing import get_args

import pytest
from pydantic import ValidationError

from spaceship_titanic.model import MODEL_NAMES
from spaceship_titanic.schemas import (
    BatchPredictionRequest,
    ModelName,
    PassengerRequest,
    PredictionRequest,
)


def test_valid_passenger_request():
    passenger = PassengerRequest(
        PassengerId = "0001_01",
        HomePlanet = "Earth",
        Age = 25,
        FoodCourt = 0
    )

    assert passenger.PassengerId == "0001_01"
    assert passenger.FoodCourt == 0

def test_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        PassengerRequest(
            PassengerId = "0001_01",
            Foodccourt = 0              # misspelled the field, expected to raise error
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_prediction_request_accepts_each_model(model_name):
    request = PredictionRequest(PassengerId="0001_01", model=model_name)

    assert request.model == model_name


def test_prediction_request_requires_model():
    with pytest.raises(ValidationError):
        PredictionRequest(PassengerId="0001_01")


def test_prediction_request_rejects_unknown_model():
    with pytest.raises(ValidationError):
        PredictionRequest(PassengerId="0001_01", model="neural_network")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_batch_request_accepts_each_model(model_name):
    batch = BatchPredictionRequest(
        model=model_name,
        passengers=[{"PassengerId": "0001_01"}],
    )

    assert batch.model == model_name


def test_batch_request_requires_model():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(passengers=[{"PassengerId": "0001_01"}])


def test_batch_request_rejects_unknown_model():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(
            model="xgboost",
            passengers=[{"PassengerId": "0001_01"}],
        )


def test_schema_model_names_match_registry():
    assert set(get_args(ModelName)) == set(MODEL_NAMES)
