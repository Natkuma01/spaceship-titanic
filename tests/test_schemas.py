import pytest
from pydantic import ValidationError

from spaceship_titanic.schemas import PassengerRequest


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

