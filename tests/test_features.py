"""Tests for deterministic passenger feature engineering."""

import pandas as pd

from spaceship_titanic.features import engineer_features


def make_passenger(**overrides: object) -> pd.DataFrame:
    """Build one valid raw passenger and optionally override selected fields."""
    passenger = {
        "PassengerId": "0042_03",
        "Cabin": "B/12/P",
        "RoomService": 0.0,
        "FoodCourt": 0.0,
        "ShoppingMall": 0.0,
        "Spa": 0.0,
        "VRDeck": 0.0,
    }
    passenger.update(overrides)
    return pd.DataFrame([passenger])


def test_engineer_features_splits_cabin() -> None:
    """Cabin is split into its deck, numeric cabin number, and side."""
    passenger = make_passenger()

    result = engineer_features(passenger)

    assert result.loc[0, "Deck"] == "B"
    assert result.loc[0, "CabinNumber"] == 12
    assert result.loc[0, "CabinSide"] == "P"


def test_engineer_features_extracts_group_member_number() -> None:
    """The suffix of PassengerId becomes a numeric group-member feature."""
    result = engineer_features(make_passenger(PassengerId="0042_03"))

    assert result.loc[0, "GroupMemberNumber"] == 3


def test_engineer_features_calculates_total_spend() -> None:
    """TotalSpend adds the five spending fields and treats missing charges as zero."""
    passenger = make_passenger(
        RoomService=100.0,
        FoodCourt=None,
        ShoppingMall=50.0,
        Spa=0.0,
        VRDeck=25.0,
    )

    result = engineer_features(passenger)

    assert result.loc[0, "TotalSpend"] == 175.0


def test_engineer_features_sets_all_missing_spend_to_zero() -> None:
    """The fixed domain rule maps five missing charges to zero total spend."""
    passenger = make_passenger(
        RoomService=None,
        FoodCourt=None,
        ShoppingMall=None,
        Spa=None,
        VRDeck=None,
    )

    result = engineer_features(passenger)

    assert result.loc[0, "TotalSpend"] == 0.0


def test_engineer_features_does_not_modify_input() -> None:
    """Feature engineering returns a new frame and preserves the raw input."""
    passenger = make_passenger()
    original = passenger.copy(deep=True)

    engineer_features(passenger)

    pd.testing.assert_frame_equal(passenger, original)


def test_engineer_features_accepts_missing_identifiers() -> None:
    """One passenger with missing cabin and ID produces missing engineered values."""
    result = engineer_features(make_passenger(PassengerId=None, Cabin=None))

    assert pd.isna(result.loc[0, "Deck"])
    assert pd.isna(result.loc[0, "CabinNumber"])
    assert pd.isna(result.loc[0, "CabinSide"])
    assert pd.isna(result.loc[0, "GroupMemberNumber"])
