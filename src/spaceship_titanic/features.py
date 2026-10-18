"""
This module creates new features from each passenger's raw data.
It uses the same feature engineering for model training and new predictions.

- It creates useful features from raw passenger information.
- Each feature uses only that passenger's row (row-based feature engineering).
- The same logic is used during both training and prediction.
"""

import pandas as pd


SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic row-based features without modifying the input."""
    copy_df = df.copy()

    cabin = copy_df["Cabin"].str.split("/", expand=True).reindex(columns=range(3))
    copy_df["Deck"], copy_df["CabinNumber"], copy_df["CabinSide"] = cabin[0], cabin[1], cabin[2]
    copy_df["CabinNumber"] = pd.to_numeric(copy_df["CabinNumber"], errors="coerce")

    copy_df["GroupMemberNumber"] = copy_df["PassengerId"].str.split("_").str[1]
    copy_df["GroupMemberNumber"] = pd.to_numeric(copy_df["GroupMemberNumber"], errors="coerce")

    spend_values = copy_df[SPEND_COLS].apply(pd.to_numeric, errors="coerce")
    copy_df["TotalSpend"] = spend_values.fillna(0).sum(axis=1)

    return copy_df
