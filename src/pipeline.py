"""
Spaceship Titanic: end-to-end training pipeline.

Runs the whole thing from the command line:
    load data -> feature engineering -> fill missing values ->
    compare models with cross-validation -> evaluate the best one ->
    retrain on all data -> write submissions/submission.csv

Run from the project root:
    python src/pipeline.py

The notebook in notebooks/ tells the same story with charts and explanations.
This script is the clean, reproducible version.
"""
import warnings
warnings.filterwarnings("ignore")  # this machine's math library prints harmless numeric warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("future.no_silent_downcasting", True)

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
FEATURES = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "Deck", "CabinSide", "GroupSize", "TotalSpend"]


def engineer(df):
    """Pull useful features out of the raw columns."""
    df = df.copy()
    # Cabin "B/0/P" -> deck and side
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"], df["CabinSide"] = cabin[0], cabin[2]
    # PassengerId "0001_01" -> travel group, and how big that group is
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    # Total spend across all amenities
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1, min_count=1)
    return df


def impute(df):
    """Fill missing values using logic from the data, not blind averages."""
    df = df.copy()

    # CryoSleep: asleep passengers spend nothing, so infer from spending when missing.
    m = df["CryoSleep"].isna()
    df.loc[m, "CryoSleep"] = df.loc[m, "TotalSpend"].fillna(0).eq(0)
    df["CryoSleep"] = df["CryoSleep"].astype(bool)

    # Spend columns: missing charge means no charge -> 0. Rebuild TotalSpend after.
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)

    # HomePlanet: people in the same group share a home planet.
    group_planet = df.dropna(subset=["HomePlanet"]).groupby("Group")["HomePlanet"].first()
    m = df["HomePlanet"].isna()
    df.loc[m, "HomePlanet"] = df.loc[m, "Group"].map(group_planet)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])

    # CabinSide: everyone in a group is on the same side of the ship.
    group_side = df.dropna(subset=["CabinSide"]).groupby("Group")["CabinSide"].first()
    m = df["CabinSide"].isna()
    df.loc[m, "CabinSide"] = df.loc[m, "Group"].map(group_side)
    df["CabinSide"] = df["CabinSide"].fillna(df["CabinSide"].mode()[0])

    # Everything else: most common value, and median for Age.
    for col in ["Deck", "Destination", "VIP"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["VIP"] = df["VIP"].astype(bool)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


def build_matrix(df, columns=None):
    """One-hot encode the features. Align to given columns for the test set."""
    X = pd.get_dummies(df[FEATURES], drop_first=True)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def main():
    train = impute(engineer(pd.read_csv("data/train.csv")))
    test = impute(engineer(pd.read_csv("data/test.csv")))

    y = train["Transported"].astype(int)
    X = build_matrix(train)
    X_test = build_matrix(test, columns=X.columns)
    print(f"Features: {X.shape[1]} | train rows: {X.shape[0]} | test rows: {X_test.shape[0]}")

    # Compare three models with 5-fold cross-validation
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    print("\n5-fold cross-validation accuracy:")
    cv = {}
    for name, model in models.items():
        s = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        cv[name] = s.mean()
        print(f"  {name:22s} {s.mean():.4f}  (+/- {s.std():.4f})")
    best_name = max(cv, key=cv.get)
    print("Best model:", best_name)

    # Honest holdout evaluation of the best model
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    best = HistGradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr)
    pred = best.predict(X_val)
    print(f"\nHoldout accuracy: {accuracy_score(y_val, pred):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_val, pred))
    print(classification_report(y_val, pred, target_names=["Not Transported", "Transported"]))

    # Retrain on all data and write the Kaggle submission
    final = HistGradientBoostingClassifier(random_state=RANDOM_STATE).fit(X, y)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": final.predict(X_test).astype(bool),
    })
    submission.to_csv("submissions/submission.csv", index=False)
    print("\nWrote submissions/submission.csv", submission.shape)


if __name__ == "__main__":
    main()
