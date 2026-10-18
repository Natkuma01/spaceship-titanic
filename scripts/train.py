"""Train and evaluate the Spaceship Titanic model."""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.base import clone


from spaceship_titanic.model import model_pipeline
import joblib
import json


RANDOM_STATE = 42

ARTIFACT_PATH = Path("artifacts/spaceship_titanic_pipeline_v1.joblib")

METADATA_PATH = Path("artifacts/spaceship_titanic_pipeline_v1_metadata.json")


def main() -> None:
    data = pd.read_csv("data/train.csv")

    X = data.drop(columns="Transported")
    y = data["Transported"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y) 

    print("Training rows: ", len(X_train))
    print("Holdout rows: ", len(X_holdout))

    # add 5 fold
    cross_validation = StratifiedKFold(
        n_splits = 5,
        shuffle = True,
        random_state=RANDOM_STATE
    )
    # for each hold, sklearn fits the entire pipeline on 4 folds
    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="accuracy"
    )
    print("Cross-validation scores: ", cv_scores)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"CV standard deviation: {cv_scores.std():.4f}")

    model_pipeline.fit(X_train, y_train)

    holdout_predictions = model_pipeline.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)

    print(f"Holdout accuracy: {holdout_accuracy:.4f}")

    print(classification_report(
        y_holdout,
        holdout_predictions,
        target_names=["Not transported", "Transported"]
    ))

    holdout_probabilities = model_pipeline.predict_proba(X_holdout)[:, 1]
    holdout_roc_auc = roc_auc_score(y_holdout, holdout_probabilities)

    print(f"Holdout ROC-AUC: {holdout_roc_auc:.4f}")

    matrix = confusion_matrix(y_holdout, holdout_predictions)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    print("Correctly predicted not transported: ", true_negative)
    print("Incorrectly predicted transported: ", false_positive)
    print("Incorrectly predicted not transported: ", false_negative)
    print("Correctly predicted transported: ", true_positive)

    production_pipeline = clone(model_pipeline)
    production_pipeline.fit(X, y)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(production_pipeline, ARTIFACT_PATH)

    print(f"Saved model artifact: {ARTIFACT_PATH}")

    metadata = {
        "artifact_version": "v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(X),
        "decision_threshold": 0.5,
        "metrics": {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "holdout_accuracy": float(holdout_accuracy),
            "holdout_roc_auc": float(holdout_roc_auc),
        }
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )
    print(f"Saved model metadata: {METADATA_PATH}")

if __name__ == "__main__":
    main()