"""Train and evaluate every Spaceship Titanic model and save each one."""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix


from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY, build_pipeline
import joblib
import json


RANDOM_STATE = 42

ARTIFACTS_DIR = Path("artifacts")

ARTIFACT_VERSION = "v1"


def artifact_path(model_name: str) -> Path:
    return ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}.joblib"


def metadata_path(model_name: str) -> Path:
    return ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}_metadata.json"


def train_and_save(model_name, X, y, X_train, X_holdout, y_train, y_holdout) -> dict:
    display_name = MODEL_REGISTRY[model_name]["display_name"]
    print(f"\n===== {display_name} ({model_name}) =====")

    # add 5 fold
    cross_validation = StratifiedKFold(
        n_splits = 5,
        shuffle = True,
        random_state=RANDOM_STATE
    )
    # for each hold, sklearn fits the entire pipeline on 4 folds
    cv_scores = cross_val_score(
        build_pipeline(model_name),
        X_train,
        y_train,
        cv=cross_validation,
        scoring="accuracy"
    )
    print("Cross-validation scores: ", cv_scores)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"CV standard deviation: {cv_scores.std():.4f}")

    evaluation_pipeline = build_pipeline(model_name)
    evaluation_pipeline.fit(X_train, y_train)

    holdout_predictions = evaluation_pipeline.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)

    print(f"Holdout accuracy: {holdout_accuracy:.4f}")

    print(classification_report(
        y_holdout,
        holdout_predictions,
        target_names=["Not transported", "Transported"]
    ))

    holdout_probabilities = evaluation_pipeline.predict_proba(X_holdout)[:, 1]
    holdout_roc_auc = roc_auc_score(y_holdout, holdout_probabilities)

    print(f"Holdout ROC-AUC: {holdout_roc_auc:.4f}")

    matrix = confusion_matrix(y_holdout, holdout_predictions)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    print("Correctly predicted not transported: ", true_negative)
    print("Incorrectly predicted transported: ", false_positive)
    print("Incorrectly predicted not transported: ", false_negative)
    print("Correctly predicted transported: ", true_positive)

    production_pipeline = build_pipeline(model_name)
    production_pipeline.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact_path = artifact_path(model_name)
    joblib.dump(production_pipeline, model_artifact_path, compress=3)

    print(f"Saved model artifact: {model_artifact_path}")

    metadata = {
        "model_name": model_name,
        "display_name": display_name,
        "artifact_version": ARTIFACT_VERSION,
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

    model_metadata_path = metadata_path(model_name)
    model_metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )
    print(f"Saved model metadata: {model_metadata_path}")

    return metadata


def main() -> None:
    data = pd.read_csv("data/train.csv")

    X = data.drop(columns="Transported")
    y = data["Transported"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y) 

    print("Training rows: ", len(X_train))
    print("Holdout rows: ", len(X_holdout))

    summary = [
        train_and_save(model_name, X, y, X_train, X_holdout, y_train, y_holdout)
        for model_name in MODEL_NAMES
    ]

    print("\n===== Summary =====")
    for metadata in summary:
        metrics = metadata["metrics"]
        print(
            f"{metadata['display_name']:<22} "
            f"CV {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}  "
            f"holdout {metrics['holdout_accuracy']:.4f}  "
            f"ROC-AUC {metrics['holdout_roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
