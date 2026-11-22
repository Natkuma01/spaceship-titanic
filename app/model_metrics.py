"""Model list helpers for the Streamlit app.

Kept apart from streamlit_app.py so they can be imported (and tested)
without starting Streamlit.
"""

from html import escape


# used when the API can't list its models; matches the API's /models endpoint
# and the metrics saved in artifacts/*_metadata.json
FALLBACK_MODELS = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "description": "Straightforward method to predict whether a passenger was transported",
        "metrics": {
            "cv_accuracy_mean": 0.7886112677076168,
            "cv_accuracy_std": 0.003809065621718025,
            "holdout_accuracy": 0.78953421506613,
            "holdout_roc_auc": 0.8816766932808456,
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Combine many decision trees, learn patterns in passenger details",
        "metrics": {
            "cv_accuracy_mean": 0.80212962053075,
            "cv_accuracy_std": 0.00642891481272452,
            "holdout_accuracy": 0.81196089706728,
            "holdout_roc_auc": 0.8946974026042742,
        },
    },
    "hist_gradient_boosting": {
        "display_name": "HistGradientBoosting",
        "description": "Build trees in stages, each tree learning from earlier mistakes",
        "metrics": {
            "cv_accuracy_mean": 0.8047160316319195,
            "cv_accuracy_std": 0.004766436463843938,
            "holdout_accuracy": 0.8188614146060954,
            "holdout_roc_auc": 0.9122558823685031,
        },
    },
}


def models_from_api(body: dict) -> dict:
    """Turn a GET /models response body into {key: model info}."""
    return {
        model["key"]: {
            "display_name": model["display_name"],
            "description": model["description"],
            "metrics": dict(model.get("metrics") or {}),
        }
        for model in body["models"]
    }


def format_percent(value) -> str:
    """0.8119 -> '81.2%'. Missing values show as a dash."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def format_cv_accuracy(metrics: dict) -> str:
    """Cross-validation accuracy as 'mean ± std', e.g. '80.2% ± 0.6%'."""
    mean = metrics.get("cv_accuracy_mean")
    std = metrics.get("cv_accuracy_std")
    if mean is None:
        return "—"
    if std is None:
        return format_percent(mean)
    return f"{format_percent(mean)} ± {format_percent(std)}"


def metrics_table_rows(models: dict) -> list[dict]:
    """One row per model for the comparison table, in the models' order."""
    rows = []
    for key, model in models.items():
        metrics = model.get("metrics") or {}
        rows.append(
            {
                "key": key,
                "Model": model["display_name"],
                "CV accuracy": format_cv_accuracy(metrics),
                "Holdout accuracy": format_percent(metrics.get("holdout_accuracy")),
                "Holdout ROC-AUC": format_percent(metrics.get("holdout_roc_auc")),
            }
        )
    return rows


def holdout_chart_data(models: dict, picked_key=None) -> list[dict]:
    """Holdout accuracy per model (as a percentage) for the bar chart.

    Models without a holdout accuracy are left out.
    """
    data = []
    for key, model in models.items():
        accuracy = (model.get("metrics") or {}).get("holdout_accuracy")
        if accuracy is None:
            continue
        data.append(
            {
                "Model": model["display_name"],
                "Holdout accuracy": round(accuracy * 100, 1),
                "Picked": key == picked_key,
            }
        )
    return data


def metrics_table_html(rows: list[dict], picked_key=None) -> str:
    """Comparison table as HTML, with the picked model's row highlighted."""
    columns = ["Model", "CV accuracy", "Holdout accuracy", "Holdout ROC-AUC"]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        row_class = ' class="picked"' if row["key"] == picked_key else ""
        marker = "▸ " if row["key"] == picked_key else ""
        cells = [f"<td>{marker}{escape(row['Model'])}</td>"] + [
            f"<td>{escape(row[column])}</td>" for column in columns[1:]
        ]
        body.append(f"<tr{row_class}>{''.join(cells)}</tr>")
    return (
        f'<table class="metrics-table"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )
