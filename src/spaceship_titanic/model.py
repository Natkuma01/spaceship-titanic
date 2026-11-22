from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from spaceship_titanic.features import SPEND_COLS, engineer_features

CATEGORICAL_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "CabinSide"]

MEDIAN_IMPUTATED_FEATURES = ["Age"]

ZERO_IMPUTED_FEATURES = [*SPEND_COLS]

PASSTHROUGH_NUMERIC_FEATURES = [
    "CabinNumber",
    "GroupMemberNumber",
    "TotalSpend"
]

age_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

spending_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value=0)
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


def build_preprocessor(impute_numeric_passthrough: bool = False) -> ColumnTransformer:
    """Build the shared column preprocessor.

    HistGradientBoosting handles missing values natively, so the extra numeric
    columns pass straight through. Logistic regression cannot handle missing
    values, so for the other models those columns get a median imputer.
    """
    if impute_numeric_passthrough:
        numeric_step = SimpleImputer(strategy="median")
    else:
        numeric_step = "passthrough"

    return ColumnTransformer(
        transformers = [
            (
                "age",
                age_pipeline,
                MEDIAN_IMPUTATED_FEATURES,
            ),
            (
                "spending",
                spending_pipeline,
                ZERO_IMPUTED_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric_passthrough",
                numeric_step,
                PASSTHROUGH_NUMERIC_FEATURES,
            )
        ],
        remainder="drop"
    )


preprocessor = build_preprocessor()


def _logistic_regression_steps() -> list:
    return [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]


def _random_forest_steps() -> list:
    return [
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def _hist_gradient_boosting_steps() -> list:
    return [
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.026,
                max_iter=592,
                max_depth=5,
                max_leaf_nodes=48,
                min_samples_leaf=38,
                l2_regularization=0.027,
                random_state=42,
            )
        ),
    ]


# key -> display name, description, model steps, whether extra numeric columns need imputing
MODEL_REGISTRY = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "description": "Straightforward method to predict whether a passenger was transported",
        "steps": _logistic_regression_steps,
        "impute_numeric_passthrough": True,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Combine many decision trees, learn patterns in passenger details",
        "steps": _random_forest_steps,
        "impute_numeric_passthrough": True,
    },
    "hist_gradient_boosting": {
        "display_name": "HistGradientBoosting",
        "description": "Build trees in stages, each tree learning from earlier mistakes",
        "steps": _hist_gradient_boosting_steps,
        "impute_numeric_passthrough": False,
    },
}

MODEL_NAMES = tuple(MODEL_REGISTRY)

DEFAULT_MODEL_NAME = "hist_gradient_boosting"


def build_pipeline(model_name: str) -> Pipeline:
    """Build a fresh, unfitted pipeline for one of the registered models."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r}. Choose one of: {', '.join(MODEL_NAMES)}"
        )
    entry = MODEL_REGISTRY[model_name]

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    engineer_features,
                    validate=False,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(entry["impute_numeric_passthrough"]),
            ),
            *entry["steps"](),
        ]
    )


# kept for existing imports: the tuned HistGradientBoosting pipeline
model_pipeline = build_pipeline(DEFAULT_MODEL_NAME)
