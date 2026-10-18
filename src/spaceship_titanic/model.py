from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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

preprocessor = ColumnTransformer(
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
            "passthrough",
            PASSTHROUGH_NUMERIC_FEATURES,
        )
    ],
    remainder="drop"
)

model_pipeline = Pipeline(
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
            preprocessor,
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(random_state=42)
        ),
    ]
)