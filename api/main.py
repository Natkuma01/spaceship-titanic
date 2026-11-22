from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from spaceship_titanic.model import MODEL_NAMES, MODEL_REGISTRY
from spaceship_titanic.schemas import (
    PassengerRequest,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionItem,
    BatchPredictionResponse,
    ModelListResponse,
    ModelSummary,
)
from uuid import uuid4


import json
import joblib
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_VERSION = "v1"


def artifact_path(model_name: str) -> Path:
    return ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}.joblib"


def metadata_path(model_name: str) -> Path:
    return ARTIFACTS_DIR / f"spaceship_titanic_{model_name}_{ARTIFACT_VERSION}_metadata.json"


def load_metadata(model_name: str) -> dict:
    return json.loads(metadata_path(model_name).read_text())


def load_model(model_name: str):
    return joblib.load(artifact_path(model_name))


def load_all_models() -> dict:
    return {model_name: load_model(model_name) for model_name in MODEL_NAMES}


def load_all_metadata() -> dict:
    return {model_name: load_metadata(model_name) for model_name in MODEL_NAMES}


def passenger_to_frame(passenger: PassengerRequest) -> pd.DataFrame:
    # the model choice is not a passenger feature
    passenger_dict = passenger.model_dump(exclude={"model"})
    return pd.DataFrame([passenger_dict])


def passengers_to_frame(batch: BatchPredictionRequest) -> pd.DataFrame:
    passenger_dicts = [
        passenger.model_dump()
        for passenger in batch.passengers
    ]
    return pd.DataFrame(passenger_dicts)


def predict_probability(model, passenger: PassengerRequest) -> float:
    frame = passenger_to_frame(passenger)
    probabilities = model.predict_proba(frame)
    probability = probabilities[0, 1]
    return float(probability)


def predict_probabilities(model, batch:BatchPredictionRequest) -> list[float]:
    frame = passengers_to_frame(batch)
    probabilities = model.predict_proba(frame)[:, 1]
    return probabilities.tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_models()
    app.state.metadata = load_all_metadata()
    yield


app = FastAPI(
    title="Spaceship Titanic API",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    all_metadata = request.app.state.metadata
    return ModelListResponse(
        models=[
            ModelSummary(
                key=model_name,
                display_name=MODEL_REGISTRY[model_name]["display_name"],
                description=MODEL_REGISTRY[model_name]["description"],
                metrics=all_metadata[model_name]["metrics"],
            )
            for model_name in MODEL_NAMES
        ]
    )


@app.get("/model-info")
def model_info(request: Request):
    return request.app.state.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PredictionRequest, request: Request) -> PredictionResponse:
    model_name = passenger.model
    try:
        probability = predict_probability(
            request.app.state.models[model_name],
            passenger
    )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        ) from None
    
    metadata = request.app.state.metadata[model_name]
    threshold = metadata["decision_threshold"]
    model_version = metadata["artifact_version"]

    return PredictionResponse(
        transported = probability >= threshold,
        probability = probability,
        model_name = model_name,
        model_version = model_version,
        request_id = str(uuid4())
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    model_name = batch.model
    try:
        probabilities = predict_probabilities(
            request.app.state.models[model_name],
            batch
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        ) from None

    metadata = request.app.state.metadata[model_name]
    threshold = metadata["decision_threshold"]
    model_version = metadata["artifact_version"]

    predictions = [
        BatchPredictionItem(
            passenger_id=passenger.PassengerId,
            transported=probability >= threshold,
            probability=probability
        )
        for passenger, probability in zip(batch.passengers, probabilities)
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        model_name=model_name,
        model_version=model_version,
        request_id=str(uuid4())
    )
