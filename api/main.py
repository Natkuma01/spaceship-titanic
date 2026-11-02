from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from spaceship_titanic.schemas import PassengerRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionItem, BatchPredictionResponse
from uuid import uuid4


import json
import joblib
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "spaceship_titanic_pipeline_v1.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "spaceship_titanic_pipeline_v1_metadata.json"
)


def load_metadata():
    return json.loads(METADATA_PATH.read_text())


def load_model():
    return joblib.load(ARTIFACT_PATH)


def passenger_to_frame(passenger: PassengerRequest) -> pd.DataFrame:
    passenger_dict = passenger.model_dump()
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
    app.state.model = load_model()
    app.state.metadata = load_metadata()
    yield


app = FastAPI(
    title="Spaceship Titanic API",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request):
    return request.app.state.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerRequest, request: Request) -> PredictionResponse:
    try:
        probability = predict_probability(
            request.app.state.model,
            passenger
    )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        ) from None
    
    metadata = request.app.state.metadata
    threshold = metadata["decision_threshold"]
    model_version = metadata["artifact_version"]

    return PredictionResponse(
        transported = probability >= threshold,
        probability = probability,
        model_version = model_version,
        request_id = str(uuid4())
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    try:
        probabilities = predict_probabilities(
            request.app.state.model,
            batch
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        ) from None

    metadata = request.app.state.metadata
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
        model_version=model_version,
        request_id=str(uuid4())
    )