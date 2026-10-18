"""Request and response schemas for model predictions."""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


# one passenger submitted for prediction
class PassengerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PassengerId: str = Field(
        pattern = r"^\d{4}_\d{2}$",
        examples = ["0001_01"],
    )
    HomePlanet: Literal["Earth", "Europa", "Mars"] | None = None
    CryoSleep: bool | None = None
    Cabin: str | None = Field(
        default = None,
        pattern = r"^[A-GT]/\d+/(P|S)$",
        examples = ["B/12/P"],
    )
    Destination: Literal[
        "TRAPPIST-1e",
        "55 Cancri e",
        "PSO J318.5-22"
    ] | None = None
    Age: float | None = Field(
        default = None,
        ge = 0,
        le = 120
    )
    VIP: bool | None = None
    RoomService: float | None = Field(
        default = None,
        ge = 0
    )
    FoodCourt: float | None = Field(
        default = None,
        ge = 0
    )
    ShoppingMall: float | None = Field(
        default = None,
        ge = 0
    )
    Spa: float | None = Field(
        default = None,
        ge = 0
    )
    VRDeck: float | None = Field(
        default = None,
        ge = 0
    )


class PredictionResponse(BaseModel):
    transported: bool 
    probability: float = Field(ge=0, le=1)
    model_version: str
    request_id: str
    
