from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool