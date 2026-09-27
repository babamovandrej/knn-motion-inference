from pydantic import BaseModel, Field, FiniteFloat

MAX_SAMPLES = 1000


class PredictionRequest(BaseModel):
    samples: list[dict[str, FiniteFloat]] = Field(
        min_length=1,
        max_length=MAX_SAMPLES,
    )


class ActivityPrediction(BaseModel):
    activity: str
    confidence: float
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    predictions: list[ActivityPrediction]
