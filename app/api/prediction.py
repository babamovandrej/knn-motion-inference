from fastapi import APIRouter, HTTPException, status

from ..schemas import ActivityPrediction, PredictionRequest, PredictionResponse
from ..services import FeatureValidationError
from .dependencies import PredictionServiceDependency

router = APIRouter(
    tags=["prediction"],
)


@router.post("/predict")
def predict(
    body: PredictionRequest,
    service: PredictionServiceDependency,
) -> PredictionResponse:
    try:
        predictions = service.predict(body.samples)
    except FeatureValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return PredictionResponse(
        predictions=[
            ActivityPrediction(
                activity=prediction.activity,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
            )
            for prediction in predictions
        ],
    )
