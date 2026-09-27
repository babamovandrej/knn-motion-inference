from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from ..services import PredictionService


def find_prediction_service(
    request: Request,
) -> PredictionService | None:
    service = getattr(request.app.state, "prediction_service", None)

    return service if isinstance(service, PredictionService) else None


def get_prediction_service(
    request: Request,
) -> PredictionService:
    service = find_prediction_service(request)

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train it with `python -m knn.main`.",
        )

    return service


PredictionServiceDependency = Annotated[
    PredictionService,
    Depends(get_prediction_service),
]
