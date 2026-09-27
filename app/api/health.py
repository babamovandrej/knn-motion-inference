from fastapi import APIRouter, Request

from ..schemas import HealthResponse
from .dependencies import find_prediction_service

router = APIRouter(
    tags=["system"],
)


@router.get("/health")
def health(
    request: Request,
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=find_prediction_service(request) is not None,
    )
