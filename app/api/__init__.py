from fastapi import APIRouter

from .health import router as health_router
from .prediction import router as prediction_router

router = APIRouter()
router.include_router(health_router)
router.include_router(prediction_router)

__all__ = ["router"]
