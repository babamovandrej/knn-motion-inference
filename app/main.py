import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api import router
from .services import PredictionService, load_bundle, resolve_model_path

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    path = resolve_model_path()

    try:
        app.state.prediction_service = PredictionService.from_bundle(
            load_bundle(path),
        )
        logger.info(
            "Model loaded from %s",
            path,
        )
    except FileNotFoundError:
        app.state.prediction_service = None
        logger.warning(
            "No model at %s; /predict returns 503 until it is trained.",
            path,
        )

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="KNN Motion Inference",
        summary="Human activity recognition from smartphone sensor features.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
