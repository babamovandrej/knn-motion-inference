import os
from pathlib import Path

from knn.utils import ModelBundle, load_model

MODEL_PATH_ENV = "KNN_MODEL_PATH"

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "knn" / "model" / "knn.joblib"
)


def resolve_model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)))


def load_bundle(
    path: Path,
) -> ModelBundle:
    return load_model(path)
