import logging
from pathlib import Path

import joblib
import sklearn
from sklearn.utils.validation import check_is_fitted

from .bundle import ModelBundle

logger = logging.getLogger(__name__)


def load_model(
    path: Path,
) -> ModelBundle:
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, ModelBundle):
        raise TypeError(
            f"Loaded object is {type(bundle).__name__}, expected ModelBundle. "
            "Retrain with `python -m knn.main`."
        )

    check_is_fitted(bundle.model)

    if bundle.sklearn_version != sklearn.__version__:
        logger.warning(
            "Model was saved with scikit-learn %s but %s is installed.",
            bundle.sklearn_version,
            sklearn.__version__,
        )

    return bundle
