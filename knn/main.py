import logging
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .eval import evaluate, save_reports
from .train import tune
from .utils import (
    ACTIVITY_LABELS,
    ModelBundle,
    load_groups,
    load_split,
    save_model,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "knn.joblib"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    search = tune(
        X_train,
        y_train,
        groups=load_groups("train"),
    )

    top = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        .head(5)[["params", "mean_test_score", "std_test_score"]]
    )
    logger.info(
        "Top CV configurations:\n%s",
        top.to_string(index=False),
    )

    best_model = search.best_estimator_
    if not isinstance(best_model, Pipeline):
        raise TypeError(
            f"Expected a Pipeline, got {type(best_model).__name__}.",
        )

    test_metrics = evaluate(
        best_model,
        X_test,
        y_test,
    )

    save_reports(
        best_model,
        search.cv_results_,
        test_metrics,
        X_test,
        y_test,
        REPORTS_DIR,
    )

    bundle = ModelBundle(
        model=best_model,
        feature_names=list(X_train.columns),
        labels=ACTIVITY_LABELS,
        params=dict(search.best_params_),
        cv_score=float(search.best_score_),
        test_metrics=test_metrics,
        sklearn_version=sklearn.__version__,
    )

    save_model(
        bundle,
        MODEL_PATH,
    )

    logger.info(
        "Model saved to %s",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
