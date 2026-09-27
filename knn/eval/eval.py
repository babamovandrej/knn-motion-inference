import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from ..utils import ACTIVITY_LABELS, EvaluationMetrics

logger = logging.getLogger(__name__)


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: "pd.Series[int]",
) -> EvaluationMetrics:
    y_prediction = np.asarray(
        model.predict(X_test),
        dtype=np.int64,
    )
    label_ids = sorted(ACTIVITY_LABELS)
    target_names = [ACTIVITY_LABELS[i] for i in label_ids]

    report = classification_report(
        y_test,
        y_prediction,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
    )
    if not isinstance(report, dict):
        raise TypeError("classification_report did not return a dict.")

    metrics: EvaluationMetrics = {
        "accuracy": float(accuracy_score(y_test, y_prediction)),
        "f1_macro": float(f1_score(y_test, y_prediction, average="macro")),
        "report": report,
        "confusion_matrix": [
            [int(count) for count in row]
            for row in confusion_matrix(
                y_test,
                y_prediction,
                labels=label_ids,
            )
        ],
    }

    logger.info(
        "Test accuracy=%.4f | macro-F1=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
    )
    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test,
            y_prediction,
            labels=label_ids,
            target_names=target_names,
            digits=4,
        ),
    )
    logger.info(
        "Confusion matrix (rows=true, cols=predicted, order=%s):\n%s",
        target_names,
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=target_names,
            columns=target_names,
        ).to_string(),
    )

    return metrics
