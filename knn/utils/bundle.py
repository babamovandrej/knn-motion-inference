from dataclasses import dataclass, field
from typing import Any, TypedDict

from sklearn.pipeline import Pipeline

type ParamValue = str | int | float | None


class EvaluationMetrics(TypedDict):
    accuracy: float
    f1_macro: float
    report: dict[str, Any]
    confusion_matrix: list[list[int]]


@dataclass
class ModelBundle:
    model: Pipeline
    feature_names: list[str]
    labels: dict[int, str]
    params: dict[str, ParamValue] = field(default_factory=dict)
    cv_score: float | None = None
    test_metrics: EvaluationMetrics | None = None
    sklearn_version: str = ""
