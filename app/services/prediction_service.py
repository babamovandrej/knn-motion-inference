from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from knn.predict import predict_proba
from knn.utils import ModelBundle

from .feature_validator import FeatureValidator


@dataclass(frozen=True)
class Prediction:
    activity: str
    confidence: float
    probabilities: dict[str, float]


class PredictionService:
    def __init__(
        self,
        bundle: ModelBundle,
        validator: FeatureValidator,
    ) -> None:
        self._bundle = bundle
        self._validator = validator

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
    ) -> "PredictionService":
        return cls(
            bundle,
            FeatureValidator(bundle.feature_names),
        )

    def predict(
        self,
        samples: Sequence[Mapping[str, float]],
    ) -> list[Prediction]:
        self._validator.validate(samples)

        probabilities = predict_proba(
            self._bundle,
            pd.DataFrame(
                list(samples),
                columns=self._bundle.feature_names,
            ),
        )

        return [
            Prediction(
                activity=str(row.idxmax()),
                confidence=float(row.max()),
                probabilities={
                    str(activity): float(value) for activity, value in row.items()
                },
            )
            for _, row in probabilities.iterrows()
        ]
