from pathlib import Path

import pytest
import sklearn

from app.services import (
    FeatureValidationError,
    FeatureValidator,
    PredictionService,
    load_bundle,
    resolve_model_path,
)
from knn.train import build_pipeline
from knn.utils import ACTIVITY_LABELS, Dataset, ModelBundle


@pytest.fixture(scope="module")
def bundle(small_train: Dataset) -> ModelBundle:
    X, y = small_train
    return ModelBundle(
        model=build_pipeline().fit(X, y),
        feature_names=list(X.columns),
        labels=ACTIVITY_LABELS,
        sklearn_version=sklearn.__version__,
    )


@pytest.fixture
def samples(small_train: Dataset) -> list[dict[str, float]]:
    X, _ = small_train
    return [
        {str(key): float(value) for key, value in row.items()}
        for row in X.head(3).to_dict(orient="records")
    ]


def test_validator_accepts_complete_samples() -> None:
    FeatureValidator(["a", "b"]).validate([{"a": 1.0, "b": 2.0}, {"b": 0.0, "a": 0.0}])


def test_validator_reports_each_invalid_sample() -> None:
    validator = FeatureValidator(["a", "b"])

    with pytest.raises(FeatureValidationError) as error:
        validator.validate(
            [{"a": 1.0, "b": 2.0}, {"a": 1.0}, {"a": 1.0, "b": 2.0, "c": 3.0}]
        )

    assert str(error.value) == (
        "sample 1: 1 missing (e.g. ['b']); sample 2: 1 unexpected (e.g. ['c'])"
    )


def test_validator_truncates_long_error_lists() -> None:
    with pytest.raises(FeatureValidationError, match="; and 2 more$"):
        FeatureValidator(["a"]).validate([{}] * 7)


def test_prediction_service_predicts_each_sample(
    bundle: ModelBundle,
    samples: list[dict[str, float]],
) -> None:
    predictions = PredictionService.from_bundle(bundle).predict(samples)

    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction.activity in ACTIVITY_LABELS.values()
        assert prediction.confidence == prediction.probabilities[prediction.activity]


def test_prediction_service_validates_before_predicting(
    bundle: ModelBundle,
    samples: list[dict[str, float]],
) -> None:
    samples[0]["bogus"] = 1.0

    with pytest.raises(FeatureValidationError):
        PredictionService.from_bundle(bundle).predict(samples)


def test_model_path_can_be_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNN_MODEL_PATH", str(tmp_path / "model.joblib"))

    assert resolve_model_path() == tmp_path / "model.joblib"


def test_default_model_path_points_at_trained_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KNN_MODEL_PATH", raising=False)

    assert resolve_model_path().parts[-3:] == ("knn", "model", "knn.joblib")


def test_load_bundle_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")
