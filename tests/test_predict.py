import pytest
import sklearn

from knn.predict import predict, predict_proba
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


def test_predict_returns_activity_names(
    bundle: ModelBundle, small_train: Dataset
) -> None:
    X, _ = small_train
    names = predict(bundle, X.head(5))

    assert len(names) == 5
    assert set(names) <= set(ACTIVITY_LABELS.values())


def test_predict_accepts_shuffled_columns(
    bundle: ModelBundle, small_train: Dataset
) -> None:
    X, _ = small_train
    shuffled = X.head(5)[X.columns[::-1]]

    assert predict(bundle, shuffled) == predict(bundle, X.head(5))


def test_predict_rejects_missing_columns(
    bundle: ModelBundle, small_train: Dataset
) -> None:
    X, _ = small_train

    with pytest.raises(ValueError, match="1 missing"):
        predict(bundle, X.head(5).drop(columns=X.columns[0]))


def test_predict_rejects_extra_columns(
    bundle: ModelBundle, small_train: Dataset
) -> None:
    X, _ = small_train

    with pytest.raises(ValueError, match="1 unexpected"):
        predict(bundle, X.head(5).assign(bogus=0.0))


def test_predict_proba_rows_sum_to_one(
    bundle: ModelBundle, small_train: Dataset
) -> None:
    X, _ = small_train
    probabilities = predict_proba(bundle, X.head(5))

    assert list(probabilities.columns) == list(ACTIVITY_LABELS.values())
    assert probabilities.sum(axis=1).round(6).eq(1.0).all()
