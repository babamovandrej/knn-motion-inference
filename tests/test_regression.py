import pytest

from knn.eval import evaluate
from knn.train import build_pipeline
from knn.utils import Dataset

MIN_TEST_ACCURACY = 0.95


@pytest.mark.slow
def test_full_model_accuracy(train_split: Dataset, test_split: Dataset) -> None:
    X_train, y_train = train_split
    X_test, y_test = test_split
    model = build_pipeline().set_params(knn__n_neighbors=9).fit(X_train, y_train)

    assert evaluate(model, X_test, y_test)["accuracy"] >= MIN_TEST_ACCURACY
