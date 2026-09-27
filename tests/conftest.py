import pytest

from knn.utils import Dataset, load_split


@pytest.fixture(scope="session")
def train_split() -> Dataset:
    return load_split("train")


@pytest.fixture(scope="session")
def test_split() -> Dataset:
    return load_split("test")


@pytest.fixture(scope="session")
def small_train(train_split: Dataset) -> Dataset:
    X, y = train_split
    sample = y.groupby(y).sample(n=100, random_state=0).index
    return X.loc[sample], y.loc[sample]
