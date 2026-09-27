from pathlib import Path

import numpy as np
import sklearn

from knn.train import build_pipeline
from knn.utils import ACTIVITY_LABELS, Dataset, ModelBundle, load_model, save_model


def test_pipeline_predicts_known_labels(small_train: Dataset) -> None:
    X, y = small_train
    model = build_pipeline().fit(X, y)

    assert set(model.predict(X)) <= set(ACTIVITY_LABELS)


def test_bundle_round_trip(small_train: Dataset, tmp_path: Path) -> None:
    X, y = small_train
    bundle = ModelBundle(
        model=build_pipeline().fit(X, y),
        feature_names=list(X.columns),
        labels=ACTIVITY_LABELS,
        sklearn_version=sklearn.__version__,
    )
    path = tmp_path / "nested" / "model.joblib"

    save_model(bundle, path)
    loaded = load_model(path)

    assert loaded.feature_names == bundle.feature_names
    assert np.array_equal(loaded.model.predict(X), bundle.model.predict(X))
