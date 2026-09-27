import numpy as np
import pandas as pd

from ..utils import ModelBundle


def _align_features(
    bundle: ModelBundle,
    X_new: pd.DataFrame,
) -> pd.DataFrame:
    missing = [c for c in bundle.feature_names if c not in X_new.columns]
    expected = set(bundle.feature_names)
    extra = [str(c) for c in X_new.columns if c not in expected]

    if missing or extra:
        raise ValueError(
            f"Feature mismatch: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]})."
        )

    return X_new.reindex(columns=bundle.feature_names)


def predict(
    bundle: ModelBundle,
    X_new: pd.DataFrame,
) -> list[str]:
    y_prediction = np.asarray(
        bundle.model.predict(_align_features(bundle, X_new)),
        dtype=np.int64,
    )

    return [bundle.labels[int(i)] for i in y_prediction]


def predict_proba(
    bundle: ModelBundle,
    X_new: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = np.asarray(
        bundle.model.predict_proba(_align_features(bundle, X_new)),
        dtype=np.float64,
    )

    return pd.DataFrame(
        probabilities,
        columns=[bundle.labels[int(i)] for i in bundle.model.classes_],
        index=X_new.index,
    )
