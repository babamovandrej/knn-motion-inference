import logging

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..utils import ParamValue

logger = logging.getLogger(__name__)

N_SPLITS = 5

PARAM_GRID: list[dict[str, list[ParamValue]]] = [
    {
        "lda__solver": ["svd"],
        "knn__n_neighbors": [3, 5, 9, 15, 25],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    },
    {
        "lda__solver": ["eigen"],
        "lda__shrinkage": ["auto", 0.05, 0.1, 0.2],
        "knn__n_neighbors": [3, 5, 9, 15, 25],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    },
]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("lda", LinearDiscriminantAnalysis()),
            ("knn", KNeighborsClassifier()),
        ]
    )


def tune(
    X_train: pd.DataFrame,
    y_train: "pd.Series[int]",
    groups: "pd.Series[int] | None" = None,
    n_jobs: int = -1,
) -> GridSearchCV:
    cv = GroupKFold(n_splits=N_SPLITS) if groups is not None else KFold(N_SPLITS)

    search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    logger.info(
        "CV (%s) best macro-F1=%.4f | params=%s",
        type(cv).__name__,
        search.best_score_,
        search.best_params_,
    )

    return search
