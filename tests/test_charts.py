from pathlib import Path

from knn.eval import evaluate, save_reports
from knn.train import build_pipeline
from knn.utils import Dataset


def test_save_reports_writes_charts(small_train: Dataset, tmp_path: Path) -> None:
    X, y = small_train
    model = build_pipeline().set_params(knn__n_neighbors=5).fit(X, y)
    cv_results = {
        "param_knn__n_neighbors": [3, 5, 9],
        "mean_test_score": [0.90, 0.92, 0.91],
        "std_test_score": [0.02, 0.01, 0.02],
    }

    paths = save_reports(
        model,
        cv_results,
        evaluate(model, X, y),
        X,
        y,
        tmp_path / "reports",
    )

    assert [path.name for path in paths] == [
        "confusion_matrix.png",
        "class_metrics.png",
        "cv_neighbours.png",
        "lda_projection.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
