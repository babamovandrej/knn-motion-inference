from collections.abc import Iterator
from pathlib import Path

import pytest
import sklearn
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import MAX_SAMPLES
from knn.train import build_pipeline
from knn.utils import ACTIVITY_LABELS, Dataset, ModelBundle, save_model


@pytest.fixture
def client(
    small_train: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    X, y = small_train
    path = tmp_path / "knn.joblib"
    save_model(
        ModelBundle(
            model=build_pipeline().fit(X, y),
            feature_names=list(X.columns),
            labels=ACTIVITY_LABELS,
            sklearn_version=sklearn.__version__,
        ),
        path,
    )
    monkeypatch.setenv("KNN_MODEL_PATH", str(path))

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def samples(small_train: Dataset) -> list[dict[str, float]]:
    X, _ = small_train
    return [
        {str(key): float(value) for key, value in row.items()}
        for row in X.head(3).to_dict(orient="records")
    ]


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_activity_per_sample(
    client: TestClient,
    samples: list[dict[str, float]],
) -> None:
    response = client.post("/predict", json={"samples": samples})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3

    for prediction in predictions:
        assert prediction["activity"] in ACTIVITY_LABELS.values()
        assert prediction["confidence"] == max(prediction["probabilities"].values())
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_predict_rejects_missing_feature(
    client: TestClient,
    samples: list[dict[str, float]],
) -> None:
    samples[1].pop(next(iter(samples[1])))

    response = client.post("/predict", json={"samples": samples})

    assert response.status_code == 422
    assert "sample 1: 1 missing" in response.json()["detail"]


def test_predict_rejects_unexpected_feature(
    client: TestClient,
    samples: list[dict[str, float]],
) -> None:
    samples[0]["bogus"] = 0.0

    response = client.post("/predict", json={"samples": samples})

    assert response.status_code == 422
    assert response.json()["detail"] == "sample 0: 1 unexpected (e.g. ['bogus'])"


@pytest.mark.parametrize(
    "payload",
    [
        {"samples": []},
        {"samples": [{"tBodyAcc-mean()-X": "fast"}]},
        {"samples": [{}] * (MAX_SAMPLES + 1)},
        {},
    ],
)
def test_predict_rejects_invalid_payload(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_model_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNN_MODEL_PATH", str(tmp_path / "missing.joblib"))

    with TestClient(create_app()) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.post("/predict", json={"samples": [{}]}).status_code == 503


@pytest.mark.parametrize("path", ["/model", "/model/features"])
def test_model_details_are_not_exposed(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404
