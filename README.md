# knn-motion-inference

[![CI](https://github.com/babamovandrej/knn-motion-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/babamovandrej/knn-motion-inference/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Human activity recognition from smartphone sensor data, taken from raw features to a tested, containerised prediction API.

A K-Nearest Neighbours model reads a window of accelerometer and gyroscope features and predicts one of six activities: walking, walking upstairs, walking downstairs, sitting, standing, or laying. On 9 people it never saw during training, it is right **96.5%** of the time.

| Metric | Value |
|---|---|
| Test accuracy (2,947 windows, 9 unseen people) | **96.50%** |
| Test macro-F1 | **96.55%** |
| Cross-validated macro-F1 (folds grouped by person) | 95.28% |
| Saved model size | 0.85 MB |

## What this project demonstrates

The dataset is a well-known benchmark, and 96.5% is in line with what strong models reach on it (see [How it compares](#how-it-compares)). The point of the project is how that number is produced, checked and served:

- **An evaluation that holds up.** Hyperparameters are chosen by cross-validation with folds grouped by person, and the test set is scored once. Shuffled cross-validation would have reported 98.3%; the grouped estimate of 95.3% is the honest one.
- **A model improvement with a measured reason.** Adding an LDA projection lifts KNN from 90.6% to 96.5%, and the README shows why the obvious alternatives (scaling, PCA) make it worse.
- **Weaknesses shown, not hidden.** 77 of the 103 test errors are sitting vs standing, and the charts show why.
- **Production habits.** A FastAPI service with input validation and clear errors, a two-stage Docker image with a health check, strict type checking, 31 tests including an accuracy regression guard, and CI on every push.

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [How it compares](#how-it-compares)
- [Quick start](#quick-start)
- [Using the model](#using-the-model)
- [Prediction API](#prediction-api)
- [Docker](#docker)
- [Project structure](#project-structure)
- [Development](#development)
- [Dataset](#dataset)
- [Roadmap](#roadmap)
- [License](#license)

## How it works

```
561 sensor features ──► LDA (5 dimensions) ──► KNN (k=25) ──► activity
```

1. **Linear Discriminant Analysis (LDA).** The 561 input features are heavily correlated. Plain KNN on all of them measures distance in a space where most directions carry no information about the activity. LDA is trained on the labels and projects every window onto the 5 directions that best separate the six activities.
2. **K-Nearest Neighbours.** A new window is classified by a vote of its 25 closest training windows in that 5-dimensional space.

The LDA step is what makes the model work. On the same test set:

| Model | Test accuracy |
|---|---|
| KNN on the raw 561 features | 90.6% |
| StandardScaler + KNN | 88.9% |
| PCA (95% variance) + KNN | 88.3% |
| **LDA + KNN** (this project) | **96.5%** |

Scaling and PCA both make things worse, because the features are already normalised to [-1, 1] and PCA keeps the directions with the most variance rather than those that best separate the activities.

### Honest model selection

Hyperparameters are chosen with cross-validation on the **training set only**. The test set is scored exactly once, after the model has been chosen.

The data comes in overlapping windows (50% overlap), many from the same person. With shuffled cross-validation, near-duplicate windows from one person land on both sides of a split, and the score is inflated. The pipeline uses `GroupKFold` with the dataset's subject ids, so every fold holds out whole people, exactly like the test set does:

| Cross-validation | Macro-F1 |
|---|---|
| Shuffled 5-fold | 98.3% (inflated) |
| **Grouped by person, 5-fold** (used) | **95.3%** |
| Actual result on 9 unseen people | 96.5% |

The search covers 100 configurations, scored by macro-F1:

| Parameter | Values |
|---|---|
| LDA solver | `svd`; or `eigen` with shrinkage `auto`, 0.05, 0.1, 0.2 |
| Number of neighbours (k) | 3, 5, 9, 15, 25 |
| Neighbour weights | `uniform`, `distance` |
| Distance | Manhattan (p=1), Euclidean (p=2) |

The chosen configuration is `svd` LDA with k=25, uniform weights and Euclidean distance.

## Results

Every training run writes these charts to [`knn/reports/`](knn/reports/).

### Confusion matrix

![Confusion matrix on the test set](knn/reports/confusion_matrix.png)

Rows are the true activity and columns are the prediction. Moving and stationary activities are almost never confused (a single sitting window was predicted as walking upstairs), and LAYING is 100% correct. Of the 103 errors, 77 are between **SITTING and STANDING**: 59 sitting windows predicted as standing, and 18 the other way round.

### Per-activity scores

![Per-activity precision, recall and F1](knn/reports/class_metrics.png)

- **Precision:** when the model predicts an activity, how often it is right.
- **Recall:** how many windows of that activity it finds.

The four moving and lying activities score above 96% on every measure. The weak spots are SITTING recall (88%) and STANDING precision (90%), which are two sides of the same confusion.

### Choice of k

![Cross-validated macro-F1 by number of neighbours](knn/reports/cv_neighbours.png)

Cross-validated macro-F1 stays within 0.4 points of the best score for every k from 3 to 25, much less than the variation between folds (the shaded band). The exact k barely matters; the LDA projection does the work.

### The space KNN searches

![Test windows in the first two LDA dimensions](knn/reports/lda_projection.png)

Each panel highlights one activity in the first two LDA dimensions. There are three clear groups: the walking activities (left), sitting and standing (bottom), and laying (top right). SITTING and STANDING occupy almost the same region, which explains the confusion matrix. With the phone at the waist, both positions are still and upright, so the sensor signals look alike.

## How it compares

For context, other standard classifiers trained on the same 561 features and scored on the same test set:

| Model | Test accuracy |
|---|---|
| Linear SVM | 96.7% |
| **LDA + KNN** (this project) | **96.5%** |
| RBF SVM | 96.2% |
| Logistic regression | 96.1% |

LDA + KNN is competitive with the strongest linear model while staying simple to explain: every prediction is a vote of 25 real training windows. The remaining gap between models is smaller than the sitting vs standing confusion they all share.

## Quick start

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:babamovandrej/knn-motion-inference.git
cd knn-motion-inference
uv sync
uv run python -m knn.main
```

The run takes 1–2 minutes and:

1. loads the training and test data from `knn/data/`,
2. tunes the LDA → KNN pipeline with cross-validation grouped by person,
3. evaluates the chosen model once on the test set and logs the per-activity report and confusion matrix,
4. writes the charts to `knn/reports/`,
5. saves the model to `knn/model/knn.joblib`.

Run it with `-m` from the project root. `python knn/main.py` fails because the package uses relative imports.

The trained model is not committed to git (`knn/model/*.joblib` is ignored), so run the pipeline once after cloning.

## Using the model

```python
import pandas as pd

from knn.predict import predict, predict_proba
from knn.utils import load_model

bundle = load_model("knn/model/knn.joblib")

X_new = pd.read_csv("knn/data/test/X_test.csv").head(3)

predict(bundle, X_new)
predict_proba(bundle, X_new)
```

`X_new` must contain the 561 feature columns by name, in any order. A missing or unexpected column raises a `ValueError` that names the columns involved.

`predict_proba` returns the share of neighbour votes per activity, which can serve as a rough confidence.

`load_model` returns a `ModelBundle`, which keeps everything needed to serve and audit the model:

| Field | Contents |
|---|---|
| `model` | The fitted LDA → KNN `Pipeline` |
| `feature_names` | The 561 feature names, in training order |
| `labels` | Activity id → name, e.g. `1 → "WALKING"` |
| `params` | The hyperparameters chosen by cross-validation |
| `cv_score` | Cross-validated macro-F1 |
| `test_metrics` | Test accuracy, macro-F1, per-class report and confusion matrix |
| `sklearn_version` | The scikit-learn version used for training; a warning is logged if the loading version differs |

## Prediction API

A FastAPI service in `app/` serves the trained model over HTTP. Train the model first, then start the server from the project root:

```bash
uv run python -m knn.main
uv run uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`, with interactive documentation at `http://127.0.0.1:8000/docs`. The model is loaded once at startup, from `knn/model/knn.joblib` by default or from the path in the `KNN_MODEL_PATH` environment variable.

| Method | Path | Returns |
|---|---|---|
| `POST` | `/predict` | The predicted activity, confidence and per-activity probabilities for each sample |
| `GET` | `/health` | Service status and whether the model is loaded |

The model is used only to serve predictions; its settings, scores and feature list are not exposed over HTTP.

### Making a prediction

The request body holds a list of 1 to 1,000 samples. Each sample maps all 561 feature names to finite numbers, in any order. The feature names are the column headers of `knn/data/test/X_test.csv`.

```python
import httpx
import pandas as pd

samples = pd.read_csv("knn/data/test/X_test.csv").head(2).to_dict(orient="records")

response = httpx.post("http://127.0.0.1:8000/predict", json={"samples": samples})
response.json()
```

Or with curl, after writing a request file from the test data:

```bash
uv run python -c "import json, pandas as pd; json.dump({'samples': pd.read_csv('knn/data/test/X_test.csv').head(2).to_dict(orient='records')}, open('request.json', 'w'))"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @request.json
```

Response:

```json
{
  "predictions": [
    {
      "activity": "STANDING",
      "confidence": 1.0,
      "probabilities": {
        "WALKING": 0.0,
        "WALKING_UPSTAIRS": 0.0,
        "WALKING_DOWNSTAIRS": 0.0,
        "SITTING": 0.0,
        "STANDING": 1.0,
        "LAYING": 0.0
      }
    }
  ]
}
```

The response has one entry per sample, in request order; the example shows the first. `confidence` is the probability of the predicted activity: the share of the 25 nearest neighbours that voted for it. A low confidence marks an uncertain prediction; most of the model's mistakes are sitting vs standing calls with split votes.

### Errors

| Status | When |
|---|---|
| `422` | A sample has missing or unexpected features, a value is not a finite number, or the list is empty or longer than 1,000. For feature errors, `detail` names the sample and the features involved, e.g. `sample 0: 1 unexpected (e.g. ['bogus'])`. |
| `503` | No trained model was found at startup. Run `uv run python -m knn.main` and restart the server. |

## Docker

The image trains the model during the build, so it runs with no local setup:

```bash
docker build -t knn-motion-inference .
docker run -p 8000:8000 knn-motion-inference
```

The API is then available at `http://127.0.0.1:8000`, exactly as above.

The build has two stages. The first installs the dependencies from `uv.lock` and trains the model; the final image copies only the Python environment, the prediction code and the trained model, leaving out the training data and training code. The API runs as a non-root user, and a Docker health check calls `/health`.

## Project structure

```
knn-motion-inference/
├── knn/
│   ├── main.py
│   ├── data/
│   ├── train/train.py
│   ├── eval/
│   │   ├── eval.py
│   │   └── charts.py
│   ├── predict/predict.py
│   ├── utils/
│   │   ├── data.py
│   │   ├── bundle.py
│   │   ├── save.py
│   │   └── load.py
│   ├── model/
│   └── reports/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── health.py
│   │   └── prediction.py
│   ├── schemas/
│   │   ├── health.py
│   │   └── prediction.py
│   └── services/
│       ├── model_loader.py
│       ├── feature_validator.py
│       └── prediction_service.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
└── pyproject.toml
```

## Development

```bash
uv sync
uv run pytest -q
uv run pytest -q -m "not slow"
uv run mypy knn app tests
uv run ruff check .
uv run ruff format .
```

The tests cover:

- **Data:** subject ids line up with the feature rows, and no volunteer appears in both the training and test sets.
- **Pipeline:** training and a save → load round trip.
- **Predictions:** activity names, column reordering, and missing or extra columns.
- **Charts:** all four report charts are written.
- **API:** both endpoints, payload validation, the 503 response when no model is loaded, and that model details are not exposed.
- **Services:** model path resolution and loading, feature validation, and prediction, each tested on its own.
- **Regression guard:** a full train and test run that fails if accuracy drops below 95%. It is marked `slow` and skipped by `-m "not slow"`.

Type checking runs mypy in strict mode, with `pandas-stubs` for pandas. scikit-learn and joblib ship no type information, so values returned by those libraries are converted to concrete types where they enter the code.

GitHub Actions runs on every push to `main` and on every pull request. One job runs ruff, mypy, the full test suite and a complete training run; a second builds the Docker image, starts it and checks that `/health` responds.

## Dataset

The data is the [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset. It has 30 volunteers aged 19–48, each wearing a Samsung Galaxy S II at the waist, and the accelerometer and gyroscope were sampled at 50 Hz.

- Signals are cut into 2.56-second windows with 50% overlap.
- Each window has 561 time- and frequency-domain features, normalised to [-1, 1].
- The split is by person: 21 volunteers for training and 9 for testing.

| Split | Windows | People | Walking | Upstairs | Downstairs | Sitting | Standing | Laying |
|---|---|---|---|---|---|---|---|---|
| Train | 7,352 | 21 | 1,226 | 1,073 | 986 | 1,286 | 1,374 | 1,407 |
| Test | 2,947 | 9 | 496 | 471 | 420 | 491 | 532 | 537 |

The files in `knn/data/` are the original UCI data in its original row order, converted to CSV:

| File | Contents |
|---|---|
| `X_train.csv`, `X_test.csv` | The 561 features per window |
| `y_train.csv`, `y_test.csv` | The activity id per window |
| `subjects_train.csv`, `subjects_test.csv` | The id of the volunteer who recorded each window |

The training subject ids group the cross-validation folds by person. The test subject ids confirm that no volunteer appears in both splits.

## Roadmap

- **Sitting vs standing.** Try features that capture posture, such as the gravity-angle features, or a specialised second-stage classifier for the two stationary upright activities, to reduce the main remaining error.

## License

[MIT](LICENSE) © 2026 Andrej Babamov
