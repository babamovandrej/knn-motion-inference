from pathlib import Path

import pandas as pd

from .eval.eval import ModelEvaluation
from .train.train import ModelTraining


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"

    X_train = pd.read_csv(data_dir / "train" / "X_train.csv")
    y_train = pd.read_csv(data_dir / "train" / "y_train.csv")["activity_id"]

    X_test = pd.read_csv(data_dir / "test" / "X_test.csv")
    y_test = pd.read_csv(data_dir / "test" / "y_test.csv")["activity_id"]

    best_score = 0.0
    best_training = None

    for n_neighbours in range(1, 16, 2):
        training = ModelTraining(
            X_train=X_train,
            y_train=y_train,
            n_neighbours=n_neighbours,
        )

        training.train()

        evaluation = ModelEvaluation(
            knn=training.knn,
            X_test=X_test,
            y_test=y_test,
        )

        score = evaluation.evaluate()

        if score > best_score:
            best_score = score
            best_training = training

    if best_training is None:
        raise RuntimeError("No model was trained.")

    print(f"score={best_score:.2f}")

    best_training.save()


if __name__ == "__main__":
    main()
