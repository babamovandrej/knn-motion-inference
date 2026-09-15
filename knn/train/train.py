import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier


class ModelTraining:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_neighbours: int,
    ) -> None:
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbours,
        )
        self.X_train = X_train
        self.y_train = y_train

    def train(self) -> None:
        self.knn.fit(self.X_train, self.y_train)

    def save(self) -> None:
        joblib.dump(
            self.knn,
            "./knn/model/knn.joblib",
        )
