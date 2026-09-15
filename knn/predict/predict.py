import numpy as np

from sklearn.neighbors import KNeighborsClassifier


class ModelPrediction:
    def __init__(self, knn: KNeighborsClassifier) -> None:
        self.knn = knn

    def predict(self, X_new) -> np.ndarray:
        return self.knn.predict(X_new)
