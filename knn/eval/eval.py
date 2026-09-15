import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class ModelEvaluation:
    def __init__(
        self,
        knn: KNeighborsClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        self.knn = knn
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self) -> float:
        y_prediction = self.knn.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_prediction)

        print(f"Test set accuracy: {accuracy:.2f}")

        print("\nClassification report:")
        print(classification_report(self.y_test, y_prediction))

        print("\nConfusion matrix:")
        print(confusion_matrix(self.y_test, y_prediction))

        return float(accuracy)
