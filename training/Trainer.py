from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# main class
class Trainer:
    # class attributes
    ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT / "data" / "train" / "training_data.csv"
    MODEL_PATH = ROOT / "model" / "model.pkl"
    EXPERIMENT_NAME = "emo-detector"
    N_NEIGHBORS = 3
    FEATURES = ["age", "cuts"]
    TARGET = "emo"

    # private methods
    @staticmethod
    def _load_data() -> tuple:
        df = pd.read_csv(Trainer.DATA_PATH)
        X = df[Trainer.FEATURES].values
        y = df[Trainer.TARGET].values
        return X, y

    @staticmethod
    def _train() -> None:
        X, y = Trainer._load_data()

        mlflow.set_experiment(Trainer.EXPERIMENT_NAME)

        with mlflow.start_run():
            model = KNeighborsClassifier(n_neighbors=Trainer.N_NEIGHBORS)
            model.fit(X, y)

            mlflow.log_param("n_neighbors", Trainer.N_NEIGHBORS)

            Trainer.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, Trainer.MODEL_PATH)

            mlflow.sklearn.log_model(model, "knn_model")

        print("Model trained successfully.")

    # public methods
    @staticmethod
    def main() -> None:
        Trainer._train()


if __name__ == "__main__":
    Trainer.main()
