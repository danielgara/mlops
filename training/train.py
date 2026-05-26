from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "training.csv"
df = pd.read_csv(DATA_PATH)

# Features and target
X = df[["age", "cuts"]].values
y = df["emo"].values

# MLflow tracking
mlflow.set_experiment("emo-detector")

with mlflow.start_run():

    n_neighbors = 3
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )
    model.fit(X, y)

    # Log parameters
    mlflow.log_param(
        "n_neighbors",
        n_neighbors
    )

    # Save model
    joblib.dump(
        model,
        "model/model.pkl"
    )

    # Register model in MLflow
    mlflow.sklearn.log_model(
        model,
        "knn_model"
    )

    print("Model trained successfully.")
