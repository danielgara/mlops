from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# Dataset
df = pd.DataFrame({
    "age": [16, 17, 18, 25, 30, 35],
    "cuts": [5, 6, 7, 0, 1, 0],
    "emo": [1, 1, 1, 0, 0, 0]
})

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
