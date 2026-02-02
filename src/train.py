import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix
import yaml
import os
import mlflow
import mlflow.sklearn

from src.features import get_features_and_target


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ]
    )


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def main():
    config = load_config("configs/config.yaml")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("covid-mortality")

    df = pd.read_csv(config["paths"]["processed_data"])
    X, y = get_features_and_target(df, config["target"]["label_column"])

    numeric_features = X.columns.tolist()
    preprocessor = build_preprocessor(numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    models = {
        "baseline_logreg": LogisticRegression(max_iter=1000),
        "strong_rf": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    os.makedirs(config["paths"]["model_dir"], exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", model),
            ])

            pipe.fit(X_train, y_train)
            metrics = evaluate_model(pipe, X_test, y_test)

            mlflow.log_param("model_type", name)

            for k, v in metrics.items():
                if k != "confusion_matrix":
                    mlflow.log_metric(k, v)

            mlflow.log_artifact("configs/config.yaml")

            model_path = os.path.join(config["paths"]["model_dir"], f"{name}.joblib")
            joblib.dump(pipe, model_path)

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            print(f"\nModel: {name}")
            print("Metrics:", metrics)
            print("Saved to:", model_path)


if __name__ == "__main__":
    main()
