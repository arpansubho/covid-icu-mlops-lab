import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "covid-mortality"
MODEL_NAME = "covid_mortality_model"
METRIC = "roc_auc"

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )

    best_run = runs[0]
    run_id = best_run.info.run_id
    print("Best run:", run_id, "ROC-AUC:", best_run.data.metrics[METRIC])

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(model_uri, MODEL_NAME)
    print("Registered model version:", registered_model.version)

    # Move to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
    )

    print(f"Model {MODEL_NAME} version {registered_model.version} moved to Staging")

if __name__ == "__main__":
    main()
