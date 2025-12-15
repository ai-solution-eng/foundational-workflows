from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from utils import refresh_mlflow_token
import os
import logging
import argparse
import bentoml


def pick_best(
    experiment_name="medmnist_experiment",
    metric="best_val_acc",
    top=1,
    local_mlflow=True,
    registered_model_name="medmnist_cnn",
    promote_to=None,
):
    if not local_mlflow:
        refresh_mlflow_token()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    # Get experiment id
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    exp_id = exp.experiment_id

    runs = client.search_runs(
        [exp_id],
        f"attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=top,
    )

    if not runs:
        raise ValueError("No runs found")

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(metric)

    # Find model version(s) produced by that run in registry
    model_versions = []
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        if getattr(mv, "run_id", None) == run_id:
            model_versions.append(mv)

    if not model_versions:
        logging.info(
            "No registered model linked to this run. You can load artifacts directly or re-log the model."
        )
    else:
        for mv in model_versions:
            logging.info(
                f"Found registered version: {mv.name} v{mv.version} stage={mv.current_stage}"
            )
            if promote_to:
                client.transition_model_version_stage(
                    name=mv.name,
                    version=mv.version,
                    stage=promote_to,
                    archive_existing_versions=False,
                )
                logging.info(f"Promoted {mv.name} v{mv.version} to {promote_to}")

    logging.info(f"Best run: run_id: {run_id} - metric: {best_metric:0.4}")

    # Provide model URI for loading via MLflow pyfunc / pytorch
    # If registered model exists, return models:/<name>/<stage or version>

    if promote_to:
        mlflow_model_uri = f"models:/{registered_model_name}/{promote_to}"
    elif model_versions:
        mlflow_model_uri = (
            f"models:/{registered_model_name}/{model_versions[0].version}"
        )
    else:
        mlflow_model_uri = f"runs:/{run_id}/{registered_model_name}"

    model = mlflow.pytorch.load_model(mlflow_model_uri)
    logging.info(f"Model URI to load: >> {mlflow_model_uri} <<")

    metadata = {
        f"{metric}": float(best_metric),
        "run_id": run_id,
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
    }

    signatures = {"predict": {"batchable": True}}

    # bentoml_uri = bentoml.pytorch.save_model(
    #     registered_model_name,
    #     model,
    #     signatures=signatures,
    #     metadata=metadata,
    #     external_modules=[models],
    # )

    bento_uri = bentoml.mlflow.import_model(registered_model_name, mlflow_model_uri)

    logging.info(
        f"Model exported with BentoML: {bento_uri.tag.name}:{bento_uri.tag.version}"
    )
    return mlflow_model_uri


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=1, help="consider top N models")
    parser.add_argument(
        "--experiment",
        default="medmnist_experiment_mlflow",
        help="must match that used during training",
    )
    parser.add_argument(
        "--metric",
        default="best_val_acc",
        help="must be a metric monitored and logged during training",
    )
    parser.add_argument(
        "--local-mlflow",
        action="store_true",
        help="If set, uses local mlflow server",
    )
    parser.add_argument(
        "--registered-model-name",
        default="medmnist_cnn",
        help="must match that used during training",
    )
    parser.add_argument("--promote_to", default=None, help="e.g. Staging or Production")
    args = parser.parse_args()

    # Set MLflow tracking URI
    if args.local_mlflow:
        assert os.path.exists("./.env.local"), FileNotFoundError(
            ".env.local file not found"
        )
        load_dotenv(dotenv_path=".env.local")
    else:
        assert os.path.exists("./.env.pcai"), FileNotFoundError(
            ".env.pcai file not found"
        )
        load_dotenv(dotenv_path=".env.pcai")

    # This is fundamental in order to track training with MLFlow.
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "NA")
    # This is fundamental in order to save artifacts into s3 bucket.
    MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "NA")

    assert MLFLOW_TRACKING_URI != "NA", ValueError(
        "MLFLOW_TRACKING_URI is not set in environment variables."
    )

    logging.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logging.info(f"Using MLFLOW_S3_ENDPOINT_URL: {MLFLOW_S3_ENDPOINT_URL}")

    logging.basicConfig(level=logging.INFO, force=True)
    pick_best(
        experiment_name=args.experiment,
        metric=args.metric,
        top=args.top,
        local_mlflow=args.local_mlflow,
        registered_model_name=args.registered_model_name,
        promote_to=args.promote_to,
    )
