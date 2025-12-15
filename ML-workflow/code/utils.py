import os
import logging

SECRET_FILE_PATH = "/etc/secrets/ezua/.auth_token"

logging.basicConfig(level=logging.INFO, force=True)


def update_auth_token(verbose=True):
    with open(SECRET_FILE_PATH, "r") as file:
        token = file.read().strip()
    os.environ["AUTH_TOKEN"] = token
    if verbose:
        logging.info(f"Auth Token refreshed\n {os.environ.get('AUTH_TOKEN')}")
    return token


def refresh_mlflow_token(verbose=True):
    token = update_auth_token(verbose=verbose)
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    if verbose:
        logging.info("MLFlow Token refreshed")
        logging.info(f"{token}")


def search_runs(
    mlflow_tracking_uri: str, mlflow_s3_endpoint_url: str, experiment_id: str
):
    import mlflow

    refresh_mlflow_token()

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    # Search all runs in that experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Extract run IDs
    run_ids = runs["run_id"].tolist()

    return run_ids


def delete_run(run_id: str, mlflow_tracking_uri: str, mlflow_s3_endpoint_url: str):
    import mlflow

    refresh_mlflow_token()

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    try:
        mlflow.delete_run(run_id)
        print(f"Deleted {run_id}")
    except Exception as e:
        print(f"Failed to delete {run_id}: {e}")


def delete_all_runs(
    mlflow_tracking_uri: str, mlflow_s3_endpoint_url: str, experiment_id: str
):

    refresh_mlflow_token()

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    runs = search_runs(mlflow_tracking_uri, mlflow_s3_endpoint_url, experiment_id)
    for run_id in runs:
        try:
            delete_run(run_id, mlflow_tracking_uri, mlflow_s3_endpoint_url)
        except Exception as e:
            print(f"Failed to delete {run_id}: {e}")
