import mlflow
import dagshub
import json
import logging
from pathlib import Path
from mlflow import MlflowClient

STAGING_ALIAS = "staging"

# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# avoid adding duplicate handlers if the module is imported/reloaded
if not logger.handlers:
    # console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # create a formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)

# initialize dagshub
try:
    # mlflow=True already sets the tracking uri to the DagsHub MLflow server
    dagshub.init(repo_owner='Ankitkumar1141', repo_name='Churn-modelling', mlflow=True)
    logger.info('DagsHub initialized successfully.')
except Exception as e:
    # fail loudly instead of silently logging to a local ./mlruns folder
    logger.error(f'[DagsHub Error] {e}')
    raise


def load_model_information(file_path):
    try:
        with open(file_path) as f:
            run_info = json.load(f)
    except FileNotFoundError:
        logger.error(f"Run information file not found: {file_path}")
        raise

    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # run information file path
    run_info_path = root_path / "run_information.json"

    # load the run information
    run_info = load_model_information(run_info_path)

    # get the model name and the uri of the model logged by evaluation.py
    model_name = run_info["model_name"]
    model_uri = run_info["artifact_path"]

    # register the model
    model_version = mlflow.register_model(model_uri=model_uri,
                                          name=model_name)

    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    # point the staging alias to this model version
    # (stages are deprecated in MLflow 3, aliases replace them)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=STAGING_ALIAS,
        version=registered_model_version
    )

    logger.info(f"Model version {registered_model_version} assigned to '{STAGING_ALIAS}' alias")