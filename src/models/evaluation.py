import pandas as pd
import joblib
import logging
import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
import dagshub
import json
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
)

TARGET = "churned"
MODEL_NAME = "Churn_Prediction_Model"
MODEL_FILE = "model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"

# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# avoid adding duplicate handlers if the module is imported/reloaded
if not logger.handlers:
    # console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # create a fomratter
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)

## initialize dagshub and mlfow
try:
    # mlflow=True already sets the tracking uri to the DagsHub MLflow server
    dagshub.init(repo_owner='Ankitkumar1141', repo_name='Churn-modelling', mlflow=True)
    logger.info('DagsHub initialized successfully.')
    mlflow.set_experiment('Model Pipeline')
except Exception as e:
    # fail loudly instead of silently logging to a local ./mlruns folder
    logger.error(f'[DagsHub Error] {e}')
    raise


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"The file to load does not exist: {data_path}")
        raise

    return data

def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model

def get_scores(model, X):
    # probability / score of the positive class, needed for a meaningful ROC-AUC
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    train_data_path = root_path / "data" / "processed" / "train.csv"
    test_data_path = root_path / "data" / "processed" / "test.csv"
    # model path
    model_path = root_path / "models" / MODEL_FILE
    preprocessor_path = root_path / "models" / PREPROCESSOR_FILE


    # load the training data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    # load the test data
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

    # split the train and test data
    X_train, y_train = make_X_and_y(train_data,TARGET)
    X_test, y_test = make_X_and_y(test_data,TARGET)
    logger.info("Data split completed")

    # load the model
    model = load_model(model_path)
    logger.info("Model Loaded successfully")


    # get the predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # get the scores (probabilities) for roc_auc
    y_train_proba = get_scores(model, X_train)
    y_test_proba = get_scores(model, X_test)
    logger.info("Predictions completed")

    # calculate the train and test roc_auc (from probabilities, not hard labels)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    logger.info("ROC-AUC score computed")

    # calculate the train and test recall
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    logger.info("Recall score computed")

    # calculate the f1 scores
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    logger.info("F1 score calculated")

    # calculate cross val scores (cross_validate supports multiple metrics)
    cv_scores = cross_validate(model,
                               X_train,
                               y_train,
                               cv=5,
                               scoring=["roc_auc", "recall", "f1"],
                               n_jobs=-1)
    logger.info("cross validation complete")

    # mean cross val scores (higher is better, so no negation)
    mean_cv_roc_auc = cv_scores["test_roc_auc"].mean()
    mean_cv_recall = cv_scores["test_recall"].mean()
    mean_cv_f1 = cv_scores["test_f1"].mean()

    # log with mlflow
    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag("model","Customer Churn Modelling")

        # log parameters (values truncated, stacking classifiers have very long nested params)
        mlflow.log_params({k: str(v)[:500] for k, v in model.get_params().items()})

        # log metrics
        mlflow.log_metric("train_roc_auc",train_roc_auc)
        mlflow.log_metric("test_roc_auc",test_roc_auc)
        mlflow.log_metric("train_recall",train_recall)
        mlflow.log_metric("test_recall",test_recall)
        mlflow.log_metric("train_f1",train_f1)
        mlflow.log_metric("test_f1",test_f1)
        mlflow.log_metric("mean_cv_roc_auc",mean_cv_roc_auc)
        mlflow.log_metric("mean_cv_recall",mean_cv_recall)
        mlflow.log_metric("mean_cv_f1",mean_cv_f1)

        # log individual cv scores
        for metric in ["roc_auc", "recall", "f1"]:
            mlflow.log_metrics({f"cv_{metric}_fold_{num}": score
                                for num, score in enumerate(cv_scores[f"test_{metric}"])})

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)

        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")

        # model signature (sample drawn once)
        sample_input = X_train.sample(20,random_state=42)
        model_signature = mlflow.models.infer_signature(model_input=sample_input,
                                    model_output=model.predict(sample_input))

        # log the final model and register it under MODEL_NAME
        # cloudpickle is used because the default skops format rejects xgboost/lightgbm types
        model_info = mlflow.sklearn.log_model(model,
                                              name="churn_model",
                                              signature=model_signature,
                                              serialization_format="cloudpickle",
                                              registered_model_name=MODEL_NAME)

        # log the preprocessor
        mlflow.log_artifact(preprocessor_path)

        # get the artifact uri of the logged model
        artifact_uri = model_info.model_uri

        logger.info("Mlflow logging complete and model logged")

    # get the run id
    run_id = run.info.run_id
    model_name = MODEL_NAME

    # save the model info
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")