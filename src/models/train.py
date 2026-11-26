import pandas as pd
import yaml
import joblib
import logging
from xgboost import XGBClassifier
from pathlib import Path

TARGET = "churned"

# create logger
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

## load_data from the directory
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise

## read_params from the yaml file
def read_params(file_path):
    with open(file_path,"r") as f:
        params_file = yaml.safe_load(f)

    return params_file

def save_model(model, save_dir: Path, model_name: str):
    save_location = save_dir / model_name
    joblib.dump(value=model,filename=save_location)


def train_model(model, X_train: pd.DataFrame, y_train):
    model.fit(X_train, y_train)
    return model

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X= data.drop(columns=[target_column])
    y= data[target_column]
    return X,y

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    data_path = root_path / "data" / "processed" / "train.csv"
    # parameters file
    params_file_path = root_path / "params.yaml"

    # load the training data
    training_data = load_data(data_path)
    logger.info("Training Data read successfully")

    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")

    # model parameters
    model_params = read_params(params_file_path)['Train']

    # xgb params (champion model selected in notebooks/Final_Estimator.ipynb)
    xgb_params = model_params["XGBoost"]
    logger.info("Read XGBoost parameters")

    # build the champion XGBoost model
    model = XGBClassifier(**xgb_params)
    logger.info("Built XGBoost model")

    # fit the model on training data
    train_model(model,X_train,y_train)
    logger.info("Model training completed")

    # model name
    model_filename = "model.joblib"
    # directory to save model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # save the model
    save_model(model=model,
            save_dir=model_save_dir,
            model_name=model_filename)
    logger.info("Trained model saved to location")