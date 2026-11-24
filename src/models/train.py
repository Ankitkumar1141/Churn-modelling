import pandas as pd
import yaml
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.ensemble import StackingClassifier

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
    
    # rf_params
    rf_params = model_params['Random_Forest']
    logger.info("Read Random Forest Parameters")
    
    # build random forest model
    rf = RandomForestClassifier(**rf_params)
    logger.info("Built random forest model")

    # light gbm params
    lgbm_params = model_params["LightGBM"]
    logger.info("Read LightGBM parameters.")
    lgbm = LGBMClassifier(**lgbm_params)
    logger.info("Built LightGBM model")
    
    # xgb params
    xgb_params = model_params["XGBoost"]
    logger.info("Read XGBoost parameters")
    xgb = XGBClassifier(**xgb_params)
    logger.info("Built XgBoost model")
    
    # meta model
    lr = LogisticRegression(max_iter=1000)
    logger.info("Meta model built")
    
    
    # form the stacking regressor
    stacking_class = StackingClassifier(estimators=[("rf_model",rf),
                                                 ("lgbm_model",lgbm),
                                                 ("xgb_model",xgb)],
                                     final_estimator=lr,
                                     cv=5,n_jobs=-1)
    logger.info("Stacking classifier built")
    
    # fit the model on training data
    train_model(stacking_class,X_train,y_train)
    logger.info("Model training completed")
    
    # model name
    model_filename = "model.joblib"
    # directory to save model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)
    

    # save the model
    save_model(model=stacking_class,
            save_dir=model_save_dir,
            model_name=model_filename)
    logger.info("Trained model saved to location")
    
    
