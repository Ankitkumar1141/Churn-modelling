import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler)
from imblearn.under_sampling import RandomUnderSampler
import joblib

from sklearn import set_config

# set the transformer outputs to pandas
set_config(transform_output='pandas')

# create logger
logger = logging.getLogger("Data_Preprocessing")
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

## read parameters
def read_params(params_path: Path) -> dict:
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

## load_data
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"File not found at {data_path}")
        raise
    return data

## split into X and y
def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def resample_data(X: pd.DataFrame, y: pd.Series, random_state: int):
    # class imbalance handling, used on the train data only
    logger.info(f"Train class distribution before resampling: {y.value_counts().to_dict()}")
    sampler = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    # reset the index so X and y stay aligned when joined
    X_resampled = X_resampled.reset_index(drop=True)
    y_resampled = y_resampled.reset_index(drop=True)
    logger.info(f"Train class distribution after resampling: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled

def identify_column_types(X: pd.DataFrame):
    # separate the remaining columns into groups for the preprocessor
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    binary_cols = [c for c in numeric_cols if X[c].nunique() == 2]
    scale_cols = [c for c in numeric_cols if c not in binary_cols]
    return scale_cols, binary_cols, categorical_cols

def transform_train_data(preprocessor, X_train: pd.DataFrame) -> pd.DataFrame:
    # fit the preprocessor on the train data and transform it
    return preprocessor.fit_transform(X_train)

def transform_test_data(preprocessor, X_test: pd.DataFrame) -> pd.DataFrame:
    # transform the test data with the already fitted preprocessor
    return preprocessor.transform(X_test)

def join_X_and_y(X: pd.DataFrame, y: pd.Series):
    # join based on indexes
    joined_df = X.join(y,how='inner')
    return joined_df

def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)

def save_transformer(transformer, save_dir: Path, transformer_name: str):
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    joblib.dump(value=transformer,filename=save_location)


if __name__ == "__main__":
    # paths
    # root path
    root_path = Path(__file__).parent.parent.parent
    # interim data load paths
    interim_train_path = root_path / "data" / "interim" / "train.csv"
    interim_test_path = root_path / "data" / "interim" / "test.csv"
    # class balance directory (output of resampling)
    class_balance_dir = root_path / "data" / "class_balance"
    class_balance_dir.mkdir(exist_ok=True, parents=True)
    balance_train_path = class_balance_dir / "train.csv"
    balance_test_path = class_balance_dir / "test.csv"
    # save data directory
    save_data_dir = root_path / "data" / "processed"
    # make dir if not present
    save_data_dir.mkdir(exist_ok=True,parents=True)
    # train and test data save paths
    # filenames
    train_trans_filename = "train.csv"
    test_trans_filename = "test.csv"
    # save path for train and test
    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename
    # parameters file
    params_file_path = root_path / "params.yaml"

    # read the parameters
    parameters = read_params(params_file_path)['Data_Preprocessing']
    target = parameters['target']
    n_neighbors = parameters['n_neighbors']
    random_state = parameters['random_state']
    logger.info("parameters read successfully")

    # ---- Step 1: load the data from the interim directory ----
    train_df = load_data(data_path=interim_train_path)
    logger.info("Train data loaded successfully")
    test_df = load_data(data_path=interim_test_path)
    logger.info("Test data loaded successfully")

    # split the train and test data
    X_train, y_train = make_X_and_y(data=train_df,target_column=target)
    X_test, y_test = make_X_and_y(data=test_df, target_column=target)
    logger.info("Data splitting completed")

    # ---- Step 2: resample the train data and save to class_balance ----
    X_train, y_train = resample_data(X=X_train, y=y_train, random_state=random_state)
    logger.info("Train data resampled")

    balanced_train_df = join_X_and_y(X_train, y_train)
    balanced_test_df = join_X_and_y(X_test, y_test)
    save_data(data=balanced_train_df, save_path=balance_train_path)
    save_data(data=balanced_test_df, save_path=balance_test_path)
    logger.info("Resampled train data and test data saved to class_balance directory")

    # ---- Step 3: load from class_balance, transform and save to processed ----
    train_df = load_data(data_path=balance_train_path)
    logger.info("Train data loaded from class_balance")
    test_df = load_data(data_path=balance_test_path)
    logger.info("Test data loaded from class_balance")

    X_train, y_train = make_X_and_y(data=train_df,target_column=target)
    X_test, y_test = make_X_and_y(data=test_df, target_column=target)
    logger.info("Data splitting completed")

    # separate the remaining columns into groups for the preprocessor
    scale_cols, binary_cols, categorical_cols = identify_column_types(X_train)
    logger.info(f"Numeric (to scale): {scale_cols}")
    logger.info(f"Binary pass-thru: {binary_cols}")
    logger.info(f"Categorical OHE: {categorical_cols}")

    # preprocessor
    column_transformer = ColumnTransformer(transformers=[
            ("scale", MinMaxScaler(), scale_cols),
            ("nominal_encode", OneHotEncoder(drop="first",
                                            handle_unknown="ignore",
                                            sparse_output=False), categorical_cols)],
                                    remainder="passthrough",
                                    verbose_feature_names_out=False)

    # missing values (balance) are filled after the transformations
    preprocessor = Pipeline(steps=[
            ("preprocessor", column_transformer),
            ("knn_imputer", KNNImputer(n_neighbors=n_neighbors))])

    # fit the preprocessor on X_train and transform it
    X_train_trans = transform_train_data(preprocessor=preprocessor, X_train=X_train)
    logger.info("Preprocessor is trained and train data is transformed")
    # transform the test data using the fitted preprocessor
    X_test_trans = transform_test_data(preprocessor=preprocessor, X_test=X_test)
    logger.info("Test data is transformed")

    # join back X and y
    train_trans_df = join_X_and_y(X_train_trans, y_train)
    test_trans_df = join_X_and_y(X_test_trans, y_test)
    logger.info("Datasets joined")

    # save the transformed data
    data_subsets = [train_trans_df, test_trans_df]
    data_paths = [save_train_trans_path,save_test_trans_path]
    filename_list = [train_trans_filename, test_trans_filename]
    for filename , path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace('.csv', '')} data saved to location")

    # save the preprocessor to location
    # transformer name
    transformer_filename = "preprocessor.joblib"
    # directory to save transformers
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)
    # save the transformer
    save_transformer(transformer=preprocessor,
                     save_dir=transformer_save_dir,
                     transformer_name=transformer_filename)
    logger.info("Preprocessor saved to location")