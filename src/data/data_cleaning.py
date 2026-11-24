import numpy as np
import pandas as pd
import logging
from pathlib import Path

# create logger
logger = logging.getLogger("data_cleaning")
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

## features to drop
drop_columns = ['customer_id',
            'name', 
            'location_id', 
            'account_id']


def convert_xlsx_to_csv(input_path: Path, output_path: Path) -> Path:
    try:
        if input_path.suffix.lower() == ".csv":
            csv_path = input_path
            return csv_path
        csv_path = output_path

        sheets = pd.read_excel(input_path, sheet_name=None)
        combined = (sheets["Demographic"]
        .merge(sheets["Location"], on="LocationId", how="left")
        .merge(sheets["Account"], on="CustomerId", how="left"))

        combined.to_csv(csv_path, index=False)
        return csv_path

    except (TypeError, FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Failed to convert {input_path} to csv: {e}")
        raise


## load data 
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("File does not exist at specified location")
    return data

## rename_columns
def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .rename(columns={
            "CustomerId": "customer_id",
            "Name": "name",
            "Gender": "gender",
            "Age": "age",
            "Salary": "salary",
            "LocationId": "location_id",
            "Churned": "churned",
            "AccountId": "account_id",
            "Tenure": "tenure",
            "Balance": "balance",
            "NumProducts": "num_products",
            "HasCreditCard": "has_credit_card",
            "IsActive": "is_active",
            "Geography": "geography"
        })
        .rename(columns=str.lower)
    )

## drop_features
def drop_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return data.drop(columns=columns)

## cleaning encoding issues and datatypes
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """Clean text encoding issues and cast categorical datatypes."""
    return (
        data
        .assign(
            name = lambda x: x['name'].str.replace(r'\?', '', regex=True).str.strip(),
            gender = lambda x: x['gender'].str.lower(),
            geography = lambda x: x['geography'].str.lower(),
            has_credit_card = lambda x: x['has_credit_card'].astype(int),
            is_active = lambda x: x['is_active'].astype(int),
            churned = lambda x: x['churned'].astype(int)
        )
    )


## cleaning pipeline
def perform_data_cleaning(data: pd.DataFrame, saved_data_path: Path) -> None:
    cleaned_data = (
        data
        .pipe(rename_columns)
        .pipe(data_cleaning)
        .pipe(drop_features, columns= drop_columns)
    )
    ## save data to the sepecified path
    cleaned_data.to_csv(saved_data_path, index= False)

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exits
    cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
    # cleaned data file name
    cleaned_data_filename = "churn_cleaned.csv"
    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
    # data load path
    data_load_path = root_path / "data" / "raw_xlsx" / "raw_data.xlsx"

    ## csv data path
    csv_data_path = root_path / "data" / "raw" / "raw_data.csv"
    ## convert xlsx to csv
    csv_data_path = convert_xlsx_to_csv(input_path=data_load_path, output_path=csv_data_path)
    

    
    # load the data
    df = load_data(csv_data_path)
    logger.info("Data read successfully")

    # clean the data and save
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
    logger.info("Data cleaned and saved")

