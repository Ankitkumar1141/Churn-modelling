import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MODEL_FILE = "model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
INFERENCE_PIPELINE_FILE = "inference_pipeline.joblib"
MODEL_MODE = os.getenv("MODEL_MODE", "two_step")  # or "full_pipeline"

# Column order used at training time
FEATURES = [
    "gender", "age", "salary", "geography", "tenure",
    "balance", "num_products", "has_credit_card", "is_active",
]
NUMERIC_FEATURES = ["age", "salary", "tenure", "balance", "num_products",
                    "has_credit_card", "is_active"]

# ASSUMPTION: replace with the unique values in YOUR data
# (check: df["geography"].unique(), df["gender"].unique() on churn_cleaned.csv)
GENDERS = ["male", "female"]
GEOGRAPHIES = ["france", "spain", "germany"]

# Decision threshold + risk tiers (scores come from a model trained on an
# undersampled 50/50 train set, so treat them as risk scores, not true probabilities)
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", 0.5))
MEDIUM_RISK = float(os.getenv("MEDIUM_RISK", 0.4))
HIGH_RISK = float(os.getenv("HIGH_RISK", 0.7))

MAX_BATCH_SIZE = 5000
