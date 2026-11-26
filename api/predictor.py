import joblib
import numpy as np
import pandas as pd
from sklearn import set_config

from . import config

# same setting as data_preprocessing.py (the model was trained on DataFrames)
set_config(transform_output="pandas")


def risk_tier(score: float) -> str:
    if score >= config.HIGH_RISK:
        return "High"
    if score >= config.MEDIUM_RISK:
        return "Medium"
    return "Low"


class ChurnPredictor:
    """
    MODEL_MODE=two_step (default): models/preprocessor.joblib -> models/model.joblib
        (exactly what src/features/data_preprocessing.py + src/models/train.py save)
    MODEL_MODE=full_pipeline: models/inference_pipeline.joblib (single fitted pipeline
        from notebooks/Final_Estimator.ipynb)
    """

    def __init__(self):
        self.mode = config.MODEL_MODE
        if self.mode == "full_pipeline":
            self.pipeline = joblib.load(config.MODEL_DIR / config.INFERENCE_PIPELINE_FILE)
        else:
            self.preprocessor = joblib.load(config.MODEL_DIR / config.PREPROCESSOR_FILE)
            self.model = joblib.load(config.MODEL_DIR / config.MODEL_FILE)

    def _scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.mode == "full_pipeline":
            return self.pipeline.predict_proba(df)[:, 1]
        return self.model.predict_proba(self.preprocessor.transform(df))[:, 1]

    def predict(self, records: list[dict]) -> list[dict]:
        df = pd.DataFrame(records)[config.FEATURES]
        # None balance would make an object column; force numeric so NaN is used
        df[config.NUMERIC_FEATURES] = df[config.NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        ).astype(float)
        scores = self._scores(df)
        return [
            {
                "churn_score": round(float(s), 4),
                "churn_prediction": int(s >= config.THRESHOLD),
                "risk_tier": risk_tier(float(s)),
            }
            for s in np.atleast_1d(scores)
        ]
