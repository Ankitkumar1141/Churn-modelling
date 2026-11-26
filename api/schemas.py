from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .config import GENDERS, GEOGRAPHIES


class Customer(BaseModel):
    gender: str
    age: int = Field(ge=18, le=100)
    salary: float = Field(ge=0)
    geography: str
    tenure: int = Field(ge=0, le=50)
    balance: Optional[float] = Field(default=None, ge=0)  # KNNImputer fills nulls
    num_products: int = Field(ge=1, le=10)
    has_credit_card: int = Field(ge=0, le=1)
    is_active: int = Field(ge=0, le=1)

    # same cleaning as data_cleaning.py: lowercase categoricals
    @field_validator("gender", "geography", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("gender")
    @classmethod
    def _check_gender(cls, v):
        if v not in GENDERS:
            raise ValueError(f"gender must be one of {GENDERS}")
        return v

    @field_validator("geography")
    @classmethod
    def _check_geo(cls, v):
        if v not in GEOGRAPHIES:
            raise ValueError(f"geography must be one of {GEOGRAPHIES}")
        return v


class Prediction(BaseModel):
    churn_score: float
    churn_prediction: int
    risk_tier: str


class BatchRequest(BaseModel):
    customers: List[Customer]


class BatchResponse(BaseModel):
    count: int
    predictions: List[Prediction]
