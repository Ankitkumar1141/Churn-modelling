from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api import config
from api.predictor import ChurnPredictor
from api.schemas import BatchRequest, BatchResponse, Customer, Prediction

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = ChurnPredictor()  # fails fast if model files are missing
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/meta")
def meta():
    """Single source of truth for the frontend (dropdown values, thresholds)."""
    return {
        "features": config.FEATURES,
        "genders": config.GENDERS,
        "geographies": config.GEOGRAPHIES,
        "threshold": config.THRESHOLD,
        "medium_risk": config.MEDIUM_RISK,
        "high_risk": config.HIGH_RISK,
        "max_batch_size": config.MAX_BATCH_SIZE,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        return state["predictor"].predict([customer.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.customers:
        raise HTTPException(status_code=422, detail="customers list is empty")
    if len(req.customers) > config.MAX_BATCH_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"Max batch size is {config.MAX_BATCH_SIZE}")
    try:
        preds = state["predictor"].predict([c.model_dump() for c in req.customers])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"count": len(preds), "predictions": preds}
