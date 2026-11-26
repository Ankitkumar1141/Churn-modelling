"""Validation tests that need no model files (run from the repo root: pytest)."""
from fastapi.testclient import TestClient
from app import app
 
client = TestClient(app)  # no `with` -> lifespan (model loading) is skipped
 
VALID = {"gender": "Male", "age": 40, "salary": 50000, "geography": "France",
         "tenure": 5, "balance": None, "num_products": 1,
         "has_credit_card": 1, "is_active": 1}
 
 
def test_meta():
    assert client.get("/meta").json()["features"][0] == "gender"
 
 
def test_rejects_bad_age():
    assert client.post("/predict", json={**VALID, "age": 5}).status_code == 422
 
 
def test_rejects_unknown_geography():
    assert client.post("/predict", json={**VALID, "geography": "mars"}).status_code == 422
 
