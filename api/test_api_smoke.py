# api/test_api_smoke.py
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body

def test_predict_sklearn():
    r = client.post("/predict?model=sklearn", json={
        "genre":"Fiction","pages":220,"complexity":0.55,"rating":4.6
    })
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body

def test_predict_torch():
    r = client.post("/predict?model=torch", json={
        "genre":"Fantasy","pages":320,"complexity":0.70,"rating":4.7
    })
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
