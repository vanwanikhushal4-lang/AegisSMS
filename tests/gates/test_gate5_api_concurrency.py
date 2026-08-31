# -*- coding: utf-8 -*-
import pytest
import os
import sys
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from api import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

def test_predict_single_sms():
    payload = {"text": "Your OTP for HDFC Bank login is 482910. Do not share with anyone."}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "TRANSACTIONAL"

def test_predict_batch_sms():
    payload = {
        "messages": [
            "Hey bro, are we meeting at 5 PM today?",
            "An amount of INR 500.00 has been debited from your Kotak Bank A/c X2056 on 31-Aug.",
            "50% off on all Myntra orders today! Use code FESTIVE50: https://myntra.com/sale"
        ]
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 3
    assert results[0]["category"] == "PERSONAL"
    assert results[1]["category"] == "TRANSACTIONAL"
    assert results[2]["category"] == "PROMOTIONAL"
