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
    payload = {"text": "URGENT: Your bank account will be blocked. Update KYC at http://sbi-kyc.top"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "SMISHING"
    assert data["risk_level"] == "CRITICAL"
    assert data["threat_signals"]["has_url"] == True

def test_predict_batch_sms():
    payload = {
        "messages": [
            "Your OTP is 482910. Do not share with anyone.",
            "50% off on Myntra sale today! Shop now at https://myntra.com/sale",
            "Dear customer, your electricity will be disconnected tonight. Pay immediately at 9876543210"
        ]
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 3
    assert results[0]["prediction"] == "HAM"
    assert results[2]["prediction"] == "SMISHING"
