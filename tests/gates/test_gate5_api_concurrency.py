# -*- coding: utf-8 -*-
import pytest
import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from api import app, IS_SCAM_THRESHOLD

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "is_scam_operating_threshold" in data

def test_predict_single_sms():
    payload = {"text": "Your OTP for HDFC Bank login is 482910. Do not share with anyone."}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "TRANSACTIONAL"
    assert data["is_scam"] == False
    assert data["operating_threshold"] == IS_SCAM_THRESHOLD

def test_predict_batch_sms_4way():
    payload = {
        "messages": [
            "Hey bro, are we meeting at 5 PM today?",
            "An amount of INR 500.00 has been debited from your Kotak Bank A/c X2056 on 31-Aug.",
            "50% off on all Myntra orders today! Use code FESTIVE50: https://myntra.com/sale",
            "URGENT: Electricity will be cut off tonight at 9:30 PM due to unpaid bill. Call officer immediately at 9876543210: http://bit.ly/msedcl-bill.apk"
        ]
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 4
    assert results[0]["category"] == "PERSONAL"
    assert results[1]["category"] == "TRANSACTIONAL"
    assert results[2]["category"] == "PROMOTIONAL"
    assert results[3]["category"] == "SCAM"
    assert results[3]["is_scam"] == True

def test_api_production_fpr_compliance():
    """Verify that the production API endpoint satisfies the Legitimate-to-SCAM FPR <= 0.5% requirement."""
    test_csv = os.path.join(BASE_DIR, "prepared_4way_p5", "test.csv")
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv, encoding="utf-8-sig")
        legit = df[df["category"].isin(["PERSONAL", "TRANSACTIONAL", "PROMOTIONAL"])]
        
        sample_legit = legit.sample(n=min(200, len(legit)), random_state=42)
        false_positives = 0
        tested = 0
        for _, row in sample_legit.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue
            res = client.post("/predict", json={"text": text})
            assert res.status_code == 200
            data = res.json()
            tested += 1
            if data["is_scam"]:
                false_positives += 1
        
        api_fpr = float(false_positives) / tested if tested > 0 else 0.0
        assert api_fpr <= 0.005, f"API False Positive Rate {api_fpr*100:.3f}% exceeds 0.5% requirement!"
