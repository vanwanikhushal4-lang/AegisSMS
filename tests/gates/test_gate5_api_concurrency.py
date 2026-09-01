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
    payload = {"text": "Your OTP for HDFC Bank login is <OTP>. Do not share with anyone."}
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
            "An amount of INR 500.00 has been debited from your Kotak Bank A/c <ACCT> on 31-Aug.",
            "50% off on all Myntra orders today! Use code FESTIVE50: https://myntra.com/sale",
            "URGENT: Electricity will be cut off tonight at 9:30 PM due to unpaid bill. Call officer immediately at <PHONE>: http://bit.ly/msedcl-bill.apk"
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

def test_api_full_testset_evaluation_and_fpr_compliance():
    """Evaluates the entire holdout test set through the production API to verify FPR <= 0.5%."""
    test_csv = os.path.join(BASE_DIR, "prepared_4way_p5", "test.csv")
    assert os.path.exists(test_csv), "Missing prepared test.csv"

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    legit_df = df[df["category"] != "SCAM"]

    # Batch test via API in chunks of 100
    batch_size = 100
    all_texts = legit_df["text"].tolist()
    total_false_positives = 0
    total_evaluated = 0

    for i in range(0, len(all_texts), batch_size):
        chunk = [str(t).strip() for t in all_texts[i:i + batch_size] if str(t).strip()]
        if not chunk:
            continue
        res = client.post("/predict/batch", json={"messages": chunk})
        assert res.status_code == 200, f"API batch predict failed: {res.text}"
        batch_results = res.json()["results"]
        for r in batch_results:
            total_evaluated += 1
            if r["is_scam"]:
                total_false_positives += 1

    api_fpr = total_false_positives / total_evaluated if total_evaluated > 0 else 0.0
    print(f"\nAPI Full Holdout Evaluation: Evaluated={total_evaluated}, False Positives={total_false_positives}, FPR={api_fpr*100:.3f}%")
    assert api_fpr <= 0.0050, f"API False Positive Rate {api_fpr*100:.3f}% ({total_false_positives}/{total_evaluated}) exceeds 0.5% limit!"
