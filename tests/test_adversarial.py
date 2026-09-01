# -*- coding: utf-8 -*-
import pytest
import os
import sys
import pickle
import numpy as np
import scipy.sparse as sp
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from preprocessing import clean_and_featurize, NUMERIC_FEATURES, ID_TO_LABEL

@pytest.fixture(scope="module")
def model_and_assets():
    artifact_dir = os.path.join(BASE_DIR, "artifacts")
    with open(os.path.join(artifact_dir, "sms_model_3way.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(artifact_dir, "vectorizer_3way.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(artifact_dir, "feature_scaler_3way.json"), "r") as f:
        scaler = json.load(f)

    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    def predict(text: str):
        c, fts = clean_and_featurize(text)
        x_t = vectorizer.transform([c])
        r_num = np.array([fts[k] for k in NUMERIC_FEATURES], dtype=np.float32)
        s_num = (r_num - mean) / std
        x_fused = sp.hstack([x_t, sp.csr_matrix(s_num.reshape(1, -1))], format="csr")
        probs = clf.predict_proba(x_fused)[0]
        is_scam = bool(probs[3] >= 0.69)
        if is_scam:
            pred_label = "SCAM"
        else:
            non_scam_idx = int(np.argmax(probs[:3]))
            pred_label = ID_TO_LABEL[non_scam_idx]
        return {
            "category": pred_label,
            "is_scam": is_scam,
            "probs": {ID_TO_LABEL[i]: float(probs[i]) for i in range(4)}
        }

    return predict

def test_personal_chat(model_and_assets):
    predict = model_and_assets
    res = predict("Hey buddy are you coming to play football this evening?")
    assert res["category"] == "PERSONAL"

def test_transactional_banking(model_and_assets):
    predict = model_and_assets
    res = predict("Sent Rs.50.00 from Kotak Bank A/c X2056 to VETAIL on 31-08-26. UPI Ref 624311493216.")
    assert res["category"] == "TRANSACTIONAL"

def test_promotional_discount(model_and_assets):
    predict = model_and_assets
    res = predict("FLAT 25% OFF on Tommy Hilfiger & Puma on Tata CLiQ Luxury. Use code LUXE25: https://tatacliq.com/sale")
    assert res["category"] == "PROMOTIONAL"

def test_electricity_smishing_scam(model_and_assets):
    predict = model_and_assets
    res = predict("Dear customer, your electricity power will be disconnected tonight at 9:30 PM. Call immediately at 08634017553 or click http://bit.ly/msedcl-pay.apk")
    assert res["category"] == "SCAM"
    assert res["probs"]["SCAM"] >= 0.70

def test_lottery_prize_scam(model_and_assets):
    predict = model_and_assets
    res = predict("WINNER! You have won 25 Lakh in KBC lottery. To claim your prize money call 9876543210 immediately.")
    assert res["category"] == "SCAM"
    assert res["probs"]["SCAM"] >= 0.85

def test_pan_block_scam(model_and_assets):
    predict = model_and_assets
    res = predict("Your SBI Bank account has been locked. Update your PAN card immediately to avoid suspension: http://sbi-kyc-update.apk")
    assert res["category"] == "SCAM"
    assert res["probs"]["SCAM"] >= 0.69
