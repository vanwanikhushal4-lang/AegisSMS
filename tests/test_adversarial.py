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
        return {
            "category": ID_TO_LABEL[int(np.argmax(probs))],
            "probs": {"PERSONAL": float(probs[0]), "TRANSACTIONAL": float(probs[1]), "PROMOTIONAL": float(probs[2])}
        }

    return predict

def test_personal_chat_categorization(model_and_assets):
    predict = model_and_assets
    res = predict("Hey bro are you free tonight? Let's grab dinner")
    assert res["category"] == "PERSONAL"
    assert res["probs"]["PERSONAL"] >= 0.70

def test_transactional_banking_categorization(model_and_assets):
    predict = model_and_assets
    res = predict("Sent Rs.50.00 from Kotak Bank A/c X2056 to VETAIL on 31-08-26. UPI Ref 624311493216.")
    assert res["category"] == "TRANSACTIONAL"

def test_transactional_otp_categorization(model_and_assets):
    predict = model_and_assets
    res = predict("Your OTP for Zepto order delivery is 9979. Share with delivery agent to confirm.")
    assert res["category"] == "TRANSACTIONAL"

def test_promotional_discount_categorization(model_and_assets):
    predict = model_and_assets
    res = predict("FLAT 15% OFF on Tommy Hilfiger & Lacoste on Tata CLiQ Luxury with code 1STLUXE: https://1kx.in/MYCLIQ")
    assert res["category"] == "PROMOTIONAL"

def test_promotional_telecom_recharge_categorization(model_and_assets):
    predict = model_and_assets
    res = predict("Abhi Recharge karein Rs348 se aur payein Unlimited 5G data for 28 days on Airtel Thanks App.")
    assert res["category"] == "PROMOTIONAL"
