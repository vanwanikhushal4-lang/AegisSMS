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

from preprocessing import clean_and_featurize, NUMERIC_FEATURES, LABEL_TO_ID, ID_TO_LABEL

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
            "pred_label": ID_TO_LABEL[int(np.argmax(probs))],
            "probs": {"HAM": float(probs[0]), "MARKETING_SPAM": float(probs[1]), "SMISHING": float(probs[2])}
        }

    return predict

def test_zero_width_character_evasion(model_and_assets):
    predict = model_and_assets
    adv_text = "URGENT: Your bank k\u200by\u200bc is expired. v\u200be\u200br\u200bi\u200bf\u200by now at http://sbi-update.top"
    res = predict(adv_text)
    assert res["pred_label"] == "SMISHING"

def test_homoglyph_and_leetspeak(model_and_assets):
    predict = model_and_assets
    adv_text = "URGENT: Y0ur acc0unt is bl0cked. Share your 0TP or passb00k to reactivate immediately"
    res = predict(adv_text)
    assert res["pred_label"] == "SMISHING"

def test_wrong_transfer_refund_social_engineering(model_and_assets):
    predict = model_and_assets
    adv_text = "Galti se Rs 10,000 aapke account me transfer ho gaya hai please iss UPI id par refund kar dijiye rohit@ybl"
    res = predict(adv_text)
    assert res["pred_label"] == "SMISHING"

def test_utility_disconnection_threat(model_and_assets):
    predict = model_and_assets
    adv_text = "Dear consumer, your electricity supply will be disconnected tonight at 9:30 PM due to unpaid bill. Call 9876543210 immediately."
    res = predict(adv_text)
    assert res["pred_label"] == "SMISHING"

def test_legitimate_otp_and_banking_ham(model_and_assets):
    predict = model_and_assets
    legit_text = "Your OTP for HDFC Bank NetBanking transaction of INR 4,500.00 is 592810. Do not share OTP with anyone."
    res = predict(legit_text)
    assert res["pred_label"] == "HAM"
    assert res["probs"]["HAM"] >= 0.70

def test_legitimate_brand_promotions(model_and_assets):
    predict = model_and_assets
    promo_text = "Flash Sale! Get 40% discount on all orders above Rs 999 on Swiggy today. Use code CRAVINGS. Order now: https://swiggy.com/offers"
    res = predict(promo_text)
    assert res["pred_label"] in ("MARKETING_SPAM", "HAM")
