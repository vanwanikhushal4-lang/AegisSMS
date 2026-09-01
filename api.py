# -*- coding: utf-8 -*-
"""
FastAPI Production Prediction Service for AegisSMS 4-Way Intent & Threat Engine
Categories: PERSONAL, TRANSACTIONAL, PROMOTIONAL, SCAM
Uniform Decision Rule: SCAM if predicted category is SCAM or P(SCAM) >= is_scam_operating_threshold
"""
import os
import json
import pickle
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import clean_and_featurize, NUMERIC_FEATURES, ID_TO_LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# Load model, vectorizer, scaler, and calibrated contract threshold
with open(os.path.join(ARTIFACT_DIR, "sms_model_3way.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "vectorizer_3way.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "feature_scaler_3way.json"), "r", encoding="utf-8") as f:
    scaler = json.load(f)
with open(os.path.join(ARTIFACT_DIR, "aegis_model_contract.json"), "r", encoding="utf-8") as f:
    contract = json.load(f)

mean = np.array(scaler["mean"], dtype=np.float32)
std = np.array(scaler["std"], dtype=np.float32)
IS_SCAM_THRESHOLD = float(contract.get("is_scam_operating_threshold", 0.69))

app = FastAPI(
    title="AegisSMS 4-Way Intent & Threat Intelligence Engine",
    version="2.3.0",
    description="Production API for classifying SMS into PERSONAL, TRANSACTIONAL, PROMOTIONAL, and SCAM."
)

class SmsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Raw SMS text message")

class BatchSmsRequest(BaseModel):
    messages: List[str] = Field(..., max_length=200, description="List of raw SMS text messages")

def predict_single(text: str) -> Dict[str, Any]:
    cleaned, raw_feats = clean_and_featurize(text)
    x_t = vectorizer.transform([cleaned])
    r_num = np.array([raw_feats[k] for k in NUMERIC_FEATURES], dtype=np.float32)
    s_num = (r_num - mean) / std
    x_fused = sp.hstack([x_t, sp.csr_matrix(s_num.reshape(1, -1))], format="csr")

    probs = clf.predict_proba(x_fused)[0]
    
    # Unified Calibrated Decision Rule
    is_scam = bool(probs[3] >= IS_SCAM_THRESHOLD)
    if is_scam:
        pred_label = "SCAM"
        confidence = float(round(probs[3], 4))
    else:
        non_scam_idx = int(np.argmax(probs[:3]))
        pred_label = ID_TO_LABEL[non_scam_idx]
        confidence = float(round(probs[non_scam_idx], 4))

    return {
        "text": text,
        "category": pred_label,
        "is_scam": is_scam,
        "confidence": confidence,
        "operating_threshold": IS_SCAM_THRESHOLD,
        "probabilities": {
            "PERSONAL": float(round(probs[0], 4)),
            "TRANSACTIONAL": float(round(probs[1], 4)),
            "PROMOTIONAL": float(round(probs[2], 4)),
            "SCAM": float(round(probs[3], 4))
        },
        "signals": {
            "has_url": bool(raw_feats["has_url"] > 0),
            "has_phone": bool(raw_feats["has_phone"] > 0),
            "urgency_detected": bool(raw_feats["urgency_count"] > 0),
            "credentials_requested": bool(raw_feats["sensitive_info_count"] > 0),
            "refund_phrases": bool(raw_feats["refund_scam_count"] > 0)
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "model_version": "2.3.0-4WAY-SCAM",
        "is_scam_operating_threshold": IS_SCAM_THRESHOLD
    }

@app.post("/predict")
def predict_sms_endpoint(req: SmsRequest):
    try:
        return predict_single(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch_endpoint(req: BatchSmsRequest):
    try:
        return {"results": [predict_single(t) for t in req.messages]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
