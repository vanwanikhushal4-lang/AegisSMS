# -*- coding: utf-8 -*-
"""
FastAPI Production Prediction Service for AegisSMS Intent Engine
Categories: PERSONAL, TRANSACTIONAL, PROMOTIONAL
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

with open(os.path.join(ARTIFACT_DIR, "sms_model_3way.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "vectorizer_3way.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "feature_scaler_3way.json"), "r") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"], dtype=np.float32)
std = np.array(scaler["std"], dtype=np.float32)

app = FastAPI(
    title="AegisSMS Multi-Class Intent Engine",
    version="2.1.0",
    description="Categorizes SMS messages into PERSONAL, TRANSACTIONAL, and PROMOTIONAL."
)

class SmsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Raw SMS text message")

class BatchSmsRequest(BaseModel):
    messages: List[str] = Field(..., max_length=200, description="List of raw SMS text messages")

def predict_single(text: str) -> Dict[str, Any]:
    cleaned, raw_feats = clean_and_featurize(text)
    x_t = vectorizer.transform([cleaned])
    r_num = np.array([raw_feats[k] for k in NUMERIC_FEATURES], dtype=np.float32)
    s_num = (r_num - mean) / std
    x_fused = sp.hstack([x_t, sp.csr_matrix(s_num.reshape(1, -1))], format="csr")

    probs = clf.predict_proba(x_fused)[0]
    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]

    return {
        "text": text,
        "category": pred_label,
        "confidence": float(round(probs[pred_id], 4)),
        "probabilities": {
            "PERSONAL": float(round(probs[0], 4)),
            "TRANSACTIONAL": float(round(probs[1], 4)),
            "PROMOTIONAL": float(round(probs[2], 4))
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
    return {"status": "HEALTHY", "model_version": "2.1.0-INTENT"}

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
