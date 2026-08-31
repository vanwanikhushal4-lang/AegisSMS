# -*- coding: utf-8 -*-
"""
FastAPI application exposing the multilingual (English / Hinglish / Hindi /
Marathi) SMS Ham-vs-Spam classifier. Loads the TFLite model and the saved
preprocessing artifacts (vocabulary, feature scaler) once at startup.
"""
import json
import os
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import clean_and_featurize, NUMERIC_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

with open(os.path.join(ARTIFACT_DIR, "model_config.json"), encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)
with open(os.path.join(ARTIFACT_DIR, "vocabulary.json"), encoding="utf-8") as f:
    VOCABULARY = json.load(f)
with open(os.path.join(ARTIFACT_DIR, "feature_scaler.json"), encoding="utf-8") as f:
    SCALER = json.load(f)

MAX_LEN = MODEL_CONFIG["max_len"]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCABULARY)}
UNK_ID = 1  # index 1 is '[UNK]' in the saved vocabulary (index 0 is padding)
FEATURE_MEAN = np.array(SCALER["mean"], dtype="float32")
FEATURE_STD = np.array(SCALER["std"], dtype="float32")
# Tuned on the validation set to favor catching fraud/phishing (F2, recall
# weighted 2x precision) over the naive 0.5 cutoff -- falls back to 0.5 if
# an older model_config.json without this key is loaded.
DECISION_THRESHOLD = MODEL_CONFIG.get("decision_threshold", 0.5)

_interpreter = tf.lite.Interpreter(model_path=os.path.join(ARTIFACT_DIR, "sms_spam_model.tflite"))
_interpreter.allocate_tensors()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()
_IDS_INPUT = next(d for d in _input_details if d["shape"][-1] == MAX_LEN)
_NUM_INPUT = next(d for d in _input_details if d["index"] != _IDS_INPUT["index"])


def _tokenize(clean_text: str) -> np.ndarray:
    ids = [TOKEN_TO_ID.get(tok, UNK_ID) for tok in clean_text.split()][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return np.array([ids], dtype=np.int32)


def _numeric_vector(features: dict) -> np.ndarray:
    raw = np.array([features[k] for k in NUMERIC_FEATURES], dtype="float32")
    scaled = (raw - FEATURE_MEAN) / FEATURE_STD
    return scaled.reshape(1, -1).astype("float32")


def predict_sms(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    clean_text, features = clean_and_featurize(text)
    _interpreter.set_tensor(_IDS_INPUT["index"], _tokenize(clean_text))
    _interpreter.set_tensor(_NUM_INPUT["index"], _numeric_vector(features))
    _interpreter.invoke()
    prob_spam = float(_interpreter.get_tensor(_output_details[0]["index"]).reshape(-1)[0])
    is_spam = prob_spam > DECISION_THRESHOLD

    return {
        "text": text,
        "label": "Spam" if is_spam else "Ham",
        "is_spam": is_spam,
        "spam_probability": round(prob_spam, 6),
        "ham_probability": round(1.0 - prob_spam, 6),
    }


app = FastAPI(
    title="Multilingual SMS Spam/Ham Detector",
    description=(
        "Classifies SMS messages in English, Hinglish, Hindi and Marathi as "
        "Ham (legitimate) or Spam using a TFLite TextCNN model."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The SMS message text to classify")


class PredictBatchRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, description="A list of SMS messages to classify")


@app.get("/health")
def health():
    return {"status": "ok", "model_config": MODEL_CONFIG}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        return predict_sms(payload.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(payload: PredictBatchRequest):
    results = []
    for msg in payload.messages:
        try:
            results.append(predict_sms(msg))
        except ValueError as e:
            results.append({"text": msg, "error": str(e)})
    return {"results": results}


if __name__ == "__main__":
    # Lets `python api.py` work directly, in addition to `python main.py`.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
