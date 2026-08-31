# -*- coding: utf-8 -*-
"""
Interactive CLI for live testing raw SMS texts against AegisSMS 3-Way Engine.
Usage:
    python interactive_predict.py "Your message here"
"""
import sys
import os
import json
import pickle
import numpy as np
import scipy.sparse as sp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

from preprocessing import clean_and_featurize, NUMERIC_FEATURES, ID_TO_LABEL

with open(os.path.join(ARTIFACT_DIR, "sms_model_3way.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "vectorizer_3way.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(ARTIFACT_DIR, "feature_scaler_3way.json"), "r") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"], dtype=np.float32)
std = np.array(scaler["std"], dtype=np.float32)

def predict_text(text: str):
    cleaned, raw_feats = clean_and_featurize(text)
    x_t = vectorizer.transform([cleaned])
    r_num = np.array([raw_feats[k] for k in NUMERIC_FEATURES], dtype=np.float32)
    s_num = (r_num - mean) / std
    x_fused = sp.hstack([x_t, sp.csr_matrix(s_num.reshape(1, -1))], format="csr")

    probs = clf.predict_proba(x_fused)[0]
    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]

    print("\n" + "="*70)
    print(f"INPUT SMS: {text}")
    print("="*70)
    print(f"VERDICT:     {pred_label}")
    print(f"CONFIDENCE:  {probs[pred_id]*100:.2f}%\n")
    print("PROBABILITY DISTRIBUTION:")
    print(f"  🟢 HAM (Legitimate):         {probs[0]*100:6.2f}%")
    print(f"  🟡 MARKETING SPAM (Promo):   {probs[1]*100:6.2f}%")
    print(f"  🔴 SMISHING (Phishing/Fraud): {probs[2]*100:6.2f}%\n")
    print("EXTRACTED THREAT SIGNALS:")
    print(f"  - Has URL:              {bool(raw_feats['has_url'] > 0)}")
    print(f"  - Has Phone:            {bool(raw_feats['has_phone'] > 0)}")
    print(f"  - Urgency Verbs:        {int(raw_feats['urgency_count'])}")
    print(f"  - Credential Keywords:  {int(raw_feats['sensitive_info_count'])}")
    print(f"  - Refund Scam Phrases:  {int(raw_feats['refund_scam_count'])}")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        predict_text(msg)
    else:
        print("Please provide a text string: python interactive_predict.py \"SMS text\"")
