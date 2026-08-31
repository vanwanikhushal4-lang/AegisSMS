# -*- coding: utf-8 -*-
"""
Interactive CLI for live testing 4-Way SMS classification (PERSONAL, TRANSACTIONAL, PROMOTIONAL, SCAM).
Usage:
    python interactive_predict.py "Your message here"
"""
import sys
import os
import json
import pickle
import numpy as np
import scipy.sparse as sp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

    category_badge = {
        "PERSONAL": "🟢 PERSONAL (Peer-to-peer / Chat)",
        "TRANSACTIONAL": "🔵 TRANSACTIONAL (Banking / OTP / Alerts)",
        "PROMOTIONAL": "🟡 PROMOTIONAL (Offers / Sales / Ads)",
        "SCAM": "🚨 SCAM / PHISHING (Malicious / Fraud Threat)"
    }.get(pred_label, pred_label)

    print("\n" + "="*70)
    print(f"INPUT SMS: {text}")
    print("="*70)
    print(f"VERDICT:     {category_badge}")
    print(f"CONFIDENCE:  {probs[pred_id]*100:.2f}%\n")
    print("4-WAY PROBABILITIES:")
    print(f"  🟢 [PERSONAL]      (Peer-to-peer / Chat):    {probs[0]*100:6.2f}%")
    print(f"  🔵 [TRANSACTIONAL] (Banking / OTP / Alerts): {probs[1]*100:6.2f}%")
    print(f"  🟡 [PROMOTIONAL]   (Offers / Sales / Ads):   {probs[2]*100:6.2f}%")
    print(f"  🚨 [SCAM]          (Phishing / Fraud):       {probs[3]*100:6.2f}%\n")
    print("EXTRACTED SIGNALS:")
    print(f"  - Has URL:              {bool(raw_feats['has_url'] > 0)}")
    print(f"  - Has Phone:            {bool(raw_feats['has_phone'] > 0)}")
    print(f"  - Urgency Verbs:        {int(raw_feats['urgency_count'])}")
    print(f"  - Credential Keywords:  {int(raw_feats['sensitive_info_count'])}")
    print(f"  - Refund Keywords:      {int(raw_feats['refund_scam_count'])}")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        predict_text(msg)
    else:
        print('Please provide a text string: python interactive_predict.py "SMS text"')
