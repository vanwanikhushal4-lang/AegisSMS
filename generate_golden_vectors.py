# -*- coding: utf-8 -*-
import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

from preprocessing import clean_and_featurize, NUMERIC_FEATURES, ID_TO_LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
PREP_DIR = os.path.join(BASE_DIR, "prepared_4way_scam")

def generate_golden_dataset(output_path = os.path.join(ARTIFACT_DIR, "golden_parity_1000.json")):
    with open(os.path.join(ARTIFACT_DIR, "sms_model_3way.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "vectorizer_3way.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "feature_scaler_3way.json"), "r") as f:
        scaler = json.load(f)

    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    test_df = pd.read_csv(os.path.join(PREP_DIR, "test.csv"), encoding="utf-8-sig")
    train_df = pd.read_csv(os.path.join(PREP_DIR, "train.csv"), encoding="utf-8-sig")

    sample_pool = pd.concat([test_df, train_df.sample(n=300, random_state=42)], ignore_index=True)
    sample_pool = sample_pool.drop_duplicates(subset=["text"]).head(1000).reset_index(drop=True)

    records = []
    for idx, row in sample_pool.iterrows():
        raw_text = str(row["text"])
        cleaned, raw_feats = clean_and_featurize(raw_text)

        x_text = vectorizer.transform([cleaned])
        raw_num = np.array([raw_feats[k] for k in NUMERIC_FEATURES], dtype=np.float32)
        scaled_num = (raw_num - mean) / std
        x_num = sp.csr_matrix(scaled_num.reshape(1, -1))
        x_fused = sp.hstack([x_text, x_num], format="csr")

        probs = clf.predict_proba(x_fused)[0]
        pred_id = int(np.argmax(probs))

        records.append({
            "vector_id": f"GOLDEN_{idx+1:04d}",
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "ground_truth_category": str(row.get("category", "UNKNOWN")).upper(),
            "predicted_category": ID_TO_LABEL[pred_id],
            "raw_numeric_features": {k: float(raw_feats[k]) for k in NUMERIC_FEATURES},
            "scaled_numeric_features": {k: float(scaled_num[i]) for i, k in enumerate(NUMERIC_FEATURES)},
            "probabilities": {
                "PERSONAL": float(round(probs[0], 6)),
                "TRANSACTIONAL": float(round(probs[1], 6)),
                "PROMOTIONAL": float(round(probs[2], 6)),
                "SCAM": float(round(probs[3], 6))
            },
            "max_parity_delta": 0.0
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} golden parity vectors at {output_path}")
    return records

if __name__ == "__main__":
    generate_golden_dataset()
