# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from preprocessing import NUMERIC_FEATURES, clean_and_featurize

def test_gate2_live_model_metrics_execution():
    artifact_dir = os.path.join(BASE_DIR, "artifacts")
    prep_dir = os.path.join(BASE_DIR, "prepared_4way_p5")

    with open(os.path.join(artifact_dir, "sms_model_3way.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(artifact_dir, "vectorizer_3way.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(artifact_dir, "aegis_model_contract.json"), "r", encoding="utf-8") as f:
        contract = json.load(f)

    mean = np.array(contract["feature_normalizer"]["mean"], dtype=np.float32)
    std = np.array(contract["feature_normalizer"]["std"], dtype=np.float32)
    thresh = contract["is_scam_operating_threshold"]

    test_df = pd.read_csv(os.path.join(prep_dir, "test.csv"), encoding="utf-8-sig")
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    x_t = vectorizer.transform(test_df["clean_text"])
    x_num = (test_df[NUMERIC_FEATURES].values.astype(np.float32) - mean) / std
    x_fused = sp.hstack([x_t, sp.csr_matrix(x_num)], format="csr")

    probs = clf.predict_proba(x_fused)
    preds = np.argmax(probs, axis=1)
    y_true = test_df["label_id"].values.astype(np.int32)

    scam_pred = (probs[:, 3] >= thresh)
    scam_true = (y_true == 3)
    legit_true = (y_true != 3)

    # Compute live metrics
    tp = np.sum(scam_pred & scam_true)
    fn = np.sum((~scam_pred) & scam_true)
    fp = np.sum(scam_pred & legit_true)
    tn = np.sum((~scam_pred) & legit_true)

    scam_rec = tp / max(tp + fn, 1)
    scam_prec = tp / max(tp + fp, 1)
    legit_fpr = fp / max(fp + tn, 1)
    overall_acc = np.mean(preds == y_true)

    print(f"\nLIVE CI TEST METRICS: Acc={overall_acc*100:.2f}%, ScamRec={scam_rec*100:.2f}%, ScamPrec={scam_prec*100:.2f}%, LegitFPR={legit_fpr*100:.3f}%")

    assert overall_acc >= 0.90, f"Accuracy ({overall_acc:.4f}) below 90%"
    assert scam_rec >= 0.95, f"SCAM Recall ({scam_rec:.4f}) below 95%"
    assert scam_prec >= 0.95, f"SCAM Precision ({scam_prec:.4f}) below 95%"
    assert legit_fpr < 0.005, f"Legitimate-to-SCAM FPR ({legit_fpr*100:.3f}%) above 0.5%"
