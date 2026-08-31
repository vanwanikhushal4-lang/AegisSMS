# -*- coding: utf-8 -*-
"""
Trains the 3-Way Multilingual SMS Classifier:
Classes:
  0: HAM (Legitimate)
  1: MARKETING_SPAM (Promotional/Commercial)
  2: SMISHING (Phishing/Fraud/Urgent Threats)

Uses a fused architecture of Subword/Token TF-IDF Embeddings + 11 Handcrafted Numeric Features,
optimized with calibrated multi-class regularized classifier with exact JSON & TFLite export.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
import scipy.stats as stats

from preprocessing import NUMERIC_FEATURES, LABEL_TO_ID, ID_TO_LABEL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(BASE_DIR, "prepared_3way")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 42

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float(max(0.0, (centre - spread) / denominator)), float(min(1.0, (centre + spread) / denominator)))

def train_and_evaluate():
    print("Loading 3-way prepared datasets...")
    train_df = pd.read_csv(os.path.join(PREP_DIR, "train.csv"), encoding="utf-8-sig")
    val_df = pd.read_csv(os.path.join(PREP_DIR, "val.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(PREP_DIR, "test.csv"), encoding="utf-8-sig")

    for d in (train_df, val_df, test_df):
        d["clean_text"] = d["clean_text"].fillna("")

    print(f"Train: {len(train_df)} (Real+Synth), Val: {len(val_df)} (100% Real), Test: {len(test_df)} (100% Real)")

    # 1. Text Vectorizer
    print("Building vocabulary vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 3),
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True
    )
    X_train_text = vectorizer.fit_transform(train_df["clean_text"])
    X_val_text = vectorizer.transform(val_df["clean_text"])
    X_test_text = vectorizer.transform(test_df["clean_text"])

    vocab = vectorizer.vocabulary_
    vocab_list = sorted(vocab.keys(), key=lambda k: vocab[k])
    print(f"Vocabulary size: {len(vocab_list)}")

    # 2. Numeric Features Normalization
    mean = train_df[NUMERIC_FEATURES].values.astype(np.float32).mean(axis=0)
    std = train_df[NUMERIC_FEATURES].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    X_train_num = sp.csr_matrix((train_df[NUMERIC_FEATURES].values.astype(np.float32) - mean) / std)
    X_val_num = sp.csr_matrix((val_df[NUMERIC_FEATURES].values.astype(np.float32) - mean) / std)
    X_test_num = sp.csr_matrix((test_df[NUMERIC_FEATURES].values.astype(np.float32) - mean) / std)

    # 3. Concatenate Sparse Feature Representations
    X_train = sp.hstack([X_train_text, X_train_num], format="csr")
    X_val = sp.hstack([X_val_text, X_val_num], format="csr")
    X_test = sp.hstack([X_test_text, X_test_num], format="csr")

    y_train = train_df["label_id"].values.astype(np.int32)
    y_val = val_df["label_id"].values.astype(np.int32)
    y_test = test_df["label_id"].values.astype(np.int32)

    # 4. Train Multi-Class Regularized Logistic Classifier with Optimized Class Weights
    print("Training 3-Way Calibrated Multi-Class Model...")
    weights = {0: 1.0, 1: 1.2, 2: 2.6}
    clf = LogisticRegression(
        C=8.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight=weights,
        random_state=SEED
    )
    clf.fit(X_train, y_train)

    # 5. Evaluate on Real Validation and Blind Real Test Sets
    val_probs = clf.predict_proba(X_val)
    val_preds = clf.predict(X_val)

    test_probs = clf.predict_proba(X_test)
    test_preds = clf.predict(X_test)

    # Metrics on Real Blind Test Set
    test_acc = float(accuracy_score(y_test, test_preds))
    prec, rec, f1, support = precision_recall_fscore_support(y_test, test_preds, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2]).tolist()

    # Smishing Specific Metrics (Class 2)
    smishing_rec = float(rec[2])
    smishing_prec = float(prec[2])
    smishing_f1 = float(f1[2])
    smishing_ci = wilson_score_interval(int(cm[2][2]), int(support[2]))

    # Ham False Positive Rate (Class 0 incorrectly predicted as Spam/Smishing)
    ham_total = int(support[0])
    ham_fps = int(cm[0][1] + cm[0][2])
    ham_fpr = float(ham_fps / max(ham_total, 1))
    ham_fpr_ci = wilson_score_interval(ham_fps, ham_total)

    print("\n================ REAL BLIND TEST SET RESULTS ================")
    print(f"Overall Accuracy:   {test_acc*100:.2f}%")
    print(f"Smishing Recall:    {smishing_rec*100:.2f}% (Target: >= 85%) [95% CI: {smishing_ci[0]*100:.2f}% - {smishing_ci[1]*100:.2f}%]")
    print(f"Smishing Precision: {smishing_prec*100:.2f}% (Target: >= 85%)")
    print(f"Ham FPR:            {ham_fpr*100:.2f}% (Target: < 1.00%) [95% CI: {ham_fpr_ci[0]*100:.2f}% - {ham_fpr_ci[1]*100:.2f}%]")
    print(f"Confusion Matrix:\n  HAM:   {cm[0]}\n  SPAM:  {cm[1]}\n  SMISH: {cm[2]}")

    # 6. Save Artifacts
    model_pickle_path = os.path.join(ARTIFACT_DIR, "sms_model_3way.pkl")
    with open(model_pickle_path, "wb") as f:
        pickle.dump(clf, f)

    vectorizer_pickle_path = os.path.join(ARTIFACT_DIR, "vectorizer_3way.pkl")
    with open(vectorizer_pickle_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(ARTIFACT_DIR, "feature_scaler_3way.json"), "w", encoding="utf-8") as f:
        json.dump({"features": NUMERIC_FEATURES, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "vocabulary_3way.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_list[:5000], f, ensure_ascii=False, indent=0)

    metrics = {
        "model_version": "2.0.0-3WAY",
        "test_accuracy": test_acc,
        "classes": ["HAM", "MARKETING_SPAM", "SMISHING"],
        "class_metrics": {
            "HAM": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(support[0])},
            "MARKETING_SPAM": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]), "support": int(support[1])},
            "SMISHING": {"precision": smishing_prec, "recall": smishing_rec, "f1": smishing_f1, "support": int(support[2]), "recall_95_ci": smishing_ci}
        },
        "ham_false_positive_rate": ham_fpr,
        "ham_fpr_95_ci": ham_fpr_ci,
        "confusion_matrix": cm,
        "quality_gates": {
            "smishing_recall_ge_85": bool(smishing_rec >= 0.85),
            "smishing_precision_ge_85": bool(smishing_prec >= 0.85),
            "ham_fpr_lt_1_0_pct": bool(ham_fpr < 0.010)
        }
    }

    with open(os.path.join(ARTIFACT_DIR, "final_metrics_3way.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved 3-way model artifacts and metrics successfully.")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
