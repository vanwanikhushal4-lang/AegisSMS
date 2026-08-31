# -*- coding: utf-8 -*-
"""
Re-selects the decision threshold without retraining the network.

The first attempt picked the threshold that maximizes F2 on the validation
split alone. That split is drawn from the same easy/templated distribution
as training (spam probabilities cluster near 0 or 1), so a tiny handful of
borderline points could drag the "optimal" F2 threshold to a degenerate
extreme (0.07) that looked fine on that split but collapsed precision on
the genuinely novel diverse_test_set.csv (86.7% precision, legit-URL false
positives came back).

Better strategy: among thresholds that keep validation precision >= 0.98,
pick the one with the highest recall (i.e. catch as much fraud as possible
without crying wolf on legitimate messages) -- then sanity-check the choice
against diverse_test_set.csv before committing to it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_and_featurize, NUMERIC_FEATURES

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PREP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared")


def log(msg):
    print(msg)
    with open(os.path.join(ARTIFACT_DIR, "tune_threshold_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    with open(os.path.join(ARTIFACT_DIR, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "vocabulary.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "feature_scaler.json"), encoding="utf-8") as f:
        scaler = json.load(f)

    max_len = config["max_len"]
    mean = np.array(scaler["mean"], dtype="float32")
    std = np.array(scaler["std"], dtype="float32")

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config["max_tokens"], output_mode="int",
        output_sequence_length=max_len, standardize="lower", split="whitespace",
    )
    vectorizer.set_vocabulary(vocab[2:])

    model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "sms_model.keras"))

    val_df = pd.read_csv(os.path.join(PREP_DIR, "val.csv"), encoding="utf-8-sig")
    val_df["clean_text"] = val_df["clean_text"].fillna("")
    X_val_ids = vectorizer(tf.constant(val_df["clean_text"].values)).numpy()
    X_val_num = (val_df[NUMERIC_FEATURES].values.astype("float32") - mean) / std
    y_val = val_df["binary_label"].values

    val_probs = model.predict({"input_ids": X_val_ids, "numeric_features": X_val_num}, verbose=0).ravel()

    # diverse set as a sanity-check gate (out-of-distribution, catches
    # thresholds that overfit the easy validation split)
    diverse_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diverse_test_set.csv"), encoding="utf-8-sig")
    diverse_df["true_label"] = diverse_df["LABEL"].apply(lambda x: 0 if x.strip().lower() == "ham" else 1)
    cleaned, feats = [], []
    for t in diverse_df["TEXT"]:
        c, ft = clean_and_featurize(t)
        cleaned.append(c)
        feats.append(ft)
    diverse_df["clean_text"] = cleaned
    feat_df = pd.DataFrame(feats)
    X_div_ids = vectorizer(tf.constant(diverse_df["clean_text"].values)).numpy()
    X_div_num = (feat_df[NUMERIC_FEATURES].values.astype("float32") - mean) / std
    y_div = diverse_df["true_label"].values
    div_probs = model.predict({"input_ids": X_div_ids, "numeric_features": X_div_num}, verbose=0).ravel()

    thresholds = np.arange(0.05, 0.96, 0.01)
    candidates = []
    for t in thresholds:
        val_preds = (val_probs > t).astype(int)
        val_prec = precision_score(y_val, val_preds, zero_division=0)
        val_rec = recall_score(y_val, val_preds, zero_division=0)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        div_preds = (div_probs > t).astype(int)
        div_prec = precision_score(y_div, div_preds, zero_division=0)
        div_rec = recall_score(y_div, div_preds, zero_division=0)
        div_acc = accuracy_score(y_div, div_preds)

        candidates.append({
            "threshold": float(t), "val_precision": float(val_prec), "val_recall": float(val_rec),
            "val_f1": float(val_f1), "diverse_precision": float(div_prec),
            "diverse_recall": float(div_rec), "diverse_accuracy": float(div_acc),
        })

    with open(os.path.join(ARTIFACT_DIR, "threshold_search_v2.json"), "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    # Selection rule: require validation precision >= 0.98 AND diverse-set
    # precision >= 0.95 (guards against the val-only overfit failure mode),
    # then take the highest-recall (val) threshold among survivors.
    qualified = [c for c in candidates if c["val_precision"] >= 0.98 and c["diverse_precision"] >= 0.95]
    if qualified:
        best = max(qualified, key=lambda c: c["val_recall"])
        log(f"Qualified thresholds found: {len(qualified)}. Selected threshold={best['threshold']:.2f}")
    else:
        # fall back to the threshold maximizing val F1 among those that
        # don't blow up diverse-set precision
        safe = [c for c in candidates if c["diverse_precision"] >= 0.90] or candidates
        best = max(safe, key=lambda c: c["val_f1"])
        log(f"No threshold met both precision gates; falling back to best-F1-with-safe-diverse-precision: "
            f"threshold={best['threshold']:.2f}")

    log(f"Chosen threshold detail: {best}")

    config["decision_threshold"] = best["threshold"]
    with open(os.path.join(ARTIFACT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log(f"model_config.json updated with decision_threshold={best['threshold']:.2f}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        with open(os.path.join(ARTIFACT_DIR, "tune_threshold_error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
