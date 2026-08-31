# -*- coding: utf-8 -*-
"""
Evaluates the trained model (both the Keras .keras model and the exported
.tflite model, to confirm they agree) on a hand-authored test CSV
(TEXT,LABEL,CATEGORY,LANG columns), with a specific check that legitimate
messages containing URLs are not flagged spam.

Usage:
    python evaluate_diverse.py                                   # diverse_test_set.csv
    python evaluate_diverse.py challenge_test_set.csv challenge  # any other test CSV
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_and_featurize, NUMERIC_FEATURES

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(test_path, output_prefix):
    with open(os.path.join(ARTIFACT_DIR, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "vocabulary.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "feature_scaler.json"), encoding="utf-8") as f:
        scaler = json.load(f)

    max_len = config["max_len"]
    threshold = config.get("decision_threshold", 0.5)
    mean = np.array(scaler["mean"], dtype="float32")
    std = np.array(scaler["std"], dtype="float32")

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config["max_tokens"], output_mode="int",
        output_sequence_length=max_len, standardize="lower", split="whitespace",
    )
    vectorizer.set_vocabulary(vocab[2:])  # first two entries are '' and '[UNK]', set_vocabulary expects without them

    df = pd.read_csv(test_path, encoding="utf-8-sig")
    df["true_label"] = df["LABEL"].apply(lambda x: 0 if x.strip().lower() == "ham" else 1)

    cleaned, feats = [], []
    for t in df["TEXT"]:
        c, ft = clean_and_featurize(t)
        cleaned.append(c)
        feats.append(ft)
    df["clean_text"] = cleaned
    feat_df = pd.DataFrame(feats)

    X_ids = vectorizer(tf.constant(df["clean_text"].values)).numpy()
    X_num = ((feat_df[NUMERIC_FEATURES].values.astype("float32") - mean) / std)

    model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "sms_model.keras"))
    probs = model.predict({"input_ids": X_ids, "numeric_features": X_num}, verbose=0).ravel()
    preds = (probs > threshold).astype(int)

    df["pred_label"] = preds
    df["pred_prob_spam"] = probs

    y_true = df["true_label"].values
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds).tolist()
    report = classification_report(y_true, preds, target_names=["Ham", "Spam"], output_dict=True, zero_division=0)

    # per-category breakdown
    cat_results = {}
    for cat, group in df.groupby("CATEGORY"):
        cat_acc = accuracy_score(group["true_label"], group["pred_label"])
        cat_results[cat] = {"n": int(len(group)), "accuracy": float(cat_acc)}

    # per-language breakdown
    lang_results = {}
    for lang, group in df.groupby("LANG"):
        lang_acc = accuracy_score(group["true_label"], group["pred_label"])
        lang_results[lang] = {"n": int(len(group)), "accuracy": float(lang_acc)}

    # Specific check: ham messages containing a legitimate URL incorrectly flagged as spam
    legit_url_rows = df[df["CATEGORY"] == "legit_url"]
    legit_url_false_positive_rate = None
    legit_url_flagged = pd.DataFrame(columns=["TEXT", "pred_prob_spam"])
    if len(legit_url_rows) > 0:
        legit_url_false_positive_rate = float((legit_url_rows["pred_label"] == 1).mean())
        legit_url_flagged = legit_url_rows[legit_url_rows["pred_label"] == 1][["TEXT", "pred_prob_spam"]]

    # every row still marked wrong, for quick inspection
    misclassified = df[df["true_label"] != df["pred_label"]][["TEXT", "LABEL", "CATEGORY", "LANG", "pred_label", "pred_prob_spam"]]

    # TFLite cross-check
    tflite_path = os.path.join(ARTIFACT_DIR, "sms_spam_model.tflite")
    tflite_agrees = None
    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        tflite_preds = []
        # run one-by-one (batch size 1) since TFLite default export is unbatched/dynamic
        for i in range(len(df)):
            ids_in = X_ids[i:i + 1].astype(np.int32)
            num_in = X_num[i:i + 1].astype(np.float32)
            for d in input_details:
                if "input_ids" in d["name"] or d["shape"][-1] == max_len:
                    interpreter.resize_tensor_input(d["index"], ids_in.shape)
                else:
                    interpreter.resize_tensor_input(d["index"], num_in.shape)
            interpreter.allocate_tensors()
            for d in input_details:
                if "input_ids" in d["name"] or d["shape"][-1] == max_len:
                    interpreter.set_tensor(d["index"], ids_in)
                else:
                    interpreter.set_tensor(d["index"], num_in)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            tflite_preds.append(int(out.ravel()[0] > threshold))
        tflite_preds = np.array(tflite_preds)
        tflite_agrees = float((tflite_preds == preds).mean())

    results = {
        "test_path": test_path, "n_samples": int(len(df)), "decision_threshold": threshold,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "classification_report": report,
        "category_breakdown": cat_results, "language_breakdown": lang_results,
        "legit_url_false_positive_rate": legit_url_false_positive_rate,
        "legit_url_flagged_as_spam": legit_url_flagged.to_dict(orient="records"),
        "tflite_agreement_with_keras": tflite_agrees,
    }
    with open(os.path.join(ARTIFACT_DIR, f"{output_prefix}_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    df.to_csv(os.path.join(ARTIFACT_DIR, f"{output_prefix}_eval_predictions.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(ARTIFACT_DIR, f"{output_prefix}_eval_status.txt"), "w", encoding="utf-8") as f:
        f.write(f"test_path={test_path}\n")
        f.write(f"decision_threshold={threshold}\n")
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"precision={prec:.4f}\n")
        f.write(f"recall={rec:.4f}\n")
        f.write(f"f1={f1:.4f}\n")
        f.write(f"confusion_matrix={cm}\n")
        f.write(f"legit_url_false_positive_rate={legit_url_false_positive_rate}\n")
        f.write(f"tflite_agreement={tflite_agrees}\n")
        f.write(f"misclassified_count={len(misclassified)}\n")
        for _, row in misclassified.iterrows():
            f.write(f"  WRONG true={row['LABEL']} pred={'Spam' if row['pred_label'] == 1 else 'Ham'} "
                    f"prob={row['pred_prob_spam']:.4f} cat={row['CATEGORY']} lang={row['LANG']}: {row['TEXT']}\n")
        f.write("DONE\n")


if __name__ == "__main__":
    import traceback

    test_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "diverse_test_set.csv")
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "diverse"
    try:
        evaluate(test_path, output_prefix)
    except Exception:
        with open(os.path.join(ARTIFACT_DIR, f"{output_prefix}_eval_error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
