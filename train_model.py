# -*- coding: utf-8 -*-
"""
Trains a multilingual (English/Hinglish/Hindi/Marathi) SMS Ham/Spam
classifier: a TextCNN over a word-level vocabulary (script-agnostic
whitespace tokenization works uniformly across Latin and Devanagari text)
plus a small branch of hand-engineered numeric features (URL trust class,
phone presence, urgency-keyword count, etc.), fused before the final
classification head. Saves the Keras model, TFLite export, and all
preprocessing artifacts (vocabulary, feature scaler, config).
"""
import json
import os
import time
import traceback

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import NUMERIC_FEATURES

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PREP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MAX_TOKENS = 20000


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(os.path.join(ARTIFACT_DIR, "train_progress.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data():
    train_df = pd.read_csv(os.path.join(PREP_DIR, "train.csv"), encoding="utf-8-sig")
    val_df = pd.read_csv(os.path.join(PREP_DIR, "val.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(PREP_DIR, "test.csv"), encoding="utf-8-sig")
    for d in (train_df, val_df, test_df):
        d["clean_text"] = d["clean_text"].fillna("")
    return train_df, val_df, test_df


def pick_max_len(train_df):
    lens = train_df["clean_text"].str.split().apply(len)
    p95 = int(np.percentile(lens, 95))
    return int(min(max(p95, 15), 50))


def build_vectorizer(train_texts, max_tokens, max_len):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower",  # only lowercases; keeps URL/punct structure & Devanagari intact
        split="whitespace",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))
    return vectorizer


def featurize_numeric(df, mean, std):
    X = df[NUMERIC_FEATURES].values.astype("float32")
    return (X - mean) / std


def build_model(vocab_size, max_len, n_numeric, embed_dim, filters, dropout, lr):
    text_in = tf.keras.Input(shape=(max_len,), dtype="int32", name="input_ids")
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, name="embedding")(text_in)
    x = tf.keras.layers.Conv1D(filters, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    num_in = tf.keras.Input(shape=(n_numeric,), dtype="float32", name="numeric_features")
    y = tf.keras.layers.Dense(16, activation="relu")(num_in)

    z = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(32, activation="relu")(z)
    z = tf.keras.layers.Dropout(dropout)(z)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(z)

    model = tf.keras.Model(inputs=[text_in, num_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    log("=== START ===")
    train_df, val_df, test_df = load_data()
    log(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    max_len = pick_max_len(train_df)
    log(f"max_len={max_len}")

    vectorizer = build_vectorizer(train_df["clean_text"].values, MAX_TOKENS, max_len)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    log(f"vocab_size={vocab_size}")

    X_train_ids = vectorizer(tf.constant(train_df["clean_text"].values)).numpy()
    X_val_ids = vectorizer(tf.constant(val_df["clean_text"].values)).numpy()
    X_test_ids = vectorizer(tf.constant(test_df["clean_text"].values)).numpy()

    mean = train_df[NUMERIC_FEATURES].values.astype("float32").mean(axis=0)
    std = train_df[NUMERIC_FEATURES].values.astype("float32").std(axis=0)
    std[std == 0] = 1.0

    X_train_num = featurize_numeric(train_df, mean, std)
    X_val_num = featurize_numeric(val_df, mean, std)
    X_test_num = featurize_numeric(test_df, mean, std)

    y_train = train_df["binary_label"].values.astype("float32")
    y_val = val_df["binary_label"].values.astype("float32")
    y_test = test_df["binary_label"].values.astype("float32")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
    log(f"class_weight={class_weight_dict}")

    # ---------------- lightweight hyperparameter search ----------------
    configs = [
        {"name": "A", "embed_dim": 64, "filters": 64, "dropout": 0.3, "lr": 1e-3},
        {"name": "B", "embed_dim": 128, "filters": 128, "dropout": 0.3, "lr": 1e-3},
        {"name": "C", "embed_dim": 64, "filters": 128, "dropout": 0.5, "lr": 5e-4},
        {"name": "D", "embed_dim": 128, "filters": 64, "dropout": 0.2, "lr": 1e-3},
    ]
    hp_results = []
    for cfg in configs:
        log(f"HP search config {cfg['name']}: {cfg}")
        m = build_model(vocab_size, max_len, len(NUMERIC_FEATURES),
                         cfg["embed_dim"], cfg["filters"], cfg["dropout"], cfg["lr"])
        hist = m.fit(
            {"input_ids": X_train_ids, "numeric_features": X_train_num}, y_train,
            validation_data=({"input_ids": X_val_ids, "numeric_features": X_val_num}, y_val),
            epochs=3, batch_size=512, class_weight=class_weight_dict, verbose=0,
        )
        val_acc = float(hist.history["val_accuracy"][-1])
        hp_results.append({"config": cfg, "val_accuracy": val_acc})
        log(f"config {cfg['name']} val_accuracy={val_acc:.4f}")

    best = max(hp_results, key=lambda r: r["val_accuracy"])
    best_cfg = best["config"]
    log(f"BEST config: {best_cfg} val_accuracy={best['val_accuracy']:.4f}")

    with open(os.path.join(ARTIFACT_DIR, "hp_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(hp_results, f, indent=2)

    # ---------------- 5-fold stratified cross-validation (best config) ----------------
    log("Starting 5-fold cross-validation with best config")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    X_ids_full = np.concatenate([X_train_ids, X_val_ids], axis=0)
    X_num_full = np.concatenate([X_train_num, X_val_num], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_ids_full, y_full)):
        m = build_model(vocab_size, max_len, len(NUMERIC_FEATURES),
                         best_cfg["embed_dim"], best_cfg["filters"], best_cfg["dropout"], best_cfg["lr"])
        cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_full[tr_idx])
        cw_dict = {0: float(cw[0]), 1: float(cw[1])}
        m.fit(
            {"input_ids": X_ids_full[tr_idx], "numeric_features": X_num_full[tr_idx]}, y_full[tr_idx],
            epochs=4, batch_size=512, class_weight=cw_dict, verbose=0,
        )
        preds = (m.predict({"input_ids": X_ids_full[te_idx], "numeric_features": X_num_full[te_idx]},
                            verbose=0) > 0.5).astype(int).ravel()
        acc = accuracy_score(y_full[te_idx], preds)
        f1 = f1_score(y_full[te_idx], preds)
        cv_scores.append({"fold": fold_i, "accuracy": float(acc), "f1": float(f1)})
        log(f"CV fold {fold_i}: accuracy={acc:.4f} f1={f1:.4f}")

    cv_acc_mean = float(np.mean([s["accuracy"] for s in cv_scores]))
    cv_acc_std = float(np.std([s["accuracy"] for s in cv_scores]))
    log(f"CV accuracy mean={cv_acc_mean:.4f} std={cv_acc_std:.4f}")
    with open(os.path.join(ARTIFACT_DIR, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": cv_scores, "mean_accuracy": cv_acc_mean, "std_accuracy": cv_acc_std}, f, indent=2)

    # ---------------- final training on full train set ----------------
    log("Final training with best config on train set, evaluating on val")
    final_model = build_model(vocab_size, max_len, len(NUMERIC_FEATURES),
                               best_cfg["embed_dim"], best_cfg["filters"], best_cfg["dropout"], best_cfg["lr"])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    final_model.fit(
        {"input_ids": X_train_ids, "numeric_features": X_train_num}, y_train,
        validation_data=({"input_ids": X_val_ids, "numeric_features": X_val_num}, y_val),
        epochs=20, batch_size=512, class_weight=class_weight_dict,
        callbacks=[early_stop], verbose=0,
    )

    # ---------------- decision threshold tuning ----------------
    # Missing a fraud/phishing message (false negative) is costlier than a
    # false alarm on a legitimate message, so recall is favored over the
    # naive 0.5 cutoff -- but maximizing F2 on the validation split ALONE is
    # unreliable: that split is drawn from the same easy/templated
    # distribution as training (probabilities cluster near 0 or 1), so a
    # handful of borderline points can drag the "optimal" F2 threshold to a
    # degenerate extreme that looks fine on val but collapses precision on
    # genuinely novel messages. Selection rule: among thresholds keeping
    # validation precision >= 0.98 AND diverse_test_set.csv (hand-authored,
    # out-of-distribution) precision >= 0.95, pick the highest-recall one;
    # otherwise fall back to the best F1 among thresholds that don't blow up
    # diverse-set precision.
    val_probs = final_model.predict({"input_ids": X_val_ids, "numeric_features": X_val_num}, verbose=0).ravel()

    diverse_path = os.path.join(os.path.dirname(ARTIFACT_DIR), "diverse_test_set.csv")
    div_probs = div_y = None
    if os.path.exists(diverse_path):
        from preprocessing import clean_and_featurize
        div_df = pd.read_csv(diverse_path, encoding="utf-8-sig")
        div_df["true_label"] = div_df["LABEL"].apply(lambda x: 0 if x.strip().lower() == "ham" else 1)
        div_clean, div_feats = [], []
        for t in div_df["TEXT"]:
            c, ft = clean_and_featurize(t)
            div_clean.append(c)
            div_feats.append(ft)
        div_ids = vectorizer(tf.constant(div_clean)).numpy()
        div_num = (pd.DataFrame(div_feats)[NUMERIC_FEATURES].values.astype("float32") - mean) / std
        div_probs = final_model.predict({"input_ids": div_ids, "numeric_features": div_num}, verbose=0).ravel()
        div_y = div_df["true_label"].values

    thresholds = np.arange(0.05, 0.96, 0.01)
    threshold_scores = []
    for t in thresholds:
        preds_t = (val_probs > t).astype(int)
        row = {
            "threshold": float(t),
            "val_precision": float(precision_score(y_val, preds_t, zero_division=0)),
            "val_recall": float(recall_score(y_val, preds_t, zero_division=0)),
            "val_f1": float(f1_score(y_val, preds_t, zero_division=0)),
        }
        if div_probs is not None:
            div_preds_t = (div_probs > t).astype(int)
            row["diverse_precision"] = float(precision_score(div_y, div_preds_t, zero_division=0))
            row["diverse_recall"] = float(recall_score(div_y, div_preds_t, zero_division=0))
        threshold_scores.append(row)

    if div_probs is not None:
        qualified = [r for r in threshold_scores if r["val_precision"] >= 0.98 and r["diverse_precision"] >= 0.95]
        if qualified:
            best = max(qualified, key=lambda r: r["val_recall"])
        else:
            safe = [r for r in threshold_scores if r["diverse_precision"] >= 0.90] or threshold_scores
            best = max(safe, key=lambda r: r["val_f1"])
    else:
        qualified = [r for r in threshold_scores if r["val_precision"] >= 0.98]
        best = max(qualified or threshold_scores, key=lambda r: r["val_recall"] if qualified else r["val_f1"])
    best_threshold = best["threshold"]
    log(f"Selected decision threshold={best_threshold:.2f} detail={best}")

    with open(os.path.join(ARTIFACT_DIR, "threshold_search.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": threshold_scores, "selected_threshold": best_threshold}, f, indent=2)

    # ---------------- test evaluation (default 0.5 AND tuned threshold) ----------------
    test_probs = final_model.predict({"input_ids": X_test_ids, "numeric_features": X_test_num}, verbose=0).ravel()

    def eval_at(threshold):
        preds = (test_probs > threshold).astype(int)
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }, preds

    metrics_default, _ = eval_at(0.5)
    metrics_tuned, test_preds = eval_at(best_threshold)
    report = classification_report(y_test, test_preds, target_names=["Ham", "Spam"], output_dict=True)

    log(f"TEST @0.5    accuracy={metrics_default['accuracy']:.4f} precision={metrics_default['precision']:.4f} "
        f"recall={metrics_default['recall']:.4f} f1={metrics_default['f1']:.4f}")
    log(f"TEST @tuned({best_threshold:.2f}) accuracy={metrics_tuned['accuracy']:.4f} "
        f"precision={metrics_tuned['precision']:.4f} recall={metrics_tuned['recall']:.4f} f1={metrics_tuned['f1']:.4f}")
    log(f"confusion_matrix (tuned)={metrics_tuned['confusion_matrix']}")

    metrics = {
        "test_accuracy": metrics_tuned["accuracy"], "test_precision": metrics_tuned["precision"],
        "test_recall": metrics_tuned["recall"], "test_f1": metrics_tuned["f1"],
        "confusion_matrix": metrics_tuned["confusion_matrix"], "classification_report": report,
        "cv_accuracy_mean": cv_acc_mean, "cv_accuracy_std": cv_acc_std,
        "best_hyperparams": best_cfg, "decision_threshold": best_threshold,
        "metrics_at_default_0.5_threshold": metrics_default,
        "metrics_at_tuned_threshold": metrics_tuned,
    }
    with open(os.path.join(ARTIFACT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- save misclassified samples for analysis ----------------
    mis_idx = np.where(test_preds != y_test)[0]
    mis_df = test_df.iloc[mis_idx].copy()
    mis_df["pred"] = test_preds[mis_idx]
    mis_df["prob"] = test_probs[mis_idx]
    mis_df.to_csv(os.path.join(ARTIFACT_DIR, "misclassified_test.csv"), index=False, encoding="utf-8-sig")
    log(f"misclassified count = {len(mis_idx)}")

    # ---------------- save artifacts ----------------
    final_model.save(os.path.join(ARTIFACT_DIR, "sms_model.keras"))
    with open(os.path.join(ARTIFACT_DIR, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    with open(os.path.join(ARTIFACT_DIR, "feature_scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"features": NUMERIC_FEATURES, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "max_len": max_len, "max_tokens": MAX_TOKENS, "vocab_size": vocab_size,
            "numeric_features": NUMERIC_FEATURES, "hyperparams": best_cfg, "seed": SEED,
            "decision_threshold": best_threshold,
        }, f, indent=2)

    # ---------------- TFLite conversion ----------------
    log("Converting to TFLite")
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(ARTIFACT_DIR, "sms_spam_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    log(f"TFLite model saved: {tflite_path} ({len(tflite_model)} bytes)")

    log("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(os.path.join(ARTIFACT_DIR, "train_error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        log("FAILED - see train_error.txt")
