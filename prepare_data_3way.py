# -*- coding: utf-8 -*-
"""
Prepares the final 3-Way datasets:
- Train: Real Train + Multi-Phase Synthetic Augmentation (properly mapped to 3-Way taxonomy)
- Val: 100% REAL (Untouched holdout, 0 synthetic)
- Test: 100% REAL (Untouched blind holdout, 0 synthetic)
"""
import os
import json
import pandas as pd
import numpy as np
from preprocessing import clean_and_featurize, NUMERIC_FEATURES, LABEL_TO_ID
from split_isolation import standardize_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset_5971")
SPLIT_DIR = os.path.join(BASE_DIR, "prepared_3way")

def map_synthetic_source_label(row_label: str, source_filename: str) -> str:
    lbl_clean = str(row_label).strip().lower()
    if lbl_clean == "ham":
        return "HAM"
    
    # Check if from fraud/smishing augmentation batch
    if "fraud" in source_filename.lower() or "smishing" in source_filename.lower():
        return "SMISHING"
    
    if lbl_clean in ("smishing", "phishing", "fraud"):
        return "SMISHING"
    
    return "MARKETING_SPAM"

def featurize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    features_list = []
    for t in df["TEXT"]:
        c, fts = clean_and_featurize(t)
        cleaned.append(c)
        features_list.append(fts)
    df["clean_text"] = cleaned
    feat_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    df["label_id"] = df["LABEL_3WAY"].map(LABEL_TO_ID)
    return df

def prepare_3way_datasets(cap_per_class_synth: int = 15000):
    # 1. Load Real Partitions
    real_train = pd.read_csv(os.path.join(SPLIT_DIR, "real_train.csv"), encoding="utf-8-sig")
    real_val = pd.read_csv(os.path.join(SPLIT_DIR, "real_val.csv"), encoding="utf-8-sig")
    real_test = pd.read_csv(os.path.join(SPLIT_DIR, "real_test.csv"), encoding="utf-8-sig")

    print(f"Loaded Real Splits -> Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")

    # 2. Load Synthetic Augmentations for Training ONLY
    synth_frames = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("Synthetic_") and fname.endswith(".csv"):
            fpath = os.path.join(DATA_DIR, fname)
            sdf = pd.read_csv(fpath, encoding="utf-8-sig")
            sdf = sdf[["LABEL", "TEXT"]].copy()
            sdf["LABEL_3WAY"] = sdf.apply(lambda r: map_synthetic_source_label(r["LABEL"], fname), axis=1)
            sdf["is_synthetic"] = True
            sdf["source"] = fname
            synth_frames.append(sdf)

    full_synth = pd.concat(synth_frames, ignore_index=True)
    full_synth = full_synth.drop_duplicates(subset=["TEXT"]).reset_index(drop=True)

    # Cap synthetic samples per class to maintain high representation
    capped_synth = []
    np.random.seed(42)
    for lbl in ["HAM", "MARKETING_SPAM", "SMISHING"]:
        pool = full_synth[full_synth["LABEL_3WAY"] == lbl]
        n_sample = min(len(pool), cap_per_class_synth)
        sampled = pool.sample(n=n_sample, random_state=42)
        capped_synth.append(sampled)

    final_train = pd.concat([real_train] + capped_synth, ignore_index=True)
    final_train = final_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Final Train Set (Real + Synth): {len(final_train)} rows.")
    print(f"  Distribution: {final_train['LABEL_3WAY'].value_counts().to_dict()}")

    # Featurize all splits
    print("Featurizing Train, Val, Test splits...")
    train_feat = featurize_dataframe(final_train)
    val_feat = featurize_dataframe(real_val)
    test_feat = featurize_dataframe(real_test)

    # Save
    train_feat.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    val_feat.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False, encoding="utf-8-sig")
    test_feat.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "train_rows": len(train_feat),
        "val_rows": len(val_feat),
        "test_rows": len(test_feat),
        "val_is_100pct_real": bool((val_feat["is_synthetic"] == False).all()),
        "test_is_100pct_real": bool((test_feat["is_synthetic"] == False).all()),
        "train_distribution": train_feat["LABEL_3WAY"].value_counts().to_dict(),
        "val_distribution": val_feat["LABEL_3WAY"].value_counts().to_dict(),
        "test_distribution": test_feat["LABEL_3WAY"].value_counts().to_dict()
    }

    with open(os.path.join(SPLIT_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Prepared 3-way datasets saved successfully. Summary:", summary)
    return summary

if __name__ == "__main__":
    prepare_3way_datasets()
