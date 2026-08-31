# -*- coding: utf-8 -*-
"""
Consolidates all data sources (original real dataset + synthetic multilingual
sets + legitimate-URL ham augmentation), maps 3-way labels to binary
Ham/Spam, caps the synthetic contribution per language so the real 5,971
messages aren't drowned out, engineers numeric features, cleans/normalizes
text (URL/phone placeholder substitution so the model can't just memorize
random URLs/phone numbers), and produces a stratified train/val/test split.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import clean_and_featurize, NUMERIC_FEATURES  # noqa: F401 (re-exported)

random_state = 42
np.random.seed(random_state)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared")
os.makedirs(OUT_DIR, exist_ok=True)

CAP_HAM = 20000
CAP_SPAM = 20000


def load_source(path, lang):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[["LABEL", "TEXT"]].copy()
    df["lang"] = lang
    return df


def _log(msg):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_data_progress.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    _log("start main")
    sources = []
    sources.append((os.path.join(DATA_DIR, "Dataset_5971.csv"), "original"))
    always_include_sources = []
    for lang in ["English", "Hinglish", "Hindi", "Marathi"]:
        sources.append((os.path.join(DATA_DIR, f"Synthetic_{lang}.csv"), lang.lower()))
        sources.append((os.path.join(DATA_DIR, f"Synthetic_Ham_URL_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Extra_Ham_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Extra_SoftSpam_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Fraud_Ham_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Fraud_Spam_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Fraud2_Ham_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_Fraud2_Spam_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_PromoLegit_Ham_{lang}.csv"), lang.lower()))
        always_include_sources.append((os.path.join(DATA_DIR, f"Synthetic_PromoLegit_SpamContrast_{lang}.csv"), lang.lower()))

    frames = [load_source(p, lang) for p, lang in sources]
    _log(f"loaded {len(frames)} frames")
    df = pd.concat(frames, ignore_index=True)
    df["LABEL"] = df["LABEL"].str.strip()
    df["binary_label"] = df["LABEL"].apply(lambda x: 0 if x.lower() == "ham" else 1)
    _log(f"concatenated total {len(df)}")

    df = df.drop_duplicates(subset=["TEXT"]).reset_index(drop=True)
    _log(f"deduped total {len(df)}")

    always_frames = [load_source(p, lang) for p, lang in always_include_sources]
    always_df = pd.concat(always_frames, ignore_index=True)
    always_df["LABEL"] = always_df["LABEL"].str.strip()
    always_df["binary_label"] = always_df["LABEL"].apply(lambda x: 0 if x.lower() == "ham" else 1)
    always_df = always_df.drop_duplicates(subset=["TEXT"]).reset_index(drop=True)
    _log(f"always_include total {len(always_df)}")

    # Cap the synthetic contribution per language so the model doesn't just
    # memorize a handful of templates; keep the entire original real corpus.
    original_mask = df["lang"] == "original"
    original_df = df[original_mask]
    synth_df = df[~original_mask]

    _log(f"original_df={len(original_df)} synth_df={len(synth_df)}")
    capped_parts = [original_df, always_df]
    for lang in ["english", "hinglish", "hindi", "marathi"]:
        lang_df = synth_df[synth_df["lang"] == lang]
        for label_val, cap in [(0, CAP_HAM), (1, CAP_SPAM)]:
            pool = lang_df[lang_df["binary_label"] == label_val]
            n = min(cap, len(pool))
            sampled = pool.sample(n=n, random_state=random_state)
            capped_parts.append(sampled)
            _log(f"capped lang={lang} label={label_val} pool={len(pool)} sampled={n}")

    final_df = pd.concat(capped_parts, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    _log(f"capped+shuffled total {len(final_df)}")

    # Clean text + engineer numeric features
    cleaned_texts = []
    feature_rows = []
    for idx, t in enumerate(final_df["TEXT"]):
        c, feats = clean_and_featurize(t)
        cleaned_texts.append(c)
        feature_rows.append(feats)
        if idx % 20000 == 0:
            _log(f"featurized {idx}")
    _log("featurization done")

    final_df["clean_text"] = cleaned_texts
    feat_df = pd.DataFrame(feature_rows)
    final_df = pd.concat([final_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # Stratify by label x language combo
    final_df["strata"] = final_df["binary_label"].astype(str) + "_" + final_df["lang"]

    train_df, temp_df = train_test_split(
        final_df, test_size=0.20, random_state=random_state, stratify=final_df["strata"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_df["strata"]
    )

    _log("split done, writing csvs")
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")
    _log("csvs written")

    summary = {
        "total_rows_before_cap": int(len(df)),
        "total_rows_after_cap": int(len(final_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_counts_overall": final_df["binary_label"].value_counts().to_dict(),
        "lang_counts_overall": final_df["lang"].value_counts().to_dict(),
        "label_counts_train": train_df["binary_label"].value_counts().to_dict(),
        "label_counts_val": val_df["binary_label"].value_counts().to_dict(),
        "label_counts_test": test_df["binary_label"].value_counts().to_dict(),
    }
    with open(os.path.join(OUT_DIR, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    import traceback
    try:
        main()
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_data_status.txt"), "w", encoding="utf-8") as f:
            f.write("SUCCESS\n")
    except Exception:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_data_status.txt"), "w", encoding="utf-8") as f:
            f.write("FAILED\n")
            f.write(traceback.format_exc())
