# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP_DIR = os.path.join(BASE_DIR, "prepared_4way_scam")

def test_gate1_zero_data_leakage():
    train_df = pd.read_csv(os.path.join(PREP_DIR, "train.csv"), encoding="utf-8-sig")
    val_df = pd.read_csv(os.path.join(PREP_DIR, "val.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(PREP_DIR, "test.csv"), encoding="utf-8-sig")

    assert (train_df["is_synthetic"] == False).all(), "Training set contains synthetic data!"
    assert (val_df["is_synthetic"] == False).all(), "Validation split contains synthetic data!"
    assert (test_df["is_synthetic"] == False).all(), "Blind Test split contains synthetic data!"

    train_texts = set(train_df["clean_text"].dropna().values)
    val_texts = set(val_df["clean_text"].dropna().values)
    test_texts = set(test_df["clean_text"].dropna().values)

    val_overlap = train_texts.intersection(val_texts)
    test_overlap = train_texts.intersection(test_texts)

    assert len(val_overlap) == 0, f"Detected {len(val_overlap)} overlapping samples between Train and Val!"
    assert len(test_overlap) == 0, f"Detected {len(test_overlap)} overlapping samples between Train and Test!"
