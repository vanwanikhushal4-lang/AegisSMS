# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP_DIR = os.path.join(BASE_DIR, "prepared_4way_p5")

def test_gate1_zero_data_and_template_leakage():
    train_df = pd.read_csv(os.path.join(PREP_DIR, "train.csv"), encoding="utf-8-sig")
    val_df = pd.read_csv(os.path.join(PREP_DIR, "val.csv"), encoding="utf-8-sig")
    test_df = pd.read_csv(os.path.join(PREP_DIR, "test.csv"), encoding="utf-8-sig")

    assert (train_df["is_synthetic"] == False).all(), "Train set has synthetic data!"
    assert (val_df["is_synthetic"] == False).all(), "Val set has synthetic data!"
    assert (test_df["is_synthetic"] == False).all(), "Test set has synthetic data!"

    train_texts = set(train_df["clean_text"].dropna().values)
    val_texts = set(val_df["clean_text"].dropna().values)
    test_texts = set(test_df["clean_text"].dropna().values)

    train_tmpls = set(train_df["template_hash"].dropna().values)
    val_tmpls = set(val_df["template_hash"].dropna().values)
    test_tmpls = set(test_df["template_hash"].dropna().values)

    # 1. Text overlap check
    assert len(train_texts.intersection(val_texts)) == 0, "Clean text overlap Train-Val!"
    assert len(train_texts.intersection(test_texts)) == 0, "Clean text overlap Train-Test!"

    # 2. Template family overlap check
    assert len(train_tmpls.intersection(val_tmpls)) == 0, "Template overlap Train-Val!"
    assert len(train_tmpls.intersection(test_tmpls)) == 0, "Template overlap Train-Test!"
