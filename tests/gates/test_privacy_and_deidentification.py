# -*- coding: utf-8 -*-
"""
Gate 6 - Automated Privacy, PII Scrubbing, and De-Identification Verification
Scans all published dataset splits and golden parity vectors for raw unmasked PII:
- Raw phone numbers (10 to 12 digits)
- Raw email addresses
- Raw bank account numbers
"""
import os
import re
import json
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP_DIR = os.path.join(BASE_DIR, "prepared_4way_p5")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# Strict Unmasked PII Detectors (must NOT match masked tokens like <PHONE>, <EMAIL>, <ACCT>)
RAW_PHONE_STRICT = re.compile(r"(?<![0-9a-zA-Z])(?:\+?91|0)?[6-9]\d{9}(?![0-9a-zA-Z])")
RAW_EMAIL_STRICT = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RAW_ACCOUNT_STRICT = re.compile(r"\b(?:a/c|account)\s*(?:no\.?)?\s*([0-9]{9,18})\b", re.IGNORECASE)

def test_published_csv_splits_are_fully_deidentified():
    for split in ["train.csv", "val.csv", "test.csv"]:
        fpath = os.path.join(PREP_DIR, split)
        assert os.path.exists(fpath), f"Missing split file {split}"
        df = pd.read_csv(fpath, encoding="utf-8-sig")

        # Scan every message text
        unmasked_phones = 0
        unmasked_emails = 0
        unmasked_accounts = 0

        for idx, row in df.iterrows():
            text = str(row["text"])
            # Remove legitimate masked tokens prior to scanning
            clean_check = text.replace("<PHONE>", "").replace("<EMAIL>", "").replace("<ACCT>", "").replace("<REF>", "").replace("<OTP>", "").replace("<VPA>", "")

            if RAW_PHONE_STRICT.search(clean_check):
                unmasked_phones += 1
            if RAW_EMAIL_STRICT.search(clean_check):
                unmasked_emails += 1
            if RAW_ACCOUNT_STRICT.search(clean_check):
                unmasked_accounts += 1

        assert unmasked_phones == 0, f"Found {unmasked_phones} unmasked phone numbers in {split}!"
        assert unmasked_emails == 0, f"Found {unmasked_emails} unmasked email addresses in {split}!"
        assert unmasked_accounts == 0, f"Found {unmasked_accounts} unmasked account numbers in {split}!"

def test_golden_parity_vectors_are_fully_deidentified():
    golden_path = os.path.join(ARTIFACT_DIR, "golden_parity_1000.json")
    assert os.path.exists(golden_path), "Missing golden_parity_1000.json"

    with open(golden_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    for v in vectors:
        text = str(v["raw_text"])
        clean_check = text.replace("<PHONE>", "").replace("<EMAIL>", "").replace("<ACCT>", "").replace("<REF>", "").replace("<OTP>", "").replace("<VPA>", "")
        assert not RAW_EMAIL_STRICT.search(clean_check), f"Unmasked email found in golden vector {v['vector_id']}"
        assert not RAW_ACCOUNT_STRICT.search(clean_check), f"Unmasked account found in golden vector {v['vector_id']}"
