# -*- coding: utf-8 -*-
"""
Gate 7 - Data Manifest Allowlist, Non-Synthetic Verification, and Integrity Gate
Ensures:
1. Every dataset source is explicitly allowlisted with immutable revisions and open licenses.
2. Zero synthetic, email, YouTube, Telegram, or blacklisted sources exist in the manifest.
3. 100% verified real mobile SMS with synthetic_count == 0.
"""
import os
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

FORBIDDEN_KEYWORDS = ["synthetic", "email", "youtube", "telegram", "scraped_web_forum", "enron", "spamassassin", "llama", "mistral"]

def test_data_manifest_allowlist_compliance():
    manifest_path = os.path.join(ARTIFACT_DIR, "data_manifest.json")
    assert os.path.exists(manifest_path), "Missing data_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["policy"]["allowlist_only"] == True
    sources = manifest["approved_sources"]
    assert len(sources) >= 4, "Expected at least 4 approved sources in data manifest"

    for src in sources:
        src_id = src["source_id"].lower()
        src_name = src["name"].lower()
        # Verify no forbidden types exist
        for kw in FORBIDDEN_KEYWORDS:
            assert kw not in src_id, f"Forbidden keyword '{kw}' found in source_id: {src_id}"
            assert kw not in src_name, f"Forbidden keyword '{kw}' found in source_name: {src_name}"

        # Verify required metadata
        assert src.get("medium") == "SMS", f"Source {src_id} medium must be SMS"
        assert src.get("real_or_synthetic") == "REAL", f"Source {src_id} must be verified REAL"
        assert src.get("immutable_revision"), f"Source {src_id} missing immutable revision"
        assert src.get("license"), f"Source {src_id} missing license"

def test_provenance_manifest_non_synthetic_verification():
    prov_path = os.path.join(ARTIFACT_DIR, "provenance_manifest.json")
    assert os.path.exists(prov_path), "Missing provenance_manifest.json"

    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert prov["synthetic_count"] == 0, "Provenance manifest reports non-zero synthetic count!"
    assert prov["verified_real_percentage"] == 100.0, "Verified real percentage must be 100.0%"
