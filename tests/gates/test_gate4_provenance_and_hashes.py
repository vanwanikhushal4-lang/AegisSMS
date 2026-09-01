# -*- coding: utf-8 -*-
import os
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

def test_gate4_provenance_manifest_and_sha256_checksums():
    # 1. Verify Manifest
    manifest_path = os.path.join(ARTIFACT_DIR, "provenance_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["synthetic_count"] == 0, "Provenance manifest contains synthetic data!"
    assert len(manifest["sources"]) >= 5, "Insufficient provenance sources listed"

    for src in manifest["sources"]:
        assert src["real_or_synthetic"] == "REAL"
        assert src["medium"] == "SMS"
        assert len(src["immutable_revision"]) > 3
        assert len(src["license"]) > 1

    # 2. Verify SHA-256 Hashes
    hashes_path = os.path.join(ARTIFACT_DIR, "artifact_hashes.sha256")
    assert os.path.exists(hashes_path), "Missing artifact_hashes.sha256"

    with open(hashes_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 4, "Missing artifact hash entries"
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            expected_hash, raw_fname = parts[0], parts[1]
            clean_fname = raw_fname.replace("\\", "/")
            path_parts = [p for p in clean_fname.split("/") if p]
            fpath = os.path.join(BASE_DIR, *path_parts)
            if not os.path.exists(fpath):
                fpath = os.path.join(ARTIFACT_DIR, *path_parts)
            assert os.path.exists(fpath), f"Artifact {raw_fname} missing from repository at {fpath}"
            with open(fpath, "rb") as af:
                computed_hash = hashlib.sha256(af.read()).hexdigest()
            assert computed_hash == expected_hash, f"Hash mismatch for {raw_fname}: expected {expected_hash}, computed {computed_hash}"
