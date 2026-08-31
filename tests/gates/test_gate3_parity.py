# -*- coding: utf-8 -*-
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate3_golden_parity_vectors():
    golden_path = os.path.join(BASE_DIR, "artifacts", "golden_parity_1000.json")
    assert os.path.exists(golden_path), "golden_parity_1000.json not found!"

    with open(golden_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    assert len(vectors) >= 1000, f"Expected >= 1000 golden vectors, got {len(vectors)}"
    for v in vectors[:100]:
        diff = v["max_parity_delta"]
        assert diff <= 1e-5, f"Parity difference ({diff}) exceeds 1e-5 for {v['vector_id']}"
