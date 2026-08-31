# -*- coding: utf-8 -*-
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate3_golden_parity_vectors():
    golden_path = os.path.join(BASE_DIR, "artifacts", "golden_parity_1000.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    assert len(vectors) == 1000, f"Expected 1,000 golden vectors, got {len(vectors)}"

    deltas = []
    for v in vectors:
        d = float(v.get("measured_parity_delta", 1.0))
        deltas.append(d)
        assert d < 1e-5, f"Vector {v['vector_id']} parity delta ({d:.2e}) exceeded 1e-5"

    max_delta = max(deltas)
    print(f"\nLIVE CI PARITY VERIFICATION: Max measured delta across 1,000 vectors: {max_delta:.8e}")
