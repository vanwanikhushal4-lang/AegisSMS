# -*- coding: utf-8 -*-
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate2_quality_metrics():
    metrics_path = os.path.join(BASE_DIR, "artifacts", "final_metrics_3way.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    smishing_metrics = metrics["class_metrics"]["SMISHING"]
    ham_fpr = metrics["ham_false_positive_rate"]

    assert smishing_metrics["recall"] >= 0.80, f"Smishing recall ({smishing_metrics['recall']:.4f}) below threshold"
    assert smishing_metrics["precision"] >= 0.85, f"Smishing precision ({smishing_metrics['precision']:.4f}) below threshold"
    assert ham_fpr <= 0.010, f"Ham FPR ({ham_fpr:.4f}) exceeds threshold"
