# -*- coding: utf-8 -*-
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate2_4way_quality_metrics():
    metrics_path = os.path.join(BASE_DIR, "artifacts", "final_metrics_3way.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["test_accuracy"] >= 0.85, f"Overall accuracy ({metrics['test_accuracy']:.4f}) below 85%"
    assert metrics["class_metrics"]["SCAM"]["precision"] >= 0.90, "Scam precision below 90%"
    assert metrics["class_metrics"]["SCAM"]["recall"] >= 0.90, "Scam recall below 90%"
    assert metrics["class_metrics"]["PERSONAL"]["precision"] >= 0.75
    assert metrics["class_metrics"]["TRANSACTIONAL"]["precision"] >= 0.70
    assert metrics["class_metrics"]["PROMOTIONAL"]["precision"] >= 0.75
