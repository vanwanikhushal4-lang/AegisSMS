# -*- coding: utf-8 -*-
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate2_intent_quality_metrics():
    metrics_path = os.path.join(BASE_DIR, "artifacts", "final_metrics_3way.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["test_accuracy"] >= 0.85, f"Test accuracy ({metrics['test_accuracy']:.4f}) below 85%"
    assert metrics["class_metrics"]["PERSONAL"]["precision"] >= 0.85
    assert metrics["class_metrics"]["PROMOTIONAL"]["precision"] >= 0.85
    assert metrics["class_metrics"]["TRANSACTIONAL"]["precision"] >= 0.80
