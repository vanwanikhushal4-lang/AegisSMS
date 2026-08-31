# -*- coding: utf-8 -*-
import os
import json
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gate3_actual_jvm_parity_execution():
    golden_path = os.path.join(BASE_DIR, "artifacts", "golden_parity_1000.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    assert len(vectors) == 1000, f"Expected 1,000 golden vectors, got {len(vectors)}"

    # 1. Compile Java Classifier and ParityTestRunner
    bin_dir = os.path.join(BASE_DIR, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    classifier_java = os.path.join(BASE_DIR, "android", "com", "payshield", "aegissms", "AegisSmsClassifier.java")
    runner_java = os.path.join(BASE_DIR, "tests", "parity", "com", "payshield", "aegissms", "ParityTestRunner.java")

    compile_cmd = ["javac", "-encoding", "UTF-8", "-d", bin_dir, classifier_java, runner_java]
    res_compile = subprocess.run(compile_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"javac failed: {res_compile.stderr}"

    # 2. Execute ParityTestRunner on JVM
    run_cmd = ["java", "-cp", bin_dir, "com.payshield.aegissms.ParityTestRunner", "."]
    res_run = subprocess.run(run_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    print("\n" + res_run.stdout)
    assert res_run.returncode == 0, f"Java Parity Execution failed: {res_run.stderr}"
    assert "STATUS: [PASS]" in res_run.stdout, "Parity check did not pass"
