# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import urllib.request
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_kotlinc():
    """Ensure kotlinc is available either in system PATH or local tools dir."""
    kotlinc_cmd = shutil.which("kotlinc")
    if kotlinc_cmd:
        return kotlinc_cmd

    # Check local bin
    local_kotlinc_win = os.path.join(BASE_DIR, "kotlinc_bin", "kotlinc", "bin", "kotlinc.bat")
    local_kotlinc_unix = os.path.join(BASE_DIR, "kotlinc_bin", "kotlinc", "bin", "kotlinc")

    if os.path.exists(local_kotlinc_win):
        return local_kotlinc_win
    if os.path.exists(local_kotlinc_unix):
        return local_kotlinc_unix

    # Auto-download standalone compiler for clean CI / fresh clone
    print("Downloading standalone kotlinc-1.9.23 for parity testing...")
    url = "https://github.com/JetBrains/kotlin/releases/download/v1.9.23/kotlin-compiler-1.9.23.zip"
    zip_path = os.path.join(BASE_DIR, "kotlin-compiler-1.9.23.zip")
    target_dir = os.path.join(BASE_DIR, "kotlinc_bin")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    if os.path.exists(local_kotlinc_win):
        return local_kotlinc_win
    if os.path.exists(local_kotlinc_unix):
        os.chmod(local_kotlinc_unix, 0o755)
        return local_kotlinc_unix
    return "kotlinc"

def ensure_json_jar():
    """Ensure org.json jar is available for standalone Kotlin execution."""
    lib_dir = os.path.join(BASE_DIR, "lib")
    os.makedirs(lib_dir, exist_ok=True)
    jar_path = os.path.join(lib_dir, "json.jar")
    if not os.path.exists(jar_path):
        url = "https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar"
        urllib.request.urlretrieve(url, jar_path)
    return jar_path

def test_gate3_actual_kotlin_parity_execution():
    golden_path = os.path.join(BASE_DIR, "artifacts", "golden_parity_1000.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    assert len(vectors) == 1000, f"Expected 1,000 golden vectors, got {len(vectors)}"

    kotlinc = ensure_kotlinc()
    json_jar = ensure_json_jar()

    bin_dir = os.path.join(BASE_DIR, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    out_jar = os.path.join(bin_dir, "kotlin_runner.jar")

    kt_classifier = os.path.join(BASE_DIR, "android", "AegisSmsClassifier.kt")
    kt_runner = os.path.join(BASE_DIR, "tests", "parity", "com", "payshield", "aegissms", "KotlinParityRunner.kt")

    # 1. Compile Kotlin Classifier & Runner
    compile_cmd = [
        kotlinc,
        "-cp", json_jar,
        "-include-runtime",
        "-d", out_jar,
        kt_classifier,
        kt_runner
    ]
    env = dict(os.environ)
    if os.name == "nt":
        env["PATH"] = r"C:\Windows\System32;C:\Windows;" + env.get("PATH", "")
    res_compile = subprocess.run(compile_cmd, cwd=BASE_DIR, capture_output=True, text=True, env=env)
    assert res_compile.returncode == 0, f"kotlinc failed:\nSTDOUT: {res_compile.stdout}\nSTDERR: {res_compile.stderr}"

    # 2. Execute Kotlin Classifier on JVM with -Xss4m
    sep = ";" if os.name == "nt" else ":"
    run_cmd = ["java", "-Xss4m", "-cp", f"{out_jar}{sep}{json_jar}", "com.payshield.aegissms.KotlinParityRunner", "."]
    res_run = subprocess.run(run_cmd, cwd=BASE_DIR, capture_output=True, text=True, env=env)
    print("\n" + res_run.stdout)
    assert res_run.returncode == 0, f"Kotlin Parity Execution failed:\nSTDOUT: {res_run.stdout}\nSTDERR: {res_run.stderr}"
    assert "STATUS: [PASS]" in res_run.stdout, "Kotlin parity check did not pass"

def test_gate3_actual_java_parity_execution():
    json_jar = ensure_json_jar()
    bin_dir = os.path.join(BASE_DIR, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    classifier_java = os.path.join(BASE_DIR, "android", "com", "payshield", "aegissms", "AegisSmsClassifier.java")
    runner_java = os.path.join(BASE_DIR, "tests", "parity", "com", "payshield", "aegissms", "ParityTestRunner.java")

    sep = ";" if os.name == "nt" else ":"
    compile_cmd = ["javac", "-encoding", "UTF-8", "-cp", json_jar, "-d", bin_dir, classifier_java, runner_java]
    res_compile = subprocess.run(compile_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"javac failed: {res_compile.stderr}"

    run_cmd = ["java", "-Xss4m", "-cp", f"{bin_dir}{sep}{json_jar}", "com.payshield.aegissms.ParityTestRunner", "."]
    res_run = subprocess.run(run_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    print("\n" + res_run.stdout)
    assert res_run.returncode == 0, f"Java Parity Execution failed: {res_run.stderr}"
    assert "STATUS: [PASS]" in res_run.stdout, "Java parity check did not pass"
