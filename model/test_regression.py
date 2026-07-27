"""
ECHO — Phase 22: Permanent Regression Test Suite
=================================================
Validates candidate model checkpoints against a fixed regression benchmark of
representative audio clips (speech, music, noise, genuine gunshots, sirens).

Any candidate model that causes regression on these baseline sanity checks MUST BE REJECTED.
"""

import os
import torch
import numpy as np

from two_pass_detector import TwoPassDetector

def run_regression_tests(checkpoint_path, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS"):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    thresh_json = os.path.join(ROOT, "config", f"thresholds_{exp_id}.json")

    detector = TwoPassDetector(
        model_path=checkpoint_path,
        arch=arch,
        threshold_json=thresh_json if os.path.exists(thresh_json) else None
    )

    test_csv = os.path.join(ROOT, "model", "data", "splits", "test.csv")
    if not os.path.exists(test_csv):
        print(f"[SKIP] {test_csv} not found.")
        return False

    import csv, soundfile as sf
    # Sample up to 100 gunshot, 100 siren, 100 normal from test set for regression benchmark
    by_class = {"normal": [], "gunshot": [], "siren": []}
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["echo_class"]
            if cls in by_class and len(by_class[cls]) < 100:
                by_class[cls].append(row["abs_path"])

    print("=" * 60)
    print(f"REGRESSION TEST SUITE BENCHMARK — {exp_id}")
    print("=" * 60)

    results = {}
    all_passed = True

    # 1. Gunshot Detection Benchmark (Pass 1 + Pass 2 Verification)
    gunshot_detected = 0
    gunshot_total = len(by_class["gunshot"])
    for path in by_class["gunshot"]:
        try:
            audio, sr = sf.read(path, dtype='float32')
            state, cand_cls, p, _ = detector.run_pass_1(audio, sr)
            if cand_cls == "gunshot" and state in ["HAZARD CANDIDATE", "UNCERTAIN"]:
                # Run Pass 2 verification over 5s buffer (pad audio to 5s if needed)
                dur = len(audio) / sr
                if dur < 5.0:
                    pad_len = int(5.0 * sr) - len(audio)
                    audio_5s = np.pad(audio, (0, max(0, pad_len)))
                else:
                    audio_5s = audio
                is_verified, v_p = detector.run_pass_2(audio_5s, sr, "gunshot")
                if is_verified or state == "HAZARD CANDIDATE":
                    gunshot_detected += 1
        except Exception:
            continue

    gunshot_acc = (gunshot_detected / gunshot_total * 100) if gunshot_total else 0
    print(f"  Gunshot Detection Benchmark : {gunshot_detected}/{gunshot_total} ({gunshot_acc:.2f}%)  [Min Target: 95.0%]")
    results["gunshot_acc"] = gunshot_acc
    if gunshot_acc < 95.0:
        all_passed = False

    # 2. Siren Detection Benchmark
    siren_detected = 0
    siren_total = len(by_class["siren"])
    for path in by_class["siren"]:
        try:
            audio, sr = sf.read(path, dtype='float32')
            state, cand_cls, p, _ = detector.run_pass_1(audio, sr)
            if cand_cls == "siren" and state in ["HAZARD CANDIDATE", "UNCERTAIN"]:
                dur = len(audio) / sr
                if dur < 5.0:
                    pad_len = int(5.0 * sr) - len(audio)
                    audio_5s = np.pad(audio, (0, max(0, pad_len)))
                else:
                    audio_5s = audio
                is_verified, v_p = detector.run_pass_2(audio_5s, sr, "siren")
                if is_verified or state == "HAZARD CANDIDATE":
                    siren_detected += 1
        except Exception:
            continue

    siren_acc = (siren_detected / siren_total * 100) if siren_total else 0
    print(f"  Siren Detection Benchmark  : {siren_detected}/{siren_total} ({siren_acc:.2f}%)  [Min Target: 80.0%]")
    results["siren_acc"] = siren_acc
    if siren_acc < 80.0:
        all_passed = False

    # 3. Normal False Alarm Benchmark
    normal_fa = 0
    normal_total = len(by_class["normal"])
    for path in by_class["normal"]:
        try:
            audio, sr = sf.read(path, dtype='float32')
            state, cand_cls, p, _ = detector.run_pass_1(audio, sr)
            if state == "HAZARD CANDIDATE":
                dur = len(audio) / sr
                if dur < 5.0:
                    pad_len = int(5.0 * sr) - len(audio)
                    audio_5s = np.pad(audio, (0, max(0, pad_len)))
                else:
                    audio_5s = audio
                is_verified, _ = detector.run_pass_2(audio_5s, sr, cand_cls)
                if is_verified:
                    normal_fa += 1
        except Exception:
            continue

    normal_fa_pct = (normal_fa / normal_total * 100) if normal_total else 0
    print(f"  Normal False Alarm Rate    : {normal_fa}/{normal_total} ({normal_fa_pct:.2f}%)   [Max Target: 1.50%]")
    results["normal_fa"] = normal_fa_pct
    if normal_fa_pct > 1.50:
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("  [REGRESSION TEST SUITE PASSED] All benchmarks met criteria!")
    else:
        print("  [REGRESSION TEST SUITE FAILED] One or more benchmarks dropped below target thresholds.")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt = os.path.join(ROOT, "checkpoints", "EXP_002_CRNN_BASELINE_3CLASS_best.pth")
    if os.path.exists(ckpt):
        run_regression_tests(ckpt, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS")
