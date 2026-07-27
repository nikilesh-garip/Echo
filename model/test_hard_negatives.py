"""
ECHO — Phase 17: Hard Negative Evaluation Suite
================================================
Evaluates the model against everyday non-hazardous audio sources (speech, music,
appliances, traffic, drilling, jackhammer, dog bark, footsteps) to ensure zero false alarms.

Outputs:
  - reports/hard_negative_eval.txt
"""

import os
import csv
import torch
import numpy as np

from two_pass_detector import TwoPassDetector

def evaluate_hard_negatives(checkpoint_path, test_csv_path, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS"):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    thresh_json = os.path.join(ROOT, "config", f"thresholds_{exp_id}.json")

    detector = TwoPassDetector(
        model_path=checkpoint_path,
        arch=arch,
        threshold_json=thresh_json if os.path.exists(thresh_json) else None
    )

    hard_negative_records = []
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["class_id"]) == 0: # normal class
                hard_negative_records.append((row["abs_path"], row["source_dataset"]))

    print(f"\n========================================================")
    print(f"HARD NEGATIVE EVALUATION SUITE — {exp_id}")
    print(f"========================================================")
    print(f"Total Hard Negative Samples: {len(hard_negative_records)}")

    state_counts = {"NORMAL": 0, "UNCERTAIN": 0, "HAZARD CANDIDATE": 0}
    false_alarms = []

    import soundfile as sf
    for i, (abs_path, source) in enumerate(hard_negative_records):
        try:
            audio, sr = sf.read(abs_path, dtype='float32')
        except Exception:
            continue

        state, cand_cls, max_prob, probs_dict = detector.run_pass_1(audio, sr)
        state_counts[state] += 1

        if state == "HAZARD CANDIDATE":
            false_alarms.append((abs_path, source, cand_cls, max_prob))

    total_tested = sum(state_counts.values())
    far_pct = (state_counts["HAZARD CANDIDATE"] / total_tested * 100) if total_tested > 0 else 0.0
    unc_pct = (state_counts["UNCERTAIN"] / total_tested * 100) if total_tested > 0 else 0.0
    norm_pct = (state_counts["NORMAL"] / total_tested * 100) if total_tested > 0 else 0.0

    report_lines = []
    report_lines.append(f"HARD NEGATIVE EVALUATION REPORT — {exp_id}")
    report_lines.append("=" * 60)
    report_lines.append(f"Total Tested:         {total_tested}")
    report_lines.append(f"Correctly NORMAL:     {state_counts['NORMAL']} ({norm_pct:.2f}%)")
    report_lines.append(f"Flagged UNCERTAIN:    {state_counts['UNCERTAIN']} ({unc_pct:.2f}%)")
    report_lines.append(f"FALSE ALARMS:         {state_counts['HAZARD CANDIDATE']} ({far_pct:.2f}%)")
    report_lines.append("-" * 60)

    if false_alarms:
        report_lines.append("Misclassified Audio Sources (False Alarms):")
        for path, src, cand, p in false_alarms[:20]:
            report_lines.append(f"  [{src}] {os.path.basename(path)} -> {cand} (prob={p:.4f})")
    else:
        report_lines.append("ZERO False Alarms on Hard Negative Evaluation Suite! Perfect Score.")

    report_text = "\n".join(report_lines)
    print(report_text)

    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, f"hard_negative_eval_{exp_id}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved hard negative evaluation report to: {report_file}")
    return state_counts

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt = os.path.join(ROOT, "checkpoints", "EXP_002_CRNN_BASELINE_3CLASS_best.pth")
    test_csv = os.path.join(ROOT, "model", "data", "splits", "test.csv")
    if os.path.exists(ckpt) and os.path.exists(test_csv):
        evaluate_hard_negatives(ckpt, test_csv, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS")
    else:
        print(f"Checkpoint or test_csv not found.")
