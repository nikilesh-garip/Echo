"""
ECHO — Phase 12: Experiment Tracking Utility
=============================================
Manages logging of all model training runs to `experiments/experiment_log.csv`.
Never overwrites past experiment logs.
"""

import os
import csv
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOG_PATH = os.path.join(ROOT_DIR, "experiments", "experiment_log.csv")

HEADERS = [
    "exp_id",
    "date",
    "arch",
    "num_classes",
    "git_commit",
    "class_mapping_version",
    "seed",
    "lr",
    "batch_size",
    "epochs",
    "augmentation",
    "train_samples",
    "val_f1_macro",
    "val_acc",
    "test_f1_macro",
    "test_acc",
    "hazard_false_alarm_rate",
    "hazard_miss_rate_per_class",
    "model_size_mb",
    "latency_ms",
    "checkpoint_path",
    "notes"
]

def init_experiment_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
        print(f"Initialized experiment log at: {LOG_PATH}")

def log_experiment(exp_data):
    """
    Appends a dictionary of experiment metadata to experiment_log.csv.
    """
    init_experiment_log()
    row = []
    for h in HEADERS:
        val = exp_data.get(h, "")
        if isinstance(val, float):
            val = f"{val:.4f}"
        row.append(str(val))

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Logged experiment {exp_data.get('exp_id')} to {LOG_PATH}")

if __name__ == "__main__":
    init_experiment_log()
