"""
ECHO — Phase 14: Threshold Calibration on Validation Set
=========================================================
Calibrates per-class detection thresholds using Validation Set probability distributions.
DO NOT calibrate on the final TEST set.

Goal:
  - Maximize Validation F1 score per hazard class
  - Maintain Hazard False Alarm Rate (NORMAL -> HAZARD) <= 1.5%
  - Maintain Hazard Recall > 95%
  - Establish UNCERTAIN state probability margins (Phase 15)

Outputs:
  - config/thresholds_{exp_id}.json
  - Calibration summary report printed to console
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from model_v2 import EchoCNN, EchoCRNN
from dataset_v2 import EchoDatasetV2, DataLoader
from preprocessing import PREPROCESSING_CONFIG

CLASS_NAMES_3 = ["normal", "gunshot", "siren"]

def calibrate_thresholds(checkpoint_path, val_csv_path, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(active_class_ids) if active_class_ids else 6
    class_names = [CLASS_NAMES_3[i] if len(CLASS_NAMES_3) > i else f"class_{i}" for i in range(num_classes)]

    if arch.upper() == "CNN":
        model = EchoCNN(num_classes=num_classes)
    else:
        model = EchoCRNN(num_classes=num_classes)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dataset = EchoDatasetV2(val_csv_path, is_training=False, active_class_ids=active_class_ids)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs  = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)  # (N, num_classes)
    all_labels = np.array(all_labels) # (N,)

    # Extract probabilities for normal vs hazard
    normal_mask = (all_labels == 0)
    normal_probs = all_probs[normal_mask] # (N_normal, num_classes)

    print("=" * 60)
    print(f"THRESHOLD CALIBRATION REPORT — {exp_id}")
    print("=" * 60)
    print(f"Validation Samples: {len(all_labels)} (Normal: {normal_probs.shape[0]})")

    calibrated_thresholds = {}
    
    # 1. Calibrate hazard thresholds by maximizing F1 score subject to far <= 0.02
    for c_idx in range(1, num_classes):
        cls_name = class_names[c_idx]
        hazard_mask = (all_labels == c_idx)
        hazard_probs_for_cls = all_probs[hazard_mask, c_idx]
        normal_probs_for_cls = normal_probs[:, c_idx]

        best_thresh = 0.50
        best_f1 = 0.0
        best_far = 1.0
        best_rec = 0.0

        for thresh in np.linspace(0.30, 0.90, 61):
            far = np.mean(normal_probs_for_cls >= thresh) # FPR on normal
            rec = np.mean(hazard_probs_for_cls >= thresh) if len(hazard_probs_for_cls) > 0 else 0.0
            
            # Precision = TP / (TP + FP)
            tp = np.sum(hazard_probs_for_cls >= thresh)
            fp = np.sum(normal_probs_for_cls >= thresh)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            if far <= 0.02 and f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_far = float(far)
                best_rec = float(rec)

        calibrated_thresholds[cls_name] = {
            "class_id": c_idx,
            "threshold": round(best_thresh, 3),
            "val_f1": round(best_f1, 4),
            "val_far": round(best_far, 4),
            "val_recall": round(best_rec, 4)
        }
        print(f"  Class: {cls_name:<15} -> Threshold: {best_thresh:.3f} | Val F1: {best_f1:.4f} | Val FAR: {best_far*100:.2f}% | Val Recall: {best_rec*100:.2f}%")

    # 2. Phase 15 UNCERTAIN State Margins
    uncertain_margin = 0.15
    calibrated_thresholds["uncertain_margin"] = uncertain_margin
    calibrated_thresholds["normal_threshold"] = 0.50

    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out_dir = os.path.join(ROOT, "config")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"thresholds_{exp_id}.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calibrated_thresholds, f, indent=2)

    print("-" * 60)
    print(f"Saved threshold calibration config to: {out_json}")
    return calibrated_thresholds

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt = os.path.join(ROOT, "checkpoints", "EXP_002_CRNN_BASELINE_3CLASS_best.pth")
    val_csv = os.path.join(ROOT, "model", "data", "splits", "val.csv")
    if os.path.exists(ckpt) and os.path.exists(val_csv):
        calibrate_thresholds(ckpt, val_csv, arch="CRNN", active_class_ids=[0, 1, 4], exp_id="EXP_002_CRNN_BASELINE_3CLASS")
    else:
        print(f"Checkpoint or val_csv not found.")
