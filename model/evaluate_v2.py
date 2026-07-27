"""
ECHO — Phase 13 Evaluation Suite (evaluate_v2.py)
==================================================
Evaluates a trained model checkpoint against validation or test manifests.
Computes:
  - Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted)
  - Per-class Precision, Recall, F1, FPR, FNR, Support
  - Project-Critical Metrics:
      * HAZARD FALSE ALARM RATE (NORMAL clips wrongly classified as any HAZARD)
      * HAZARD MISS RATE (HAZARD clips wrongly classified as NORMAL)
  - Confusion Matrix with explicit class names

Saves evaluation report to `reports/evaluation_{exp_id}.txt`.
"""

import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model_v2 import EchoCNN, EchoCRNN
from dataset_v2 import EchoDatasetV2, DataLoader, PREPROCESSING_CONFIG

CLASS_NAMES_6 = ["normal", "gunshot", "explosion", "human_distress", "siren", "fire_alarm"]
CLASS_NAMES_3 = ["normal", "gunshot", "siren"]

def evaluate_checkpoint(checkpoint_path, split_csv, arch="CNN", active_class_ids=None, exp_id="EXP_001"):
    """
    Evaluates model checkpoint against split_csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if active_class_ids is not None:
        class_names = [CLASS_NAMES_6[i] for i in sorted(active_class_ids)]
        num_classes = len(active_class_ids)
    else:
        class_names = CLASS_NAMES_6
        num_classes = len(class_names)

    # Initialize model
    if arch.upper() == "CNN":
        model = EchoCNN(num_classes=num_classes)
    elif arch.upper() == "CRNN":
        model = EchoCRNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = EchoDatasetV2(split_csv, is_training=False, active_class_ids=active_class_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0:
        print("Warning: Evaluated on 0 samples.")
        return {}

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Per-class FPR and FNR
    # FPR = FP / (FP + TN)
    # FNR = FN / (FN + TP)
    class_metrics = {}
    precisions, recalls, f1s, supports = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(num_classes)), zero_division=0)

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        class_metrics[name] = {
            "precision": precisions[i],
            "recall": recalls[i],
            "f1": f1s[i],
            "support": supports[i],
            "fpr": fpr,
            "fnr": fnr
        }

    # Project-Critical Metrics:
    # 1. HAZARD FALSE ALARM RATE = (NORMAL clips classified as any HAZARD) / total NORMAL clips
    # Normal is index 0
    total_normal = np.sum(cm[0, :])
    if total_normal > 0:
        normal_false_alarms = total_normal - cm[0, 0] # any prediction != 0
        hazard_far = normal_false_alarms / total_normal
    else:
        hazard_far = 0.0

    # 2. HAZARD MISS RATE = (HAZARD clips classified as NORMAL) / total HAZARD clips
    total_hazards = np.sum(cm[1:, :])
    if total_hazards > 0:
        hazard_misses = np.sum(cm[1:, 0]) # hazard row, normal col
        hazard_mr = hazard_misses / total_hazards
    else:
        hazard_mr = 0.0

    # Format report string
    report_lines = []
    report_lines.append(f"ECHO Model Evaluation Report — {exp_id}")
    report_lines.append("=" * 60)
    report_lines.append(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    report_lines.append(f"Split CSV:  {os.path.basename(split_csv)} ({len(all_labels)} samples)")
    report_lines.append(f"Arch:       {arch} | Num Classes: {num_classes}")
    report_lines.append("-" * 60)
    report_lines.append(f"Overall Accuracy:       {acc:.4f}")
    report_lines.append(f"Macro Precision:        {p_macro:.4f}")
    report_lines.append(f"Macro Recall:           {r_macro:.4f}")
    report_lines.append(f"Macro F1 Score:         {f1_macro:.4f}")
    report_lines.append(f"Weighted F1 Score:      {f1_weighted:.4f}")
    report_lines.append("-" * 60)
    report_lines.append(f"HAZARD FALSE ALARM RATE: {hazard_far:.4f} ({hazard_far*100:.2f}%)")
    report_lines.append(f"HAZARD MISS RATE:       {hazard_mr:.4f} ({hazard_mr*100:.2f}%)")
    report_lines.append("=" * 60)
    report_lines.append("\nClass-wise Metrics:")
    report_lines.append(f"{'Class':<18} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'FPR':<7} | {'FNR':<7} | {'Support':<7}")
    report_lines.append("-" * 75)
    for name in class_names:
        m = class_metrics[name]
        report_lines.append(
            f"{name:<18} | {m['precision']:.4f}  | {m['recall']:.4f}  | {m['f1']:.4f}  | {m['fpr']:.4f}  | {m['fnr']:.4f}  | {m['support']:<7}"
        )

    report_lines.append("\nConfusion Matrix:")
    header_str = "        " + "  ".join(f"{c[:6]:>6}" for c in class_names)
    report_lines.append(header_str)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:6]:>6}  " + "  ".join(f"{val:6d}" for val in row)
        report_lines.append(row_str)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save to file
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, f"evaluation_{exp_id}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved evaluation report to: {report_file}")

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "hazard_far": hazard_far,
        "hazard_mr": hazard_mr,
        "cm": cm.tolist()
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Echo Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CNN", choices=["CNN", "CRNN"])
    parser.add_argument("--active-classes", type=str, default=None, help="Comma separated IDs e.g. 0,1,4")
    parser.add_argument("--exp-id", type=str, default="EXP_EVAL")
    args = parser.parse_args()

    active_ids = [int(x) for x in args.active_classes.split(",")] if args.active_classes else None
    evaluate_checkpoint(args.checkpoint, args.split_csv, arch=args.arch, active_class_ids=active_ids, exp_id=args.exp_id)
