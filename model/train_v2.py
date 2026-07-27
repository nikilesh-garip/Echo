"""
ECHO — Clean Training Pipeline (train_v2.py)
============================================
Trains candidate model architectures (EchoCNN or EchoCRNN) using Phase 6 dataset splits.
Methodology:
  - Strict train/val isolation (never uses test set during training or LR tuning)
  - Inverse frequency class-weighted CrossEntropyLoss for imbalance handling
  - Automatic best checkpoint selection based on Validation Macro F1
  - Logs experiment details and metrics to experiments/experiment_log.csv via tracker.py
  - Runs full validation evaluation via evaluate_v2.py
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

from model_v2 import EchoCNN, EchoCRNN, count_parameters
from dataset_v2 import get_dataloaders_v2, PREPROCESSING_CONFIG
from tracker import log_experiment
from evaluate_v2 import evaluate_checkpoint

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(train_dataset, num_classes):
    counts = torch.zeros(num_classes)
    for _, mapped_id, _ in train_dataset.records:
        counts[mapped_id] += 1

    total = counts.sum()
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    weights = weights / weights.mean()
    return weights

def train_experiment(
    exp_id="EXP_001_CNN_BASELINE_3CLASS",
    arch="CNN",
    lr=0.001,
    batch_size=32,
    epochs=30,
    active_class_ids=None,
    seed=42,
    notes="Baseline training run"
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"Starting Training: {exp_id}")
    print(f"  Architecture: {arch} | Device: {device} | LR: {lr} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"========================================================\n")

    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    train_csv = os.path.join(ROOT, "model", "data", "splits", "train.csv")
    val_csv   = os.path.join(ROOT, "model", "data", "splits", "val.csv")
    test_csv  = os.path.join(ROOT, "model", "data", "splits", "test.csv")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders_v2(
        train_csv, val_csv, test_csv, batch_size=batch_size, active_class_ids=active_class_ids
    )

    num_classes = len(active_class_ids) if active_class_ids is not None else 6

    # Model Selection
    if arch.upper() == "CNN":
        model = EchoCNN(num_classes=num_classes).to(device)
    elif arch.upper() == "CRNN":
        model = EchoCRNN(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Compute class weights for loss function
    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
    print(f"Computed Class Weights: {class_weights.cpu().numpy().round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_dir, f"{exp_id}_best.pth")

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_total += inputs.size(0)

        epoch_train_loss = running_train_loss / train_total

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_labels)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        _, _, epoch_val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)

        scheduler.step(epoch_val_f1)

        is_best = epoch_val_f1 > best_val_f1
        if is_best:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), best_checkpoint_path)

        best_marker = " --> BEST!" if is_best else ""
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (macro): {epoch_val_f1:.4f}{best_marker}")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s. Best Val F1: {best_val_f1:.4f}")
    print(f"Saved best model checkpoint to: {best_checkpoint_path}")

    # Benchmark Model Size and Latency on CPU
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 1, 64, 63)
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    # Benchmark 20 runs
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - t0) * 1000.0) # ms

    avg_latency_ms = np.mean(latencies)
    model_size_mb = os.path.getsize(best_checkpoint_path) / (1024 * 1024)

    # Run full Evaluation on Validation set
    print("\n--- Running Final Validation Evaluation ---")
    val_results = evaluate_checkpoint(
        best_checkpoint_path,
        val_csv,
        arch=arch,
        active_class_ids=active_class_ids,
        exp_id=exp_id
    )

    # Prepare experiment log record
    exp_record = {
        "exp_id": exp_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arch": arch,
        "num_classes": num_classes,
        "git_commit": "89e95b5",
        "class_mapping_version": "v1.0",
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "augmentation": "Waveform (Shift/Gain) + SpecAugment (Freq/Time Mask)",
        "train_samples": len(train_loader.dataset),
        "val_f1_macro": val_results.get("f1_macro", best_val_f1),
        "val_acc": val_results.get("accuracy", 0.0),
        "test_f1_macro": "UNTOUCHED",
        "test_acc": "UNTOUCHED",
        "hazard_false_alarm_rate": val_results.get("hazard_far", 0.0),
        "hazard_miss_rate_per_class": val_results.get("hazard_mr", 0.0),
        "model_size_mb": round(model_size_mb, 2),
        "latency_ms": round(avg_latency_ms, 2),
        "checkpoint_path": best_checkpoint_path,
        "notes": notes
    }

    log_experiment(exp_record)
    print(f"\nExperiment {exp_id} logged successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Echo ML Model Baseline")
    parser.add_argument("--exp-id", type=str, default="EXP_001_CNN_BASELINE_3CLASS")
    parser.add_argument("--arch", type=str, default="CNN", choices=["CNN", "CRNN"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--active-classes", type=str, default="0,1,4", help="Comma separated active class IDs e.g. 0,1,4 or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", type=str, default="3-class baseline training run")

    args = parser.parse_args()

    if args.active_classes.lower() == "all":
        active_ids = None
    else:
        active_ids = [int(x) for x in args.active_classes.split(",")]

    train_experiment(
        exp_id=args.exp_id,
        arch=args.arch,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        active_class_ids=active_ids,
        seed=args.seed,
        notes=args.notes
    )
