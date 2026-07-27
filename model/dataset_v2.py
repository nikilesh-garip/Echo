"""
ECHO — Clean PyTorch Dataset & DataLoader Builder (Phase 7-9)
==============================================================
Loads split manifests (train.csv, val.csv, test.csv) created in Phase 6.
Applies canonical preprocessing (model/preprocessing.py) and augmentations (model/augmentation.py).
"""

import os
import csv
import torch
import soundfile as sf
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocessing import load_and_preprocess_audio, LogMelSpectrogramExtractor, PREPROCESSING_CONFIG
from augmentation import apply_waveform_augmentations, SpecAugment

class EchoDatasetV2(Dataset):
    def __init__(self, manifest_csv, is_training=False, active_class_ids=None, config=PREPROCESSING_CONFIG):
        """
        Args:
            manifest_csv (str): Path to train.csv, val.csv, or test.csv
            is_training (bool): If True, enables waveform & SpecAugment augmentations
            active_class_ids (list or tuple, optional): Filter dataset to specific class IDs (e.g. [0, 1, 4] for 3-class)
            config (dict): Preprocessing parameters
        """
        self.is_training = is_training
        self.config = config
        self.spec_extractor = LogMelSpectrogramExtractor(config=config)
        self.spec_augment = SpecAugment() if is_training else None

        # Build class ID remapping dictionary if filtering active classes
        self.class_remap = {}
        if active_class_ids is not None:
            sorted_active = sorted(list(active_class_ids))
            for new_idx, orig_id in enumerate(sorted_active):
                self.class_remap[orig_id] = new_idx

        self.records = []
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                abs_path = row["abs_path"]
                cls_id = int(row["class_id"])
                cls_name = row["echo_class"]

                if active_class_ids is not None and cls_id not in active_class_ids:
                    continue

                mapped_id = self.class_remap.get(cls_id, cls_id)

                if os.path.exists(abs_path):
                    self.records.append((abs_path, mapped_id, cls_name))

        print(f"Loaded {len(self.records)} samples from {os.path.basename(manifest_csv)} (is_training={is_training}, num_classes={len(self.class_remap) if self.class_remap else 'all'})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        abs_path, mapped_id, cls_name = self.records[idx]

        # Load waveform
        waveform_np, sr = sf.read(abs_path, dtype='float32')
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)

        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0) # (1, N)

        # Resample to target 16kHz
        target_sr = self.config["sample_rate"]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        # Apply Waveform Augmentation if training
        if self.is_training:
            waveform = apply_waveform_augmentations(waveform, sample_rate=target_sr)

        # Fitting to target window length (2 seconds)
        target_samples = int(target_sr * self.config["window_size_seconds"])
        current_samples = waveform.shape[-1]

        if current_samples > target_samples:
            if self.is_training:
                max_start = current_samples - target_samples
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start_idx = (current_samples - target_samples) // 2
            waveform = waveform[:, start_idx : start_idx + target_samples]
        elif current_samples < target_samples:
            pad_amount = target_samples - current_samples
            waveform = F.pad(waveform, (0, pad_amount))

        # Extract Log-Mel Spectrogram (1, 64, T)
        spec = self.spec_extractor(waveform)

        # Apply SpecAugment if training
        if self.is_training and self.spec_augment is not None:
            spec = self.spec_augment(spec)

        return spec, torch.tensor(mapped_id, dtype=torch.long)

def get_dataloaders_v2(train_csv, val_csv, test_csv, batch_size=32, active_class_ids=None):
    train_dataset = EchoDatasetV2(train_csv, is_training=True, active_class_ids=active_class_ids)
    val_dataset   = EchoDatasetV2(val_csv, is_training=False, active_class_ids=active_class_ids)
    test_dataset  = EchoDatasetV2(test_csv, is_training=False, active_class_ids=active_class_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_csv = os.path.join(ROOT, "model", "data", "splits", "train.csv")
    val_csv   = os.path.join(ROOT, "model", "data", "splits", "val.csv")
    test_csv  = os.path.join(ROOT, "model", "data", "splits", "test.csv")

    if os.path.exists(train_csv):
        print("Testing DatasetV2 ...")
        train_loader, val_loader, test_loader = get_dataloaders_v2(
            train_csv, val_csv, test_csv, batch_size=16, active_class_ids=[0, 1, 4]
        )
        spec_batch, label_batch = next(iter(train_loader))
        print(f"Batch Spec Shape: {spec_batch.shape}, Batch Label Shape: {label_batch.shape}")
        print(f"Unique mapped labels in batch: {torch.unique(label_batch).tolist()}")
        print("DatasetV2 Verification PASSED!")
