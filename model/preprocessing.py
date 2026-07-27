"""
ECHO — Phase 7: Canonical Preprocessing Pipeline
=================================================
Single Source of Truth for audio loading and Log-Mel Spectrogram feature extraction.
Identical preprocessing applied across Training, Validation, Testing, and Runtime Two-Pass Inference.

Canonical Parameters:
  - Sample Rate: 16,000 Hz (mono)
  - Window Duration: 2.0 seconds (32,000 samples)
  - N_FFT: 1024
  - Hop Length: 512 (yields ~63 time frames per 2s window)
  - N_MELS: 64
  - F_MIN: 20 Hz, F_MAX: 8000 Hz
  - Scale: AmplitudeToDB (log-mel)
  - Normalization: Per-clip zero-mean unit-variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# Canonical configuration dictionary
PREPROCESSING_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "window_size_seconds": 2.0,
    "target_samples": 32000,  # 16000 * 2.0
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "f_min": 20.0,
    "f_max": 8000.0,
    "normalization": "per_clip_mean_std",
}

class LogMelSpectrogramExtractor(nn.Module):
    """
    Torch module for extracting Log-Mel Spectrograms.
    Can be included directly in the PyTorch model or used in Dataset transforms.
    """
    def __init__(self, config=PREPROCESSING_CONFIG):
        super().__init__()
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=config["n_fft"],
            win_length=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            f_min=config["f_min"],
            f_max=config["f_max"],
            center=True,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform):
        """
        Input shapes supported:
          - (num_samples,)        -> Output: (1, 64, T)
          - (1, num_samples)      -> Output: (1, 64, T)
          - (batch, 1, num_samples)-> Output: (batch, 1, 64, T)
        """
        is_batched = (waveform.ndim == 3)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0) # (1, 1, N)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1) # (1, 1, N)

        mel_spec = self.mel_transform(waveform) # (B, 1, n_mels, T)
        log_mel = self.amplitude_to_db(mel_spec)

        # Per-clip zero-mean unit-variance normalization
        if self.config["normalization"] == "per_clip_mean_std":
            mean = log_mel.mean(dim=(-2, -1), keepdim=True)
            std = log_mel.std(dim=(-2, -1), keepdim=True)
            log_mel = (log_mel - mean) / (std + 1e-6)

        if not is_batched:
            return log_mel.squeeze(0) # (1, 64, T)
        return log_mel # (B, 1, 64, T)

def load_and_preprocess_audio(file_path, target_seconds=2.0, is_training=False, config=PREPROCESSING_CONFIG):
    """
    Loads audio file, converts to mono 16kHz, handles duration fitting (crop/pad),
    and returns a Log-Mel spectrogram tensor.

    Args:
        file_path (str): Absolute path to audio WAV file.
        target_seconds (float): Target window duration (default 2.0s).
        is_training (bool): If True, applies random 2s cropping for audio > 2s.
                            If False, applies center cropping.
        config (dict): Preprocessing parameters.

    Returns:
        torch.Tensor: Spectrogram tensor of shape (1, 64, T).
    """
    import soundfile as sf
    waveform_np, sr = sf.read(file_path, dtype='float32')

    # Convert multi-channel to mono
    if waveform_np.ndim > 1:
        waveform_np = np.mean(waveform_np, axis=1)

    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0) # (1, num_samples)

    # Resample to 16 kHz if necessary
    target_sr = config["sample_rate"]
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

    target_samples = int(target_sr * target_seconds)
    current_samples = waveform.shape[-1]

    # Crop or Pad to target duration
    if current_samples > target_samples:
        if is_training:
            # Random crop during training
            max_start = current_samples - target_samples
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
        else:
            # Center crop during val/test
            start_idx = (current_samples - target_samples) // 2
        waveform = waveform[:, start_idx : start_idx + target_samples]
    elif current_samples < target_samples:
        # Zero padding to target length
        pad_amount = target_samples - current_samples
        waveform = F.pad(waveform, (0, pad_amount))

    extractor = LogMelSpectrogramExtractor(config=config)
    spec = extractor(waveform) # (1, 64, T)
    return spec

if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    dummy_waveform = torch.randn(1, 32000)
    extractor = LogMelSpectrogramExtractor()
    spec = extractor(dummy_waveform)
    print(f"Input waveform shape: {dummy_waveform.shape}")
    print(f"Extracted Log-Mel Spectrogram shape: {spec.shape}")
    assert spec.shape == (1, 64, 63), f"Expected shape (1, 64, 63), got {spec.shape}"
    print("Preprocessing Pipeline Verification PASSED!")
