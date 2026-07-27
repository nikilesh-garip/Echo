"""
ECHO — Phase 8: Training Augmentations
=======================================
Applies TRAINING-ONLY data augmentations.
Validation and Testing datasets MUST NEVER be augmented.

Augmentations implemented:
  1. Time Shift (Random Roll ±100ms)
  2. Gain Variation (Random ±6dB)
  3. SpecAugment (Frequency Masking & Time Masking on Mel Spectrograms)
"""

import random
import torch
import torch.nn as nn
import torchaudio

class SpecAugment(nn.Module):
    """
    Applies Frequency Masking and Time Masking on Log-Mel Spectrograms.
    Input shape: (..., n_mels, T)
    """
    def __init__(self, freq_mask_max=8, time_mask_max=12):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_max)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_max)

    def forward(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec

def apply_waveform_augmentations(waveform, sample_rate=16000):
    """
    Applies time shift and gain variation to 1D/2D raw waveform tensor.
    waveform shape: (1, num_samples) or (num_samples,)
    """
    # 1. Time Shift (random roll up to 100ms)
    max_shift = int(sample_rate * 0.10) # 100ms = 1600 samples
    shift = random.randint(-max_shift, max_shift)
    waveform = torch.roll(waveform, shifts=shift, dims=-1)

    # 2. Gain Variation (±6 dB -> factor between ~0.50 and ~2.0)
    gain_db = random.uniform(-6.0, 6.0)
    gain_factor = 10 ** (gain_db / 20.0)
    waveform = waveform * gain_factor

    return waveform

if __name__ == "__main__":
    print("Testing Augmentation Module...")
    dummy_spec = torch.randn(1, 1, 64, 63)
    aug = SpecAugment()
    augmented_spec = aug(dummy_spec)
    print(f"SpecAugment output shape: {augmented_spec.shape}")
    assert augmented_spec.shape == dummy_spec.shape
    print("Augmentation Verification PASSED!")
