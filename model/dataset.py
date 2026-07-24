import os
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# Global configuration for preprocessing (Do not duplicate this logic anywhere else)
PREPROCESSING_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
    "window_size_seconds": 2, # Will be adjusted based on Primary/Verification passes
}

CLASS_MAPPING = {
    "normal": 0,
    "gunshot": 1,
    "explosion": 2,
    "scream": 3,
    "glass_breaking": 4,
    "fire_alarm": 5,
    "siren": 6,
    "shouting": 7
}

def get_augmented_spectrogram(mel_spec):
    """Computes Sobel and Laplacian derivatives and concatenates them to form the augmented mel-spectrogram."""
    # mel_spec shape: (1, 64, T) or (batch, 1, 64, T). We handle both.
    is_batched = mel_spec.ndim == 4
    if is_batched:
        x = mel_spec
    else:
        x = mel_spec.unsqueeze(0) # (1, 1, 64, T)
    
    # Sobel kernels
    hu = torch.tensor([[-1.0, -2.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  2.0,  1.0]], device=x.device).view(1, 1, 3, 3)
                       
    hv = torch.tensor([[-1.0,  0.0,  1.0],
                       [-2.0,  0.0,  2.0],
                       [-1.0,  0.0,  1.0]], device=x.device).view(1, 1, 3, 3)
                       
    # Laplacian kernel
    hl = torch.tensor([[ 0.0,  1.0,  0.0],
                       [ 1.0, -4.0,  1.0],
                       [ 0.0,  1.0,  0.0]], device=x.device).view(1, 1, 3, 3)
                       
    # Run convolutions with reflection padding to keep dimensions
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    x_u = F.conv2d(x_pad, hu)
    x_v = F.conv2d(x_pad, hv)
    
    # First spatial derivative (Sobel magnitude)
    x_first = torch.sqrt(x_u ** 2 + x_v ** 2 + 1e-8)
    
    # Second spatial derivative (Laplacian response)
    x_second = F.conv2d(x_pad, hl)
    
    # Concatenate along the frequency axis (dim=2)
    augmented = torch.cat([x, x_first, x_second], dim=2)
    
    if is_batched:
        return augmented
    else:
        return augmented.squeeze(0) # (1, 192, T)

class EchoDataset(Dataset):
    def __init__(self, data_list, config=PREPROCESSING_CONFIG, augment=False):
        """
        data_list: list of tuples (file_path, label_str)
        """
        self.data_list = data_list
        self.config = config
        self.augment = augment
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        file_path, label_str = self.data_list[idx]
        
        # Use soundfile directly to avoid torchaudio backend issues
        import soundfile as sf
        waveform_np, sr = sf.read(file_path, dtype='float32')
        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform_np).t()
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.config["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, self.config["sample_rate"])
            waveform = resampler(waveform)
            
        # 2. Augmentation (Time shift, Gain variation)
        if self.augment:
            # Time shift
            shift = int(random.uniform(-0.1, 0.1) * self.config["sample_rate"])
            waveform = torch.roll(waveform, shift, dims=-1)
            
            # Gain variation
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
            
        # Ensure fixed length based on window size (pad or truncate)
        target_length = int(self.config["sample_rate"] * self.config["window_size_seconds"])
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # 3. Log-Mel Spectrogram Extraction (64 mel bins)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["sample_rate"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            n_mels=self.config["n_mels"]
        )
        mel_spec = mel_spectrogram(waveform)
        
        # Convert to log scale
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # 4. Apply Spatial Derivative Feature Augmentation
        augmented_spec = get_augmented_spectrogram(log_mel_spec) # (1, 192, T)
        
        label_idx = CLASS_MAPPING.get(label_str, 0)
        return augmented_spec, torch.tensor(label_idx, dtype=torch.long)

def get_dataloaders(train_data, val_data, test_data, batch_size=32, config=PREPROCESSING_CONFIG):
    train_dataset = EchoDataset(train_data, config=config, augment=True)
    val_dataset = EchoDataset(val_data, config=config, augment=False)
    test_dataset = EchoDataset(test_data, config=config, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
