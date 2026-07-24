import os
import torch
import torchaudio
from dataset import EchoDataset, PREPROCESSING_CONFIG, get_dataloaders

import soundfile as sf

def create_dummy_audio(filename, sr=16000, duration=3):
    # Create a simple sine wave as dummy audio
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    sf.write(filename, waveform[0].numpy(), sr)

def test_pipeline():
    print("Testing Audio Data Pipeline...")
    
    # Create some dummy files
    os.makedirs("data/dummy", exist_ok=True)
    create_dummy_audio("data/dummy/test1.wav")
    create_dummy_audio("data/dummy/test2.wav", duration=1.5) # test padding
    create_dummy_audio("data/dummy/test3.wav", duration=5.0) # test truncation
    
    dummy_data = [
        ("data/dummy/test1.wav", "gunshot"),
        ("data/dummy/test2.wav", "scream"),
        ("data/dummy/test3.wav", "normal")
    ]
    
    train_loader, _, _ = get_dataloaders(dummy_data, dummy_data, dummy_data, batch_size=2)
    
    for batch_idx, (log_mel, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Log-Mel Spectrogram Shape: {log_mel.shape} -> (batch, channel, n_mels, T)")
        print(f"Labels: {labels}")
        
        # Expected shape: (batch_size, 1, 64, T)
        assert log_mel.shape[1] == 1, "Channel dimension must be 1"
        assert log_mel.shape[2] == 3 * PREPROCESSING_CONFIG["n_mels"], f"Mel bins must be {3 * PREPROCESSING_CONFIG['n_mels']}"
        
        # target_T = target_length / hop_length + 1
        expected_T = int(PREPROCESSING_CONFIG["sample_rate"] * PREPROCESSING_CONFIG["window_size_seconds"] / PREPROCESSING_CONFIG["hop_length"]) + 1
        print(f"Expected T: {expected_T}, Actual T: {log_mel.shape[3]}")
        
        break
        
    print("Pipeline Test Passed Successfully!")

if __name__ == "__main__":
    test_pipeline()
