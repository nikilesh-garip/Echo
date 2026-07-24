import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from model import EchoCRNN
from dataset import PREPROCESSING_CONFIG, CLASS_MAPPING

class TwoPassDetector:
    def __init__(self, model_path, config=PREPROCESSING_CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EchoCRNN(num_classes=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Invert class mapping for human readable labels
        self.idx_to_class = {v: k for k, v in CLASS_MAPPING.items()}
        
    def _extract_log_mel(self, waveform):
        # Extract mel spectrogram using torchaudio
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["sample_rate"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            n_mels=self.config["n_mels"]
        )
        mel_spec = mel_spectrogram(waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return log_mel_spec # (1, 64, T)

    def preprocess_audio(self, audio_np, sr, target_seconds):
        """
        Preprocesses a raw numpy array to the model input format
        """
        # Ensure mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
            
        # Resample if needed
        if sr != self.config["sample_rate"]:
            # Quick linear resampling using numpy
            num_samples = int(len(audio_np) * self.config["sample_rate"] / sr)
            audio_np = np.interp(
                np.linspace(0, len(audio_np), num_samples, endpoint=False),
                np.arange(len(audio_np)),
                audio_np
            )
            
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0) # (1, num_samples)
        
        # Fit to target window length (pad or truncate)
        target_length = int(self.config["sample_rate"] * target_seconds)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Extract Log-Mel
        log_mel = self._extract_log_mel(waveform) # (1, 64, T)
        return log_mel.unsqueeze(0).to(self.device) # (1, 1, 64, T)

    def run_pass_1(self, audio_2s_np, sr):
        """
        Pass 1: Primary detection with a 2-second window
        Threshold: 0.50
        """
        input_tensor = self.preprocess_audio(audio_2s_np, sr, target_seconds=2.0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            
        # Find highest non-normal hazard class probability
        max_prob = 0.0
        candidate_idx = 0
        
        # idx 0 is "normal"
        for idx in range(1, 8):
            prob = probs[idx].item()
            if prob > max_prob:
                max_prob = prob
                candidate_idx = idx
                
        # Print probabilities for debugging
        print("\n--- Pass 1 Detection Probabilities ---")
        for i in range(8):
            print(f"  {self.idx_to_class[i]}: {probs[i].item():.4f}")
            
        candidate_class = self.idx_to_class[candidate_idx]
        print(f"  Selected candidate: {candidate_class if max_prob >= 0.50 else 'normal'} (max_hazard_prob={max_prob:.4f})")
        
        # If candidate exceeds 0.50, return it
        if max_prob >= 0.50:
            return True, candidate_class, max_prob
            
        # Return normal class
        return False, "normal", probs[0].item()

    def run_pass_2(self, audio_5s_np, sr, target_class):
        """
        Pass 2: Verification detection centered around candidate event (5-second window)
        Threshold: 0.70
        """
        input_tensor = self.preprocess_audio(audio_5s_np, sr, target_seconds=5.0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            
        target_idx = CLASS_MAPPING.get(target_class, 0)
        verification_prob = probs[target_idx].item()
        
        is_verified = verification_prob >= 0.70
        return is_verified, verification_prob
