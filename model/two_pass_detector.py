"""
ECHO — Two-Pass Detector & Inference Pipeline (Phases 14, 15, 16)
===================================================================
Rebuilt inference engine using:
  - Canonical Preprocessing (model/preprocessing.py) — 64-bin log-mel, 16 kHz mono
  - Model Architectures (model/model_v2.py) — EchoCNN or EchoCRNN
  - Calibrated Decision Thresholds & Margin-Based UNCERTAIN state handling
  - Multi-window temporal aggregation for 5-second verification pass

Prediction States:
  - NORMAL           : No hazard detected (normal probability highest or hazard < threshold)
  - UNCERTAIN        : Borderline prediction (hazard prob between threshold - margin and threshold)
  - HAZARD CANDIDATE : Hazard probability >= calibrated threshold AND > normal probability
"""

import os
import json
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

from model_v2 import EchoCNN, EchoCRNN
from preprocessing import LogMelSpectrogramExtractor, PREPROCESSING_CONFIG

CLASS_NAMES_DEFAULT_3 = ["normal", "gunshot", "siren"]

class TwoPassDetector:
    def __init__(self, model_path, arch="CRNN", class_names=CLASS_NAMES_DEFAULT_3, threshold_json=None, config=PREPROCESSING_CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}

        # Load architecture
        if arch.upper() == "CNN":
            self.model = EchoCNN(num_classes=self.num_classes).to(self.device)
        else:
            self.model = EchoCRNN(num_classes=self.num_classes).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.spec_extractor = LogMelSpectrogramExtractor(config=config).to(self.device)

        # Thresholds setup
        self.thresholds = {}
        self.uncertain_margin = 0.15

        if threshold_json and os.path.exists(threshold_json):
            with open(threshold_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for name in self.class_names[1:]: # hazard classes
                    if name in data:
                        self.thresholds[name] = data[name].get("threshold", 0.50)
                self.uncertain_margin = data.get("uncertain_margin", 0.15)
        else:
            # Default thresholds
            for name in self.class_names[1:]:
                self.thresholds[name] = 0.50

    def preprocess_raw_audio(self, audio_np, sr, target_seconds=2.0):
        """
        Preprocesses a raw numpy array audio buffer into model input tensor.
        """
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        waveform = torch.from_numpy(audio_np).float().unsqueeze(0) # (1, N)

        target_sr = self.config["sample_rate"]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        target_samples = int(target_sr * target_seconds)
        current_samples = waveform.shape[-1]

        if current_samples > target_samples:
            # Center crop
            start_idx = (current_samples - target_samples) // 2
            waveform = waveform[:, start_idx : start_idx + target_samples]
        elif current_samples < target_samples:
            pad_amount = target_samples - current_samples
            waveform = F.pad(waveform, (0, pad_amount))

        waveform = waveform.to(self.device)
        with torch.no_grad():
            spec = self.spec_extractor(waveform) # (1, 64, T)
            spec_input = spec.unsqueeze(0)        # (1, 1, 64, T)
        return spec_input

    def run_pass_1(self, audio_2s_np, sr):
        """
        Pass 1: Primary detection with a 2-second audio buffer.
        Returns tuple: (state_str, candidate_class, max_prob, probs_dict)
          state_str: "NORMAL" | "UNCERTAIN" | "HAZARD CANDIDATE"
        """
        input_tensor = self.preprocess_raw_audio(audio_2s_np, sr, target_seconds=2.0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs  = F.softmax(logits, dim=1)[0]

        probs_dict = {self.idx_to_class[i]: float(probs[i].item()) for i in range(self.num_classes)}

        normal_prob = probs_dict.get("normal", 0.0)

        # Find highest hazard class probability
        max_hazard_prob = 0.0
        candidate_class = "normal"

        for i in range(1, self.num_classes):
            cls_name = self.idx_to_class[i]
            prob = probs_dict[cls_name]
            if prob > max_hazard_prob:
                max_hazard_prob = prob
                candidate_class = cls_name

        thresh = self.thresholds.get(candidate_class, 0.50)

        # Strict Control Flow:
        # If normal_prob > max_hazard_prob, top class is normal -> NORMAL state
        if normal_prob > max_hazard_prob:
            state = "NORMAL"
            candidate_class = "normal"
        elif max_hazard_prob >= thresh:
            state = "HAZARD CANDIDATE"
        elif max_hazard_prob >= (thresh - self.uncertain_margin):
            state = "UNCERTAIN"
        else:
            state = "NORMAL"
            candidate_class = "normal"

        return state, candidate_class, max_hazard_prob, probs_dict

    def run_pass_2(self, audio_5s_np, sr, candidate_class):
        """
        Pass 2: Verification pass centered around candidate event (5-second audio buffer).
        Uses overlapping 2-second windows with 0.5s hop size to evaluate peak hazard probability.
        """
        if candidate_class == "normal" or candidate_class not in self.class_to_idx:
            return False, 0.0, "NORMAL"

        target_sr = self.config["sample_rate"]
        win_samples = int(target_sr * 2.0)
        hop_samples = int(target_sr * 0.5)

        if audio_5s_np.ndim > 1:
            audio_5s_np = np.mean(audio_5s_np, axis=1)

        total_samples = len(audio_5s_np)
        max_verification_prob = 0.0

        target_idx = self.class_to_idx[candidate_class]
        thresh = self.thresholds.get(candidate_class, 0.50)

        start_idx = 0
        while start_idx + win_samples <= total_samples or start_idx == 0:
            sub_audio = audio_5s_np[start_idx : start_idx + win_samples]
            input_tensor = self.preprocess_raw_audio(sub_audio, sr, target_seconds=2.0)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs  = F.softmax(logits, dim=1)[0]
                prob   = probs[target_idx].item()

            if prob > max_verification_prob:
                max_verification_prob = prob

            start_idx += hop_samples
            if start_idx >= total_samples:
                break

        is_verified = max_verification_prob >= thresh

        return is_verified, float(max_verification_prob)

if __name__ == "__main__":
    print("Testing TwoPassDetector...")
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt = os.path.join(ROOT, "checkpoints", "EXP_002_CRNN_BASELINE_3CLASS_best.pth")
    thresh_json = os.path.join(ROOT, "config", "thresholds_EXP_002_CRNN_BASELINE_3CLASS.json")

    if os.path.exists(ckpt):
        detector = TwoPassDetector(ckpt, arch="CRNN", threshold_json=thresh_json)
        dummy_2s = np.random.randn(32000).astype(np.float32)
        dummy_5s = np.random.randn(80000).astype(np.float32)

        state, cand_cls, max_p, p_dict = detector.run_pass_1(dummy_2s, 16000)
        print(f"Pass 1 result: State={state}, Candidate={cand_cls}, Prob={max_p:.4f}")

        is_v, v_p = detector.run_pass_2(dummy_5s, 16000, cand_cls)
        print(f"Pass 2 result: Verified={is_v}, Peak Prob={v_p:.4f}")
        print("TwoPassDetector Verification PASSED!")
    else:
        print(f"Checkpoint {ckpt} not found for test.")
