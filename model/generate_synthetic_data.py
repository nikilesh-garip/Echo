import os
import numpy as np
import soundfile as sf

def generate_gunshot(duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Exponential decay of white noise
    decay = np.exp(-15 * t)
    noise = np.random.normal(0, 0.5, len(t))
    signal = noise * decay
    return signal

def generate_explosion(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Low frequency rumble + exponential decay noise
    decay = np.exp(-3 * t)
    noise = np.random.normal(0, 0.4, len(t))
    # Simple low pass approximation
    rumble = np.sin(2 * np.pi * 50 * t) * np.exp(-1 * t)
    signal = (noise * decay) + 0.5 * rumble
    return signal

def generate_scream(duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Harmonic modulated tone around 1kHz to 1.5kHz
    f_mod = 1200 + 300 * np.sin(2 * np.pi * 8 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    # Add screaming vibrato and noise
    signal += np.random.normal(0, 0.2, len(t))
    signal = signal * np.sin(np.pi * t / duration) # fade in/out
    return signal

def generate_glass_breaking(duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Multiple sharp impulses at high frequency
    signal = np.zeros_like(t)
    for _ in range(10):
        start_time = np.random.uniform(0.1, 1.2)
        idx = int(start_time * sr)
        # decaying high frequency tone
        t_decay = t[idx:] - start_time
        decay = np.exp(-25 * t_decay)
        freq = np.random.uniform(3000, 6000)
        signal[idx:] += np.sin(2 * np.pi * freq * t_decay) * decay * np.random.uniform(0.3, 0.8)
    # Add minor noise
    signal += np.random.normal(0, 0.05, len(t))
    return signal

def generate_fire_alarm(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Beeping pure tone (e.g. 3000 Hz)
    beep_freq = 3000
    beep_signal = np.sin(2 * np.pi * beep_freq * t)
    # Square wave envelope (0.3s on, 0.3s off)
    envelope = (np.sin(2 * np.pi * (1 / 0.6) * t) > 0).astype(float)
    signal = beep_signal * envelope
    return signal

def generate_siren(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Wailing siren: FM sweep between 600 and 1200 Hz
    f_mod = 900 + 300 * np.sin(2 * np.pi * 0.5 * t) # 2 sec cycle
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    signal = np.sin(phase)
    return signal

def generate_shouting(duration=2.5, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Modulated low frequency vocal simulation
    f_mod = 250 + 50 * np.sin(2 * np.pi * 10 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    signal = np.sin(phase) + 0.3 * np.sin(3 * phase)
    # Turn it into intermittent bursts of shout
    envelope = (np.sin(2 * np.pi * 1.5 * t) > 0.1).astype(float)
    signal = signal * envelope + np.random.normal(0, 0.15, len(t))
    return signal

def generate_normal(duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Low level white noise (or clean background)
    signal = np.random.normal(0, 0.02, len(t))
    return signal

def build_dataset(base_dir="data/synthetic", samples_per_class=50):
    os.makedirs(base_dir, exist_ok=True)
    
    generators = {
        "gunshot": generate_gunshot,
        "explosion": generate_explosion,
        "scream": generate_scream,
        "glass_breaking": generate_glass_breaking,
        "fire_alarm": generate_fire_alarm,
        "siren": generate_siren,
        "shouting": generate_shouting,
        "normal": generate_normal
    }
    
    metadata = []
    
    for class_name, gen_func in generators.items():
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Generating synthetic sounds for class: {class_name}...")
        
        for i in range(samples_per_class):
            filename = f"{class_name}_{i:03d}.wav"
            filepath = os.path.join(class_dir, filename)
            # Add some variability in duration
            duration = np.random.uniform(2.0, 4.0)
            audio = gen_func(duration=duration)
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9
            sf.write(filepath, audio, 16000)
            
            # Save relative or full path + label
            metadata.append((filepath, class_name))
            
    # Save a metadata txt/csv for dataset loader
    metadata_path = os.path.join(base_dir, "metadata.csv")
    with open(metadata_path, "w") as f:
        for path, label in metadata:
            # write absolute/relative path
            f.write(f"{path},{label}\n")
            
    print(f"Dataset generation complete! Created {len(metadata)} samples in {base_dir}")

if __name__ == "__main__":
    build_dataset()
