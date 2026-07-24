import os
import zipfile
import csv
import shutil
import numpy as np
import soundfile as sf
import librosa
import glob

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RAW_US8K_DIR = os.path.join(BASE_DIR, "data", "raw", "UrbanSound8K")
ZIP_ESC50_PATH = os.path.join(BASE_DIR, "esc50_temp.zip")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

ACTIVE_CLASSES = ["normal", "gunshot", "siren", "glass_breaking", "shouting"]

def setup_directories():
    print("Clearing and setting up processed directories...")
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for c in ACTIVE_CLASSES:
        os.makedirs(os.path.join(PROCESSED_DIR, c), exist_ok=True)

def process_audio_file(source_path, target_path):
    """Loads, converts to mono, resamples to 16kHz, and saves the audio."""
    try:
        audio, sr = sf.read(source_path)
        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Resample to 16000 Hz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Save
        sf.write(target_path, audio, 16000)
        return True
    except Exception as e:
        print(f"Error processing file {source_path}: {e}")
        return False

def ingest_urbansound8k():
    print("\nProcessing UrbanSound8K dataset...")
    csv_path = os.path.join(RAW_US8K_DIR, "UrbanSound8K.csv")
    if not os.path.exists(csv_path):
        print(f"UrbanSound8K metadata not found at: {csv_path}. Skipping UrbanSound8K.")
        return

    # Counts for category balancing
    counts = {
        "gunshot": 0,
        "siren": 0,
        "normal": {}  # dict of sub-category counts to balance background noises
    }
    
    max_normal_per_subclass = 50  # 8 background classes * 50 = 400 normal samples
    max_gunshot = 350
    max_siren = 300

    processed_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            us_class = row['class']
            slice_file_name = row['slice_file_name']
            fold = f"fold{row['fold']}"
            
            source_path = os.path.join(RAW_US8K_DIR, fold, slice_file_name)
            if not os.path.exists(source_path):
                continue

            target_class = None
            if us_class == "gun_shot":
                if counts["gunshot"] < max_gunshot:
                    target_class = "gunshot"
                    counts["gunshot"] += 1
            elif us_class == "siren":
                if counts["siren"] < max_siren:
                    target_class = "siren"
                    counts["siren"] += 1
            else:
                # Map other 8 categories to normal
                subclass_count = counts["normal"].get(us_class, 0)
                if subclass_count < max_normal_per_subclass:
                    target_class = "normal"
                    counts["normal"][us_class] = subclass_count + 1

            if target_class:
                target_path = os.path.join(PROCESSED_DIR, target_class, f"us8k_{slice_file_name}")
                if process_audio_file(source_path, target_path):
                    processed_count += 1

    print(f"UrbanSound8K Ingestion Complete: processed {processed_count} files.")
    print(f"  - Gunshot: {counts['gunshot']}")
    print(f"  - Siren: {counts['siren']}")
    print(f"  - Normal: {sum(counts['normal'].values())} (from background subclasses)")

def ingest_esc50():
    print("\nProcessing ESC-50 dataset...")
    if not os.path.exists(ZIP_ESC50_PATH):
        print(f"ESC-50 zip file not found at: {ZIP_ESC50_PATH}. Skipping ESC-50.")
        return

    extract_dir = os.path.join(BASE_DIR, "esc50_temp_dir")
    print("Extracting ESC-50...")
    with zipfile.ZipFile(ZIP_ESC50_PATH, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    csv_path = os.path.join(extract_dir, "ESC-50-master", "meta", "esc50.csv")
    audio_dir = os.path.join(extract_dir, "ESC-50-master", "audio")

    if not os.path.exists(csv_path):
        print("ESC-50 CSV meta-data file not found inside zip. Cleaning up.")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return

    esc50_mapping = {
        "glass_breaking": "glass_breaking",
        "crying_baby": "shouting",
        "siren": "siren"
    }

    counts = {k: 0 for k in esc50_mapping.values()}
    processed_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['category']
            filename = row['filename']
            
            if category in esc50_mapping:
                target_class = esc50_mapping[category]
                source_path = os.path.join(audio_dir, filename)
                target_path = os.path.join(PROCESSED_DIR, target_class, f"esc50_{filename}")
                
                if os.path.exists(source_path):
                    if process_audio_file(source_path, target_path):
                        counts[target_class] += 1
                        processed_count += 1

    # Cleanup temp extraction folder
    print("Cleaning up ESC-50 temporary directories...")
    shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"ESC-50 Ingestion Complete: processed {processed_count} files.")
    for target_class, count in counts.items():
        print(f"  - {target_class}: {count}")

def generate_metadata_csv():
    print("\nGenerating final metadata.csv...")
    out_csv = os.path.join(PROCESSED_DIR, "metadata.csv")
    records = []
    
    for c in ACTIVE_CLASSES:
        class_dir = os.path.join(PROCESSED_DIR, c)
        if os.path.exists(class_dir):
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            for wav in wav_files:
                # Save as absolute path so train.py can easily load it from any Cwd
                records.append([os.path.abspath(wav), c])

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(records)

    print(f"Dataset preparation complete! Compiled {len(records)} records in {out_csv}")
    print("\nClass Distribution Summary:")
    from collections import Counter
    dist = Counter([r[1] for r in records])
    for cls in ACTIVE_CLASSES:
        print(f"  - {cls}: {dist.get(cls, 0)} files")

if __name__ == "__main__":
    setup_directories()
    ingest_urbansound8k()
    ingest_esc50()
    generate_metadata_csv()
