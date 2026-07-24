import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import urllib.request
import zipfile
import csv

# Mappings from ESC-50 class names to our 8 hazard classes
ESC50_MAPPING = {
    "siren": "siren",
    "dog": "dog_barking",
    "glass_breaking": "glass_breaking",
    "crying_baby": "shouting", # proxy
}

def setup_directories():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
    os.makedirs(base_path, exist_ok=True)
    classes = ["normal", "gunshot", "explosion", "shouting", "glass_breaking", "fire_alarm", "siren", "dog_barking"]
    for c in classes:
        os.makedirs(os.path.join(base_path, c), exist_ok=True)
    return base_path

def ingest_esc50(demo_only=False, base_path=""):
    print("Downloading ESC-50 zip from Github...")
    zip_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    temp_zip = "esc50_temp.zip"
    extract_dir = "esc50_temp_dir"
    
    if not os.path.exists(temp_zip):
        urllib.request.urlretrieve(zip_url, temp_zip)
    
    print("Extracting ESC-50...")
    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    csv_path = os.path.join(extract_dir, "ESC-50-master", "meta", "esc50.csv")
    audio_dir = os.path.join(extract_dir, "ESC-50-master", "audio")
    
    counts = {k: 0 for k in ESC50_MAPPING.values()}
    max_per_class = 3 if demo_only else 40
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            esc_class = row['category']
            filename = row['filename']
            
            if esc_class in ESC50_MAPPING:
                target_class = ESC50_MAPPING[esc_class]
                if counts[target_class] < max_per_class:
                    source_path = os.path.join(audio_dir, filename)
                    
                    audio_array, sr = sf.read(source_path)
                    
                    # Ensure mono
                    if audio_array.ndim > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Resample to 16kHz
                    if sr != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    
                    # Save
                    out_filename = f"{target_class}_esc50_{counts[target_class]:03d}.wav"
                    save_path = os.path.join(base_path, target_class, out_filename)
                    sf.write(save_path, audio_array, 16000)
                    counts[target_class] += 1
                    print(f"Saved {save_path}")
                    
    # Cleanup
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(temp_zip):
        os.remove(temp_zip)

def ingest_local_raw(base_path):
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw"))
    self_recorded_dir = os.path.join(raw_dir, "self_recorded")
    urbansound_dir = os.path.join(raw_dir, "UrbanSound8K")
    
    os.makedirs(self_recorded_dir, exist_ok=True)
    os.makedirs(urbansound_dir, exist_ok=True)
    
    print("\n--- Local Dataset Processing ---")
    print("Scanning self_recorded for new files...")
    
    for root, dirs, files in os.walk(self_recorded_dir):
        class_name = os.path.basename(root)
        if class_name in ["normal", "gunshot", "explosion", "shouting", "glass_breaking", "fire_alarm", "siren", "dog_barking"]:
            for file in files:
                if file.endswith(".wav"):
                    source_path = os.path.join(root, file)
                    target_path = os.path.join(base_path, class_name, f"custom_{file}")
                    if not os.path.exists(target_path):
                        print(f"Processing custom file: {source_path}")
                        waveform, sr = sf.read(source_path)
                        if waveform.ndim > 1:
                            waveform = np.mean(waveform, axis=1)
                        if sr != 16000:
                            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                        sf.write(target_path, waveform, 16000)

def ingest_urbansound8k(base_path, max_per_class=400):
    print("Downloading UrbanSound8K tar.gz from Zenodo...")
    tar_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    temp_tar = "urbansound8k_temp.tar.gz"
    extract_dir = "urbansound8k_temp_dir"
    
    import requests
    if not os.path.exists(temp_tar):
        print(f"Starting download of {tar_url} (this may take a while)...")
        response = requests.get(tar_url, stream=True)
        if response.status_code == 200:
            with open(temp_tar, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Failed to download UrbanSound8K. Status Code: {response.status_code}")
            return
            
    print("Extracting UrbanSound8K...")
    import tarfile
    with tarfile.open(temp_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        
    csv_path = os.path.join(extract_dir, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
    audio_dir = os.path.join(extract_dir, "UrbanSound8K", "audio")
    
    us8k_mapping = {
        "siren": "siren",
        "dog_bark": "dog_barking",
        "gun_shot": "gunshot"
    }
    
    counts = {k: 0 for k in us8k_mapping.values()}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            us_class = row['class']
            if us_class in us8k_mapping:
                target_class = us8k_mapping[us_class]
                if counts[target_class] < max_per_class:
                    fold = f"fold{row['fold']}"
                    filename = row['slice_file_name']
                    source_path = os.path.join(audio_dir, fold, filename)
                    
                    if os.path.exists(source_path):
                        audio_array, sr = sf.read(source_path)
                        if audio_array.ndim > 1:
                            audio_array = np.mean(audio_array, axis=1)
                        if sr != 16000:
                            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                            
                        out_filename = f"{target_class}_us8k_{counts[target_class]:03d}.wav"
                        save_path = os.path.join(base_path, target_class, out_filename)
                        sf.write(save_path, audio_array, 16000)
                        counts[target_class] += 1
                        if counts[target_class] % 50 == 0:
                            print(f"Processed {counts[target_class]} {target_class} files from UrbanSound8K...")
                            
    # Cleanup
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(temp_tar):
        os.remove(temp_tar)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-only", action="store_true", help="Only download a few ESC-50 samples")
    parser.add_argument("--download-urbansound", action="store_true", help="Download and extract the 6GB UrbanSound8K dataset")
    args = parser.parse_args()
    
    base_path = setup_directories()
    
    try:
        ingest_esc50(args.demo_only, base_path)
    except Exception as e:
        print(f"Error processing ESC-50 from Github: {e}")
        
    if args.download_urbansound:
        try:
            ingest_urbansound8k(base_path, max_per_class=200 if not args.demo_only else 5)
        except Exception as e:
            print(f"Error processing UrbanSound8K: {e}")
            
    ingest_local_raw(base_path)
    print("Ingestion complete!")
