import os
import time
import urllib.request
import zipfile
import struct
import py7zr

# Define paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FSD50K_DIR = os.path.join(RAW_DIR, "FSD50K")
VOICE_DIR = os.path.join(RAW_DIR, "VOICe")

# Ensure target directories exist
os.makedirs(FSD50K_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)

# Dataset URLs and target file names
FSD50K_GT_URL = "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip"
FSD50K_Z01_URL = "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01"
FSD50K_ZIP_URL = "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip"
VOICE_URL = "https://zenodo.org/records/3514950/files/VOICe_clean.7z"

def download_file(name, url, dest_path):
    print(f"\n--- Downloading {name} ---")
    print(f"Source: {url}")
    print(f"Destination: {dest_path}")
    
    # If file exists, check size or skip
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}. Skipping download.")
        return True
        
    start_time = time.time()
    
    def reporthook(count, block_size, total_size):
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
        percent = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
        # Print progress update every ~10MB or at completion
        if count % 2000 == 0 or percent == 100:
            print(f"... {percent}% completed. {progress_size / (1024*1024):.1f} MB. Speed: {speed} KB/s. Elapsed: {duration:.1f}s", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"Successfully downloaded {name} in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return False

def patch_zip_eocd(zip_path):
    print(f"Patching EOCD header: {zip_path}")
    if not os.path.exists(zip_path):
        print("Zip file does not exist.")
        return False
        
    size = os.path.getsize(zip_path)
    with open(zip_path, "r+b") as f:
        # Search EOCD signature in last 65536 bytes
        search_range = min(size, 65536)
        f.seek(size - search_range)
        data = f.read(search_range)
        
        sig = b"PK\x05\x06"
        pos = data.rfind(sig)
        if pos == -1:
            print("EOCD signature not found.")
            return False
            
        eocd_offset = size - search_range + pos
        
        # Patch the disk number and central directory disk number fields to 0
        f.seek(eocd_offset + 4)
        f.write(struct.pack("<HH", 0, 0))
        print(f"Patched EOCD disk numbers to 0 at offset {eocd_offset}")
        return True

def monkeypatch_zip64():
    """Monkey-patches zipfile._EndRecData64 to ignore spanned checks."""
    print("Applying EOCD64 monkey-patch to zipfile module...")
    
    def patched_EndRecData64(fpin, offset, endrec):
        try:
            fpin.seek(offset - zipfile.sizeEndCentDir64Locator, 2)
        except OSError:
            return endrec

        data = fpin.read(zipfile.sizeEndCentDir64Locator)
        if len(data) != zipfile.sizeEndCentDir64Locator:
            return endrec
        sig, diskno, reloff, disks = struct.unpack(zipfile.structEndArchive64Locator, data)
        if sig != zipfile.stringEndArchive64Locator:
            return endrec

        # Bypass check: if diskno != 0 or disks > 1: raise BadZipFile(...)
        
        # Read the actual ZIP64 EOCD record
        fpin.seek(offset - zipfile.sizeEndCentDir64Locator - zipfile.sizeEndCentDir64, 2)
        data = fpin.read(zipfile.sizeEndCentDir64)
        if len(data) != zipfile.sizeEndCentDir64:
            return endrec
        sig, sz, create_version, read_version, disk_num, disk_dir, \
            dircount, dircount2, dirsize, diroffset = \
            struct.unpack(zipfile.structEndArchive64, data)
        if sig != zipfile.stringEndArchive64:
            return endrec

        # Override disk information to simulate a single-volume ZIP64
        endrec[zipfile._ECD_SIGNATURE] = sig
        endrec[zipfile._ECD_DISK_NUMBER] = 0
        endrec[zipfile._ECD_DISK_START] = 0
        endrec[zipfile._ECD_ENTRIES_THIS_DISK] = dircount2
        endrec[zipfile._ECD_ENTRIES_TOTAL] = dircount2
        endrec[zipfile._ECD_SIZE] = dirsize
        endrec[zipfile._ECD_OFFSET] = diroffset
        return endrec
        
    zipfile._EndRecData64 = patched_EndRecData64

def extract_spanned_zip(z01_path, zip_path, extract_dest):
    print(f"\n--- Processing Spanned Archive Ingest ---")
    print(f"Parts: {z01_path} and {zip_path}")
    
    combined_zip = os.path.join(os.path.dirname(zip_path), "FSD50K.eval_audio.combined.zip")
    
    # 1. Concatenate parts
    if not os.path.exists(combined_zip):
        print(f"Concatenating split parts to: {combined_zip} ...")
        t0 = time.time()
        with open(combined_zip, "wb") as outfile:
            for part in [z01_path, zip_path]:
                print(f"Appending {part}...")
                with open(part, "rb") as infile:
                    while True:
                        chunk = infile.read(1024 * 1024 * 16) # 16MB chunks
                        if not chunk:
                            break
                        outfile.write(chunk)
        print(f"Successfully concatenated files in {time.time() - t0:.1f}s.")
    else:
        print(f"Combined file already exists: {combined_zip}. Skipping concatenation.")

    # 2. Patch EOCD
    if not patch_zip_eocd(combined_zip):
        print("Failed to patch ZIP headers. Extraction may fail.")
        
    # 3. Apply Monkeypatch
    monkeypatch_zip64()

    # 4. Extract
    print(f"Extracting combined ZIP to: {extract_dest} ...")
    t0 = time.time()
    try:
        with zipfile.ZipFile(combined_zip) as z_ref:
            print("Adjusting offsets for Disk 0 files...")
            for info in z_ref.filelist:
                if info.volume == 0:
                    info.header_offset -= 3221225472
            print("Re-computing _end_offset boundaries...")
            end_offset = z_ref.start_dir
            for info in sorted(z_ref.filelist, key=lambda info: info.header_offset, reverse=True):
                info._end_offset = end_offset
                end_offset = info.header_offset
            z_ref.extractall(extract_dest)
        print(f"Successfully extracted spanned ZIP in {time.time() - t0:.1f}s.")
        
        # Cleanup split archives and combined temp file
        print("Cleaning up temporary archives...")
        for p in [z01_path, zip_path, combined_zip]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed: {p}")
        return True
    except Exception as e:
        print(f"Error extracting combined ZIP: {e}")
        return False

def extract_7z(file_path, dest_dir):
    print(f"\n--- Extracting VOICe Dataset (7z) ---")
    print(f"Source: {file_path}")
    print(f"Target Directory: {dest_dir}")
    
    t0 = time.time()
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as sz_ref:
            sz_ref.extractall(dest_dir)
        print(f"Successfully extracted VOICe clean set in {time.time() - t0:.1f}s.")
        
        # Cleanup archive
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        return True
    except Exception as e:
        print(f"Error during VOICe extraction: {e}")
        return False

def main():
    print("==================================================")
    print("ECHO DATASET DOWNLOADER & EXTRACTOR PIPELINE")
    print("==================================================")
    
    # 1. Download and extract FSD50K Ground Truth
    gt_zip = os.path.join(FSD50K_DIR, "FSD50K.ground_truth.zip")
    if download_file("FSD50K Ground Truth", FSD50K_GT_URL, gt_zip):
        print("Extracting FSD50K Ground Truth...")
        try:
            with zipfile.ZipFile(gt_zip, 'r') as z:
                z.extractall(FSD50K_DIR)
            print("Successfully extracted Ground Truth.")
            if os.path.exists(gt_zip):
                os.remove(gt_zip)
        except Exception as e:
            print(f"Error unzipping Ground Truth: {e}")

    # 2. Download FSD50K Eval audio parts
    z01_path = os.path.join(FSD50K_DIR, "FSD50K.eval_audio.z01")
    zip_path = os.path.join(FSD50K_DIR, "FSD50K.eval_audio.zip")
    
    # Verify split file existence or download them
    success_z01 = download_file("FSD50K Eval Audio Part 1 (.z01)", FSD50K_Z01_URL, z01_path)
    success_zip = download_file("FSD50K Eval Audio Part 2 (.zip)", FSD50K_ZIP_URL, zip_path)
    
    if success_z01 and success_zip:
        extract_spanned_zip(z01_path, zip_path, FSD50K_DIR)
    else:
        print("FSD50K audio split files download failed. Skipping extraction.")

    # 3. Download and extract VOICe clean set
    voice_7z = os.path.join(VOICE_DIR, "VOICe_clean.7z")
    if download_file("VOICe Clean Dataset", VOICE_URL, voice_7z):
        extract_7z(voice_7z, VOICE_DIR)
    else:
        print("VOICe Clean Dataset download failed. Skipping extraction.")

    print("\nDataset pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
