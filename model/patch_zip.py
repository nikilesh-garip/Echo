import os
import struct

def patch_zip_eocd(zip_path):
    print(f"Opening ZIP file to patch EOCD: {zip_path}")
    if not os.path.exists(zip_path):
        print("File does not exist.")
        return False
        
    size = os.path.getsize(zip_path)
    
    # We open in read-write binary mode
    with open(zip_path, "r+b") as f:
        # We search for EOCD signature (PK\x05\x06) in the last 65536 bytes (max comment size is 65535)
        search_range = min(size, 65536)
        f.seek(size - search_range)
        data = f.read(search_range)
        
        # Search backwards for the signature PK\x05\x06
        sig = b"PK\x05\x06"
        pos = data.rfind(sig)
        if pos == -1:
            print("EOCD signature not found.")
            return False
            
        eocd_offset = size - search_range + pos
        print(f"Found EOCD signature at offset: {eocd_offset}")
        
        # Read the EOCD fields starting from signature offset
        f.seek(eocd_offset)
        eocd_data = f.read(22) # standard EOCD size (without comment)
        
        # Unpack standard EOCD record
        # Signature (4B), Disk number (2B), Disk w/ CD (2B), Disk records (2B), Total records (2B), CD size (4B), CD offset (4B), Comment len (2B)
        fields = struct.unpack("<IHHHHIIH", eocd_data[:22])
        print("Original EOCD fields:")
        print(f"  Signature: {hex(fields[0])}")
        print(f"  Number of this disk: {fields[1]}")
        print(f"  Disk where central directory starts: {fields[2]}")
        print(f"  Number of central directory records on this disk: {fields[3]}")
        print(f"  Total number of central directory records: {fields[4]}")
        print(f"  Size of central directory: {fields[5]} bytes")
        print(f"  Offset of central directory: {fields[6]}")
        print(f"  Comment length: {fields[7]}")
        
        # Patch the disk number fields to 0, which makes it look like a single-volume ZIP file
        f.seek(eocd_offset + 4)
        f.write(struct.pack("<HH", 0, 0))
        
        print("Successfully patched EOCD disk numbers to 0!")
        return True

if __name__ == "__main__":
    zip_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "raw", "FSD50K", "FSD50K.eval_audio.zip"))
    patch_zip_eocd(zip_file)
