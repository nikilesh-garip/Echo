import os
import csv
import glob

def generate_metadata():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
    out_csv = os.path.join(base_path, "metadata.csv")
    
    classes = ["normal", "gunshot", "explosion", "shouting", "glass_breaking", "fire_alarm", "siren", "dog_barking"]
    
    data_records = []
    
    for c in classes:
        class_dir = os.path.join(base_path, c)
        if os.path.exists(class_dir):
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            for wav in wav_files:
                data_records.append({"filepath": wav, "label": c})
                
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        # train.py expects file_path,label with no headers
        writer = csv.writer(f)
        for r in data_records:
            writer.writerow([r["filepath"], r["label"]])
            
    print(f"Generated metadata.csv with {len(data_records)} records at {out_csv}")

if __name__ == "__main__":
    generate_metadata()
