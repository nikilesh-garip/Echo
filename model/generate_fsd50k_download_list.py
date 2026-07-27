"""
ECHO — Generate FSD50K Dev Needed Files List
==============================================
Identifies exact fnames needed from FSD50K dev.csv based on config/class_mapping.yaml rules.
Outputs:
  - reports/fsd50k_needed_fnames.txt (list of fnames + mapped class)
  - Summary breakdown per class
"""

import os
import csv
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FSD50K_DEV   = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "dev.csv")
FSD50K_VOCAB = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "vocabulary.csv")
OUT_FILE     = os.path.join(ROOT, "reports", "fsd50k_needed_fnames.txt")

FIREWORKS_EXCL  = {"Fireworks"}
MUSIC_EXCL      = {"Music"}
ANIMAL_EXCL     = {"Animal"}
VEHICLE_EXCL    = {"Vehicle", "Car", "Ringtone", "Doorbell",
                   "Bicycle_bell", "Clock", "Motor_vehicle_(road)",
                   "Race_car_and_auto_racing"}

def load_vocab(vocab_path):
    id_to_name = {}
    with open(vocab_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                id_to_name[row[0]] = row[1]
    return id_to_name

def fsd50k_label_to_echo(labels_set):
    if FIREWORKS_EXCL & labels_set:
        return None

    if "Gunshot_and_gunfire" in labels_set:
        if not (MUSIC_EXCL & labels_set) and not (ANIMAL_EXCL & labels_set):
            return "gunshot"
        return None

    if "Explosion" in labels_set:
        if not (MUSIC_EXCL & labels_set):
            return "explosion"
        return None

    if labels_set & {"Screaming", "Shout", "Yell"}:
        if "Crowd" in labels_set and "Music" in labels_set:
            return None
        return "human_distress"

    if "Siren" in labels_set:
        return "siren"

    if "Alarm" in labels_set:
        if not (VEHICLE_EXCL & labels_set):
            return "fire_alarm"
        return None

    hazard_labels = {"Gunshot_and_gunfire","Explosion","Screaming","Shout","Yell",
                     "Siren","Alarm","Shatter","Glass","Fire"}
    normal_triggers = {"Speech","Male_speech_and_man_speaking",
                       "Female_speech_and_woman_speaking","Conversation",
                       "Laughter","Music","Crowd","Traffic_noise_and_roadway_noise",
                       "Dog","Rain","Walk_and_footsteps","Typing"}
    if (labels_set & normal_triggers) and not (labels_set & hazard_labels):
        return "normal"

    return None

def main():
    if not os.path.exists(FSD50K_DEV):
        print(f"Error: {FSD50K_DEV} not found.")
        return

    vocab = load_vocab(FSD50K_VOCAB)
    
    needed_by_class = defaultdict(list)
    total_needed = 0

    with open(FSD50K_DEV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["fname"].strip()
            labels = frozenset(
                vocab.get(l.strip(), l.strip()) for l in row["labels"].split(",")
            )
            echo_cls = fsd50k_label_to_echo(labels)
            if echo_cls is not None:
                needed_by_class[echo_cls].append(fname)
                total_needed += 1

    print("=" * 60)
    print("FSD50K DEV NEEDED CLIPS ANALYSIS")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("# FSD50K Dev Needed Files\n")
        f.write("# Format: fname.wav,echo_class\n\n")
        for cls, fnames in sorted(needed_by_class.items()):
            print(f"  {cls:<20}: {len(fnames):5d} clips")
            for fn in fnames:
                f.write(f"{fn}.wav,{cls}\n")

    print("=" * 60)
    print(f"Total needed FSD50K dev clips: {total_needed}")
    # Estimate size assuming average 250 KB per clip
    est_mb = (total_needed * 250) / 1024
    print(f"Estimated disk size for needed subset: ~{est_mb:.1f} MB (vs ~35 GB for full set)")
    print(f"Output saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()
