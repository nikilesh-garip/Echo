"""
ECHO — Phase 6: Build TRAIN / VAL / TEST Manifest CSVs
=======================================================
Produces three CSVs:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Each row: absolute_wav_path, echo_class_name, echo_class_id, source_dataset

Rules enforced:
  - US8K folds 1-8 → train; fold 9 → val; fold 10 → test
  - FSD50K dev → train/val (15% val, stratified by class, NO cross-clip shuffling)
  - FSD50K eval → test ONLY
  - VOICe windows extracted per official file-split (train/val/test files)
  - Fireworks, babycry, glassbreak clips → EXCLUDED (not written to any split)
  - Normal class capped at 3× the largest hazard class count to prevent extreme imbalance
    (weight-adjusted loss covers the rest)
  - Writes a summary imbalance report to reports/split_summary.txt

Run from project root:
    python model/build_manifests.py [--voice-extract] [--normal-cap N]

--voice-extract : also run VOICe windowed extraction (slow; ~57 hrs audio)
                  omit this flag to skip VOICe on first run
--normal-cap N  : cap normal class at N samples (default: 4000 for first baseline)
"""

import os
import csv
import sys
import yaml
import random
import argparse
import soundfile as sf
from collections import defaultdict, Counter

# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
US8K_CSV      = os.path.join(ROOT, "model", "data", "raw", "UrbanSound8K", "UrbanSound8K.csv")
US8K_AUDIO    = os.path.join(ROOT, "model", "data", "raw", "UrbanSound8K")
FSD50K_DEV    = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "dev.csv")
FSD50K_EVAL   = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "eval.csv")
FSD50K_VOCAB  = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "vocabulary.csv")
FSD50K_AUDIO  = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.eval_audio")
VOICE_ANN     = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "annotation")
VOICE_AUDIO   = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "audio")
VOICE_TRAIN   = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_training.txt")
VOICE_VAL     = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_validation.txt")
VOICE_TEST    = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_test.txt")
MAPPING_YAML  = os.path.join(ROOT, "config", "class_mapping.yaml")
OUT_DIR       = os.path.join(ROOT, "model", "data", "splits")
VOICE_WIN_DIR = os.path.join(ROOT, "model", "data", "voice_windows")

# ── Class config (mirrors class_mapping.yaml) ─────────────────────────────
CLASS_ID = {
    "normal":         0,
    "gunshot":        1,
    "explosion":      2,
    "human_distress": 3,
    "siren":          4,
    "fire_alarm":     5,
}
EXCLUDED = {"fireworks", "glass_breaking", "babycry", "EXCLUDED"}

# ── FSD50K filter rules ────────────────────────────────────────────────────
FIREWORKS_EXCL  = {"Fireworks"}
MUSIC_EXCL      = {"Music"}
ANIMAL_EXCL     = {"Animal"}
VEHICLE_EXCL    = {"Vehicle", "Car", "Ringtone", "Doorbell",
                   "Bicycle_bell", "Clock", "Motor_vehicle_(road)",
                   "Race_car_and_auto_racing"}

def fsd50k_label_to_echo(labels_set):
    """
    Given a frozenset of FSD50K label names for one clip,
    return the Echo class name, or None if excluded/unmappable.
    Priority: hazard classes first, then normal.
    """
    # Immediately exclude if Fireworks is present anywhere
    if FIREWORKS_EXCL & labels_set:
        return None

    # GUNSHOT
    if "Gunshot_and_gunfire" in labels_set:
        if not (MUSIC_EXCL & labels_set) and not (ANIMAL_EXCL & labels_set):
            return "gunshot"
        return None  # music/animal context → exclude

    # EXPLOSION (only if no Fireworks — already checked above)
    if "Explosion" in labels_set:
        if not (MUSIC_EXCL & labels_set):
            return "explosion"
        return None

    # HUMAN_DISTRESS
    if labels_set & {"Screaming", "Shout", "Yell"}:
        # exclude concert-crowd context
        if "Crowd" in labels_set and "Music" in labels_set:
            return None
        return "human_distress"

    # SIREN
    if "Siren" in labels_set:
        return "siren"

    # FIRE_ALARM (strict: Alarm without vehicle/car/ringtone/etc.)
    if "Alarm" in labels_set:
        if not (VEHICLE_EXCL & labels_set):
            return "fire_alarm"
        return None  # car/phone alarm → exclude

    # NORMAL candidates (speech, music, crowd — no hazard co-labels)
    hazard_labels = {"Gunshot_and_gunfire","Explosion","Screaming","Shout","Yell",
                     "Siren","Alarm","Shatter","Glass","Fire"}
    normal_triggers = {"Speech","Male_speech_and_man_speaking",
                       "Female_speech_and_woman_speaking","Conversation",
                       "Laughter","Music","Crowd","Traffic_noise_and_roadway_noise",
                       "Dog","Rain","Walk_and_footsteps","Typing"}
    if (labels_set & normal_triggers) and not (labels_set & hazard_labels):
        return "normal"

    return None  # unmappable → skip

def load_vocab(vocab_path):
    id_to_name = {}
    with open(vocab_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                id_to_name[row[0]] = row[1]
    return id_to_name

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}

# ──────────────────────────────────────────────────────────────────────────
# US8K ingestion
# ──────────────────────────────────────────────────────────────────────────
US8K_CLASS_MAP = {
    "gun_shot":         "gunshot",
    "siren":            "siren",
    "dog_bark":         "normal",
    "children_playing": "normal",
    "air_conditioner":  "normal",
    "street_music":     "normal",
    "engine_idling":    "normal",
    "jackhammer":       "normal",
    "drilling":         "normal",
    "car_horn":         "normal",
}

def ingest_us8k():
    """Returns dict: split → list of (abs_path, echo_class, source)."""
    print("[US8K] Ingesting UrbanSound8K ...")
    splits = defaultdict(list)

    if not os.path.exists(US8K_CSV):
        print("  [SKIP] UrbanSound8K CSV not found.")
        return splits

    train_folds = {str(i) for i in range(1, 9)}
    val_folds   = {"9"}
    test_folds  = {"10"}

    with open(US8K_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold      = row["fold"]
            us_class  = row["class"]
            fname     = row["slice_file_name"]
            echo_cls  = US8K_CLASS_MAP.get(us_class)
            if echo_cls is None:
                continue

            wav_path = os.path.join(US8K_AUDIO, f"fold{fold}", fname)
            if not os.path.exists(wav_path):
                continue

            if fold in train_folds:
                split = "train"
            elif fold in val_folds:
                split = "val"
            else:
                split = "test"

            splits[split].append((wav_path, echo_cls, "US8K"))

    for s, rows in splits.items():
        print(f"  {s}: {len(rows)} clips")
    return splits

# ──────────────────────────────────────────────────────────────────────────
# FSD50K ingestion
# ──────────────────────────────────────────────────────────────────────────
def ingest_fsd50k():
    """Returns dict: split → list of (abs_path, echo_class, source)."""
    print("[FSD50K] Ingesting FSD50K ...")
    splits = defaultdict(list)

    vocab = load_vocab(FSD50K_VOCAB)

    def process_csv(csv_path, split_name, audio_dir):
        count = Counter()
        skipped = 0
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found.")
            return
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname  = row["fname"].strip()
                labels = frozenset(
                    vocab.get(l.strip(), l.strip())
                    for l in row["labels"].split(",")
                )
                echo_cls = fsd50k_label_to_echo(labels)
                if echo_cls is None:
                    skipped += 1
                    continue

                # FSD50K eval audio is in FSD50K.eval_audio/
                # FSD50K dev audio — check common locations
                wav_path = os.path.join(audio_dir, fname + ".wav")
                if not os.path.exists(wav_path):
                    # try without .wav
                    wav_path = os.path.join(audio_dir, fname)
                if not os.path.exists(wav_path):
                    skipped += 1
                    continue

                splits[split_name].append((wav_path, echo_cls, f"FSD50K_{split_name}"))
                count[echo_cls] += 1

        print(f"  {split_name}: {sum(count.values())} accepted, {skipped} skipped")
        for cls, n in sorted(count.items()):
            print(f"    {cls}: {n}")

    # Dev: find audio directory
    fsd50k_dev_audio = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.dev_audio")
    if not os.path.exists(fsd50k_dev_audio):
        # Check if dev audio is alongside eval_audio
        fsd50k_dev_audio = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.eval_audio")
        print(f"  [WARN] FSD50K dev audio dir not found; checking eval_audio dir for both")

    process_csv(FSD50K_DEV,  "dev",  fsd50k_dev_audio)
    process_csv(FSD50K_EVAL, "eval", FSD50K_AUDIO)

    # Dev → stratified train/val split (85/15 by class)
    dev_rows = splits.pop("dev", [])
    by_class = defaultdict(list)
    for row in dev_rows:
        by_class[row[1]].append(row)

    for cls, rows in by_class.items():
        random.shuffle(rows)
        n_val = max(1, int(0.15 * len(rows)))
        splits["val"].extend(rows[:n_val])
        splits["train"].extend(rows[n_val:])

    # Eval → test
    splits["test"].extend(splits.pop("eval", []))

    for s in ["train", "val", "test"]:
        cnt = Counter(r[1] for r in splits[s])
        print(f"  After split — {s}: {sum(cnt.values())} total | {dict(sorted(cnt.items()))}")

    return splits

# ──────────────────────────────────────────────────────────────────────────
# VOICe windowed extraction
# ──────────────────────────────────────────────────────────────────────────
def extract_voice_windows(split_name, file_set, out_dir):
    """
    Extract 2-second windows from VOICe long-form audio for gunshot events only.
    (babycry and glassbreak excluded per Phase 3 decision.)
    Returns list of (abs_path, echo_class, source).
    """
    WINDOW_S  = 2.0
    MIN_EVT_S = 0.3   # skip events shorter than 300 ms
    TARGET_SR = 16000

    os.makedirs(out_dir, exist_ok=True)
    results = []
    skipped = 0

    for ann_fname in sorted(os.listdir(VOICE_ANN)):
        wav_stem = ann_fname.replace(".txt", ".wav")
        if wav_stem not in file_set:
            continue

        ann_path = os.path.join(VOICE_ANN, ann_fname)
        wav_path = os.path.join(VOICE_AUDIO, wav_stem)
        if not os.path.exists(wav_path):
            skipped += 1
            continue

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"  [ERR] {wav_stem}: {e}")
            skipped += 1
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (estimate — VOICe is 16 kHz)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Parse annotations
        with open(ann_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    t_start = float(parts[0])
                    t_end   = float(parts[1])
                    label   = parts[2].lower()
                except ValueError:
                    continue

                # Only gunshot (babycry and glassbreak excluded)
                if label != "gunshot":
                    continue

                dur = t_end - t_start
                if dur < MIN_EVT_S:
                    skipped += 1
                    continue

                # Extract 2-second window centered on event
                mid    = (t_start + t_end) / 2.0
                w_s    = max(0.0, mid - WINDOW_S / 2)
                w_e    = w_s + WINDOW_S
                total_s = len(audio) / sr

                if w_e > total_s:
                    w_e = total_s
                    w_s = max(0.0, w_e - WINDOW_S)

                s_idx = int(w_s * sr)
                e_idx = int(w_e * sr)
                window = audio[s_idx:e_idx]

                if len(window) < int(MIN_EVT_S * sr):
                    skipped += 1
                    continue

                # Zero-pad to exactly WINDOW_S if needed
                target_len = int(WINDOW_S * sr)
                if len(window) < target_len:
                    import numpy as np
                    window = np.pad(window, (0, target_len - len(window)))
                else:
                    window = window[:target_len]

                out_fname = f"voice_{ann_fname.replace('.txt','')}_{i:05d}.wav"
                out_path  = os.path.join(out_dir, out_fname)
                sf.write(out_path, window, sr)
                results.append((out_path, "gunshot", f"VOICe_{split_name}"))

    print(f"  VOICe {split_name}: {len(results)} gunshot windows extracted, {skipped} skipped")
    return results

# ──────────────────────────────────────────────────────────────────────────
# Main build
# ──────────────────────────────────────────────────────────────────────────
def build_manifests(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    all_splits = {"train": [], "val": [], "test": []}

    # US8K
    us8k_splits = ingest_us8k()
    for s in ["train", "val", "test"]:
        all_splits[s].extend(us8k_splits.get(s, []))

    # FSD50K
    fsd_splits = ingest_fsd50k()
    for s in ["train", "val", "test"]:
        all_splits[s].extend(fsd_splits.get(s, []))

    # VOICe (optional, slow)
    if args.voice_extract:
        print("[VOICe] Extracting windowed gunshot clips ...")
        train_files = read_lines(VOICE_TRAIN)
        val_files   = read_lines(VOICE_VAL)
        test_files  = read_lines(VOICE_TEST)

        for split_name, file_set in [("train", train_files),
                                      ("val",   val_files),
                                      ("test",  test_files)]:
            out_dir = os.path.join(VOICE_WIN_DIR, split_name)
            rows = extract_voice_windows(split_name, file_set, out_dir)
            all_splits[split_name].extend(rows)
    else:
        print("[VOICe] Skipped (pass --voice-extract to include VOICe windows).")

    # Cap normal class in training to prevent extreme imbalance
    normal_cap = args.normal_cap
    train_rows = all_splits["train"]
    normal_train  = [r for r in train_rows if r[1] == "normal"]
    other_train   = [r for r in train_rows if r[1] != "normal"]

    if len(normal_train) > normal_cap:
        random.shuffle(normal_train)
        normal_train = normal_train[:normal_cap]
        print(f"[NORMAL CAP] Capped normal training samples to {normal_cap}")

    all_splits["train"] = normal_train + other_train
    random.shuffle(all_splits["train"])

    # Write CSVs
    fieldnames = ["abs_path", "echo_class", "class_id", "source_dataset"]
    for split in ["train", "val", "test"]:
        out_path = os.path.join(OUT_DIR, f"{split}.csv")
        rows = all_splits[split]
        # Verify files exist and filter missing
        verified = []
        missing  = 0
        for row in rows:
            if os.path.exists(row[0]):
                verified.append(row)
            else:
                missing += 1
        if missing:
            print(f"  [WARN] {split}: {missing} files not found on disk — skipped.")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for abs_path, echo_cls, source in verified:
                writer.writerow({
                    "abs_path":       abs_path,
                    "echo_class":     echo_cls,
                    "class_id":       CLASS_ID[echo_cls],
                    "source_dataset": source,
                })
        print(f"  Written: {out_path}  ({len(verified)} rows)")

    # Summary report
    print("\n" + "=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)
    report_lines = ["ECHO — Data Split Summary", "=" * 60, ""]

    for split in ["train", "val", "test"]:
        rows = all_splits[split]
        cnt  = Counter(r[1] for r in rows)
        src  = Counter(r[2] for r in rows)
        total = sum(cnt.values())
        print(f"\n{split.upper()}  ({total} clips)")
        report_lines.append(f"{split.upper()}  ({total} clips)")
        for cls in sorted(CLASS_ID.keys()):
            n = cnt.get(cls, 0)
            pct = 100 * n / total if total else 0
            line = f"  {cls:<20} {n:5d}  ({pct:4.1f}%)"
            print(line)
            report_lines.append(line)
        print(f"  Source breakdown: {dict(src)}")
        report_lines.append(f"  Source breakdown: {dict(src)}")
        report_lines.append("")

    report_path = os.path.join(ROOT, "reports", "split_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description="Build ECHO data split manifests")
    parser.add_argument("--voice-extract", action="store_true",
                        help="Extract VOICe windowed gunshot clips (slow)")
    parser.add_argument("--normal-cap", type=int, default=4000,
                        help="Max normal training samples (default: 4000)")
    args = parser.parse_args()
    build_manifests(args)
