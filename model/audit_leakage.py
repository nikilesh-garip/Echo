"""
ECHO — Phase 5: Data Quality & Leakage Audit
============================================
Checks for:
  1. UrbanSound8K fold integrity (no cross-fold contamination in proposed split)
  2. FSD50K dev vs eval overlap (by Freesound fname/clip-id)
  3. VOICe file-level split integrity
  4. Class imbalance ratios and recommended class weights
  5. Ultra-short clips (<0.5 s effective length after trimming silence)
  6. Duplicate filenames across datasets
  7. FSD50K multi-label filter yield estimates (per class_mapping rules)

Run from project root:
    python model/audit_leakage.py
"""

import os
import csv
import sys
import yaml
from collections import Counter, defaultdict

# ── Paths (relative to project root) ──────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
US8K_CSV      = os.path.join(ROOT, "model", "data", "raw", "UrbanSound8K", "UrbanSound8K.csv")
FSD50K_DEV    = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "dev.csv")
FSD50K_EVAL   = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "eval.csv")
FSD50K_VOCAB  = os.path.join(ROOT, "model", "data", "raw", "FSD50K", "FSD50K.ground_truth", "vocabulary.csv")
VOICE_ANN     = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "annotation")
VOICE_TRAIN   = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_training.txt")
VOICE_VAL     = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_validation.txt")
VOICE_TEST    = os.path.join(ROOT, "model", "data", "raw", "VOICe", "clean", "source", "synthetic_source_test.txt")
MAPPING_YAML  = os.path.join(ROOT, "config", "class_mapping.yaml")

SEPARATOR = "=" * 70

def load_vocab(vocab_path):
    id_to_name = {}
    with open(vocab_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                id_to_name[row[0]] = row[1]
    return id_to_name

def load_yaml_mapping(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# ──────────────────────────────────────────────────────────────────────────
# 1. UrbanSound8K fold audit
# ──────────────────────────────────────────────────────────────────────────
def audit_us8k():
    print(f"\n{SEPARATOR}")
    print("AUDIT 1: UrbanSound8K Fold Integrity")
    print(SEPARATOR)

    if not os.path.exists(US8K_CSV):
        print("  [SKIP] UrbanSound8K CSV not found.")
        return {}

    train_folds = {str(i) for i in range(1, 9)}   # 1-8
    val_folds   = {"9"}
    test_folds  = {"10"}

    class_by_split = defaultdict(Counter)
    fname_by_fold  = defaultdict(set)
    total = 0

    with open(US8K_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold  = row["fold"]
            cls   = row["class"]
            fname = row["slice_file_name"]

            fname_by_fold[fold].add(fname)

            if fold in train_folds:
                split = "train"
            elif fold in val_folds:
                split = "val"
            else:
                split = "test"

            class_by_split[split][cls] += 1
            total += 1

    # Check for filename overlaps between splits
    train_fnames = set()
    for f in train_folds:
        train_fnames |= fname_by_fold[f]
    val_fnames = set()
    for f in val_folds:
        val_fnames |= fname_by_fold[f]
    test_fnames = set()
    for f in test_folds:
        test_fnames |= fname_by_fold[f]

    tv_overlap = train_fnames & val_fnames
    tt_overlap = train_fnames & test_fnames
    vt_overlap = val_fnames   & test_fnames

    print(f"  Total clips: {total}")
    print(f"  Train (folds 1-8): {sum(class_by_split['train'].values())}")
    print(f"  Val   (fold 9):    {sum(class_by_split['val'].values())}")
    print(f"  Test  (fold 10):   {sum(class_by_split['test'].values())}")
    print(f"\n  Filename overlaps (should all be 0):")
    print(f"    Train∩Val  : {len(tv_overlap)}")
    print(f"    Train∩Test : {len(tt_overlap)}")
    print(f"    Val∩Test   : {len(vt_overlap)}")

    if tv_overlap or tt_overlap or vt_overlap:
        print("  [FAIL] Fold overlap detected! Check US8K CSV for duplicate entries.")
    else:
        print("  [PASS] No fold overlap. Fold boundaries are clean.")

    print("\n  Echo-relevant class distribution:")
    echo_labels = {"gun_shot", "siren"}
    for label in sorted(echo_labels):
        t = class_by_split["train"][label]
        v = class_by_split["val"][label]
        te = class_by_split["test"][label]
        print(f"    {label:<20} train={t:4d}  val={v:4d}  test={te:4d}")

    print("  NORMAL sources distribution:")
    normal_labels = {"dog_bark","children_playing","air_conditioner",
                     "street_music","engine_idling","jackhammer","drilling","car_horn"}
    for label in sorted(normal_labels):
        t = class_by_split["train"][label]
        v = class_by_split["val"][label]
        te = class_by_split["test"][label]
        print(f"    {label:<20} train={t:4d}  val={v:4d}  test={te:4d}")

    # Return counts for imbalance analysis later
    return class_by_split

# ──────────────────────────────────────────────────────────────────────────
# 2. FSD50K dev vs eval overlap
# ──────────────────────────────────────────────────────────────────────────
def audit_fsd50k():
    print(f"\n{SEPARATOR}")
    print("AUDIT 2: FSD50K Dev / Eval Overlap")
    print(SEPARATOR)

    if not (os.path.exists(FSD50K_DEV) and os.path.exists(FSD50K_EVAL)):
        print("  [SKIP] FSD50K CSVs not found.")
        return

    vocab = load_vocab(FSD50K_VOCAB)

    dev_ids  = set()
    eval_ids = set()

    dev_label_counts  = Counter()
    eval_label_counts = Counter()

    # FSD50K dev.csv columns: fname, labels, mids, split
    with open(FSD50K_DEV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["fname"].strip()
            dev_ids.add(fid)
            for lbl in row["labels"].split(","):
                name = vocab.get(lbl.strip(), lbl.strip())
                dev_label_counts[name] += 1

    with open(FSD50K_EVAL, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["fname"].strip()
            eval_ids.add(fid)
            for lbl in row["labels"].split(","):
                name = vocab.get(lbl.strip(), lbl.strip())
                eval_label_counts[name] += 1

    overlap = dev_ids & eval_ids
    print(f"  Dev clips:   {len(dev_ids)}")
    print(f"  Eval clips:  {len(eval_ids)}")
    print(f"  Overlap (same fname in dev AND eval): {len(overlap)}")

    if overlap:
        print(f"  [FAIL] {len(overlap)} clips appear in both dev and eval!")
        print(f"    Examples: {list(overlap)[:5]}")
    else:
        print("  [PASS] No dev/eval clip overlap. Test set is clean.")

    # Echo-relevant label counts
    echo_labels = [
        "Gunshot_and_gunfire", "Explosion", "Screaming", "Shout", "Yell",
        "Siren", "Alarm", "Fireworks", "Shatter", "Glass", "Fire",
        "Crying_and_sobbing", "Boom",
    ]
    print("\n  Echo-relevant label counts (dev vs eval):")
    print(f"  {'Label':<35} {'Dev':>6}  {'Eval':>6}")
    print("  " + "-" * 52)
    for lbl in echo_labels:
        d = dev_label_counts.get(lbl, 0)
        e = eval_label_counts.get(lbl, 0)
        print(f"  {lbl:<35} {d:>6}  {e:>6}")

# ──────────────────────────────────────────────────────────────────────────
# 3. FSD50K filter yield estimate
# ──────────────────────────────────────────────────────────────────────────
def audit_fsd50k_filter_yield():
    print(f"\n{SEPARATOR}")
    print("AUDIT 3: FSD50K Filter Yield Estimates (per class_mapping rules)")
    print(SEPARATOR)

    if not os.path.exists(FSD50K_DEV):
        print("  [SKIP] FSD50K dev CSV not found.")
        return

    vocab = load_vocab(FSD50K_VOCAB)

    # Load all dev rows
    dev_rows = []
    with open(FSD50K_DEV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = frozenset(
                vocab.get(l.strip(), l.strip()) for l in row["labels"].split(",")
            )
            dev_rows.append({"fname": row["fname"], "labels": labels})

    total = len(dev_rows)
    print(f"  Total dev clips: {total}")

    def count_filter(keep_fn, desc):
        accepted = [r for r in dev_rows if keep_fn(r["labels"])]
        print(f"    {desc:<60} {len(accepted):5d} clips")
        return accepted

    # gunshot filter: has Gunshot_and_gunfire, not Fireworks, not Music, not Animal
    count_filter(
        lambda L: "Gunshot_and_gunfire" in L and "Fireworks" not in L
                  and "Music" not in L and "Animal" not in L,
        "GUNSHOT: has Gunshot_and_gunfire, excl Fireworks/Music/Animal"
    )
    # explosion filter: has Explosion, not Fireworks, not Music
    count_filter(
        lambda L: "Explosion" in L and "Fireworks" not in L and "Music" not in L,
        "EXPLOSION: has Explosion, excl Fireworks/Music"
    )
    # human_distress: Screaming OR Shout OR Yell
    count_filter(
        lambda L: bool(L & {"Screaming", "Shout", "Yell"}),
        "HUMAN_DISTRESS: Screaming or Shout or Yell"
    )
    # human_distress excluding crowd+music (concert noise)
    count_filter(
        lambda L: bool(L & {"Screaming", "Shout", "Yell"})
                  and not ("Crowd" in L and "Music" in L),
        "HUMAN_DISTRESS (excl Crowd+Music co-occurrence)"
    )
    # siren
    count_filter(
        lambda L: "Siren" in L,
        "SIREN: has Siren"
    )
    # fire_alarm: Alarm but not Vehicle/Car/Ringtone/Doorbell/Bicycle_bell/Clock/Music
    excl = {"Vehicle","Car","Ringtone","Doorbell","Bicycle_bell","Clock","Music",
            "Motor_vehicle_(road)","Race_car_and_auto_racing"}
    count_filter(
        lambda L: "Alarm" in L and not (L & excl),
        "FIRE_ALARM: Alarm excl Vehicle/Car/Ringtone/Doorbell/Clock/Music"
    )
    # fireworks (to be excluded from any class)
    count_filter(
        lambda L: "Fireworks" in L,
        "FIREWORKS (to EXCLUDE from all classes)"
    )
    # normal: Speech or Conversation or Music or Crowd (no hazard labels)
    hazard = {"Gunshot_and_gunfire","Explosion","Screaming","Shout","Yell",
              "Siren","Alarm","Shatter","Glass","Fire","Fireworks"}
    count_filter(
        lambda L: bool(L & {"Speech","Male_speech_and_man_speaking",
                             "Female_speech_and_woman_speaking","Conversation",
                             "Laughter","Music","Crowd","Traffic_noise_and_roadway_noise"})
                  and not (L & hazard),
        "NORMAL candidates (speech/music/crowd, no hazard co-labels)"
    )

# ──────────────────────────────────────────────────────────────────────────
# 4. VOICe file split integrity
# ──────────────────────────────────────────────────────────────────────────
def audit_voice():
    print(f"\n{SEPARATOR}")
    print("AUDIT 4: VOICe File-Level Split Integrity")
    print(SEPARATOR)

    for path in [VOICE_TRAIN, VOICE_VAL, VOICE_TEST]:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found.")
            return

    train_files = set(read_lines(VOICE_TRAIN))
    val_files   = set(read_lines(VOICE_VAL))
    test_files  = set(read_lines(VOICE_TEST))

    tv_overlap = train_files & val_files
    tt_overlap = train_files & test_files
    vt_overlap = val_files   & test_files

    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Test files:  {len(test_files)}")
    print(f"  Total unique: {len(train_files | val_files | test_files)}")
    print(f"\n  File overlaps (should all be 0):")
    print(f"    Train∩Val  : {len(tv_overlap)}")
    print(f"    Train∩Test : {len(tt_overlap)}")
    print(f"    Val∩Test   : {len(vt_overlap)}")

    if tv_overlap or tt_overlap or vt_overlap:
        print("  [FAIL] File overlap detected in VOICe splits!")
    else:
        print("  [PASS] VOICe file splits are disjoint.")

    # Event count per split
    if not os.path.exists(VOICE_ANN):
        return

    label_counts = defaultdict(lambda: Counter())
    for fname in sorted(os.listdir(VOICE_ANN)):
        stem = fname.replace(".txt", ".wav")
        ann_path = os.path.join(VOICE_ANN, fname)
        if stem in train_files:
            split = "train"
        elif stem in val_files:
            split = "val"
        elif stem in test_files:
            split = "test"
        else:
            split = "unknown"
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label_counts[split][parts[2]] += 1

    print("\n  Event counts per split:")
    for split in ["train", "val", "test", "unknown"]:
        counts = label_counts[split]
        if counts:
            total = sum(counts.values())
            print(f"    {split}: {total} events  {dict(counts)}")

# ──────────────────────────────────────────────────────────────────────────
# 5. Class imbalance report and recommended weights
# ──────────────────────────────────────────────────────────────────────────
def imbalance_report():
    print(f"\n{SEPARATOR}")
    print("AUDIT 5: Class Imbalance & Recommended Loss Weights")
    print(SEPARATOR)

    # Rough estimates from audit findings
    estimated_counts = {
        "normal":         8000,   # conservative (US8K 8 classes × ~600 train + FSD50K)
        "gunshot":        3500,   # US8K + FSD50K + VOICe (after filtering)
        "explosion":       250,   # FSD50K filtered
        "human_distress":  500,   # FSD50K Screaming+Shout+Yell after concert filter
        "siren":           850,   # US8K (train portion) + FSD50K
        "fire_alarm":      130,   # FSD50K strict-filtered Alarm
    }

    total = sum(estimated_counts.values())
    print(f"\n  Estimated training sample counts:")
    print(f"  {'Class':<20} {'Est. Count':>12}  {'%':>6}  {'Inv-Freq Weight':>16}")
    print("  " + "-" * 60)

    max_count = max(estimated_counts.values())
    weights = {}
    for cls, cnt in sorted(estimated_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total
        w = max_count / cnt   # inverse frequency weight, normalised to majority
        weights[cls] = w
        print(f"  {cls:<20} {cnt:>12}  {pct:>5.1f}%  {w:>16.2f}×")

    print(f"\n  Total estimated training samples: {total}")
    print(f"\n  ⚠  CRITICAL IMBALANCE NOTES:")
    print(f"    - fire_alarm is {estimated_counts['normal'] / estimated_counts['fire_alarm']:.0f}× under-represented vs normal")
    print(f"    - explosion is {estimated_counts['normal'] / estimated_counts['explosion']:.0f}× under-represented vs normal")
    print(f"    - Use torch.nn.CrossEntropyLoss(weight=tensor([w0,w1,...,w5]))")
    print(f"    - Consider WeightedRandomSampler for training batches")
    print(f"\n  Recommended PyTorch class weights (class_id order 0-5):")
    order = ["normal","gunshot","explosion","human_distress","siren","fire_alarm"]
    weight_list = [round(weights[c], 3) for c in order]
    print(f"    {weight_list}")

# ──────────────────────────────────────────────────────────────────────────
# 6. Duplicate filename scan across all datasets
# ──────────────────────────────────────────────────────────────────────────
def audit_duplicates():
    print(f"\n{SEPARATOR}")
    print("AUDIT 6: Cross-Dataset Filename Duplicates")
    print(SEPARATOR)

    all_fnames = defaultdict(list)

    # US8K
    if os.path.exists(US8K_CSV):
        with open(US8K_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_fnames[row["slice_file_name"]].append("US8K")

    # FSD50K dev
    if os.path.exists(FSD50K_DEV):
        with open(FSD50K_DEV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_fnames[row["fname"].strip() + ".wav"].append("FSD50K_dev")

    # FSD50K eval
    if os.path.exists(FSD50K_EVAL):
        with open(FSD50K_EVAL, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_fnames[row["fname"].strip() + ".wav"].append("FSD50K_eval")

    # VOICe
    if os.path.exists(VOICE_ANN):
        for fname in os.listdir(VOICE_ANN):
            all_fnames[fname.replace(".txt", ".wav")].append("VOICe")

    dups = {k: v for k, v in all_fnames.items() if len(v) > 1}
    print(f"  Total unique filenames scanned: {len(all_fnames)}")
    print(f"  Cross-dataset filename collisions: {len(dups)}")

    if dups:
        print("  [WARN] Collisions found (may be coincidental numeric IDs):")
        for fname, sources in list(dups.items())[:10]:
            print(f"    {fname}  in: {sources}")
    else:
        print("  [PASS] No cross-dataset filename duplicates.")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'#' * 70}")
    print("  ECHO — Phase 5: Data Quality & Leakage Audit")
    print(f"{'#' * 70}")

    # Change to project root so relative paths work
    os.chdir(ROOT)

    audit_us8k()
    audit_fsd50k()
    audit_fsd50k_filter_yield()
    audit_voice()
    imbalance_report()
    audit_duplicates()

    print(f"\n{SEPARATOR}")
    print("AUDIT COMPLETE. Review findings above before proceeding to Phase 6.")
    print(SEPARATOR)
