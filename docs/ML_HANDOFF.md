# ECHO ML Handoff Document
**Single Source of Truth — Branch: ml/rebuild-audio-pipeline**
**Commit: 97b9ba7**
**Last Updated: 2026-07-27T14:30 IST — Phases 0-6 COMPLETE**

---

## Current Phase: PHASE 7 — Preprocessing Pipeline (NEXT)

**Status:** Phases 0-6 complete and committed. Manifests built.
FSD50K dev audio NOT downloaded. Decision: proceed without it initially,
then download targeted subset if baseline metrics require more data for
explosion/human_distress/fire_alarm/siren.

---

## APPROVED CLASS TABLE (LOCKED — DO NOT CHANGE)

| class_id | Name | Severity | Est. Train Samples | Status |
|---|---|---|---|---|
| 0 | normal | NORMAL | ~4,000 (capped) | GOOD |
| 1 | gunshot | DANGER | ~2,854 (US8K+VOICe) | GOOD |
| 2 | explosion | DANGER | 0 (FSD50K dev missing) | CRITICAL GAP |
| 3 | human_distress | WARNING | 0 (FSD50K dev missing) | CRITICAL GAP |
| 4 | siren | WARNING | ~764 (US8K only) | OK |
| 5 | fire_alarm | DANGER | 0 (FSD50K dev missing) | CRITICAL GAP |

**CRITICAL:** explosion, human_distress, fire_alarm have ZERO training samples.
FSD50K dev audio must be partially downloaded to cover these classes.
See "FSD50K Dev Partial Download" section.

---

## Current Data Split State

### train.csv — 7,618 clips
| Class | Count | % | Sources |
|---|---|---|---|
| normal | 4,000 | 52.5% | US8K folds 1-8 |
| gunshot | 2,854 | 37.5% | US8K (311) + VOICe_train (2,543) |
| siren | 764 | 10.0% | US8K folds 1-8 |
| explosion | 0 | 0% | FSD50K dev MISSING |
| human_distress | 0 | 0% | FSD50K dev MISSING |
| fire_alarm | 0 | 0% | FSD50K dev MISSING |

### val.csv — 1,669 clips
| Class | Count | Sources |
|---|---|---|
| normal | 703 | US8K fold 9 |
| gunshot | 884 | US8K (31) + VOICe_val (853) |
| siren | 82 | US8K fold 9 |
| explosion | 0 | FSD50K dev MISSING |
| human_distress | 0 | FSD50K dev MISSING |
| fire_alarm | 0 | FSD50K dev MISSING |

### test.csv — 5,776 clips (UNTOUCHED until final evaluation)
| Class | Count | Sources |
|---|---|---|
| normal | 4,169 | US8K fold 10 (837) + FSD50K eval (3,447) |
| gunshot | 930 | US8K (32) + FSD50K eval (128) + VOICe_test (770) |
| siren | 134 | US8K (83) + FSD50K eval (51) |
| explosion | 63 | FSD50K eval |
| human_distress | 271 | FSD50K eval |
| fire_alarm | 209 | FSD50K eval |

---

## FSD50K Dev Partial Download (NEXT BLOCKER)

The FSD50K dev audio is split into 6 zip files on Zenodo (record 4060432).
We do NOT need all 40,966 clips — only ~2,000 that match our filters:
  - explosion (after Fireworks filter): ~720 clips
  - human_distress (Screaming+Shout+Yell): ~441 clips
  - fire_alarm (Alarm, strict filter): ~785 clips
  - siren (Siren): ~77 clips
  - normal (speech/music/crowd, no hazard): ~15,000 but we only need ~2,000

The fname IDs of needed clips are in FSD50K.ground_truth/dev.csv.
Strategy: download each zip part, filter only the needed fnames, delete the rest.
Total needed clips: ~4,000-5,000 = approx 3-5 GB (vs 35 GB full download).

Script to pre-generate the needed fname list:
  python model/generate_fsd50k_download_list.py
  (this script needs to be created — see Phase 7 action items)

---

## Phases Completed

| Phase | Status | Output Files |
|---|---|---|
| Phase 0: Preserve baseline | DONE | branch ml/rebuild-audio-pipeline from cbd94c5 |
| Phase 1: Dataset Audit | DONE | reports/dataset_audit.md |
| Phase 2: Label Inventory | DONE | reports/label_inventory.md |
| Phase 3: User Gate | DONE | Class decisions locked (see approved class table above) |
| Phase 4: Class Mapping | DONE | config/class_mapping.yaml |
| Phase 5: Leakage Audit | DONE | All 6 audits PASS (no leakage found) |
| Phase 6: Split Manifests | DONE | model/data/splits/train.csv, val.csv, test.csv |
| Phase 7+ | TODO | See pending tasks below |

---

## Baseline Metrics (Old Pipeline — Reference Only)

| Metric | Value | Validity |
|---|---|---|
| Test Accuracy | 96.59% | INVALID |
| Macro F1 | 0.9358 | INVALID |
| Architecture | EchoTransformer (CNN+Transformer, 192-bin) | Overparameterized |
| Classes trained | 5 of 8 | explosion/scream/fire_alarm = 0 data |
| Split method | Random 70/15/15 | Leakage confirmed |
| Training samples | 819 | Insufficient |

---

## Environment

```
Python: 3.11 (C:\Program Files\Python311\python.exe)
PyTorch: installed (verify version)
torchaudio: installed
librosa: installed
soundfile: installed
pyyaml: installed
scikit-learn: installed
Branch: ml/rebuild-audio-pipeline
Commit: 97b9ba7
```

Resume commands:
```powershell
cd C:\Users\nikhi\OneDrive\Documents\MINI_PROJECT
git checkout ml/rebuild-audio-pipeline

# Verify splits exist
python -c "import csv; r=list(csv.DictReader(open('model/data/splits/train.csv'))); print(f'Train: {len(r)} rows')"

# Phase 7: run preprocessing pipeline builder
$env:PYTHONIOENCODING="utf-8"; cd model
python build_preprocessing_pipeline.py   # CREATE THIS in Phase 7

# Phase 9: train baseline (3-class only until FSD50K dev downloaded)
python train_v2.py --exp-id EXP_001_CNN_BASELINE_3CLASS --epochs 30
```

---

## Pending Tasks (in order)

### IMMEDIATE BLOCKER
- [ ] Download FSD50K dev audio (targeted subset):
      1. python model/generate_fsd50k_download_list.py  (CREATE)
      2. Download only needed zip parts from Zenodo record 4060432
      3. Re-run: python model/build_manifests.py --voice-extract to rebuild manifests with FSD50K dev

### Phase 7 — Preprocessing Pipeline
- [ ] Create model/preprocessing.py — single canonical pipeline:
      16 kHz, mono, 64-bin log-mel, 2s window, per-clip mean-std normalisation
      Training: random or event-aware 2s crop from longer clips
      Val/Test: overlapping windows (0.5s hop), aggregate by max pooling
- [ ] Verify preprocessing is IDENTICAL between training and inference (two_pass_detector)

### Phase 8 — Augmentation
- [ ] SpecAugment (time masking + frequency masking) — training only
- [ ] Background noise mixing: mix hazard clips with US8K normal clips at SNR 5-20 dB
- [ ] Gain variation: ±6 dB random
- [ ] Time shift: ±100 ms random roll
- [ ] NEVER augment val or test data

### Phase 9 — Baseline Training
- [ ] EXP_001_CNN_BASELINE: small 3-conv CNN, 64-bin log-mel, 3 classes (normal/gunshot/siren)
      Measure: params, size, latency, val F1
- [ ] EXP_002_CRNN_BASELINE: Log-Mel → 3×ConvBlock → GRU(128) → Dense(6)
      Same 3 classes first, then expand to 6 when FSD50K dev available
- [ ] Create experiments/experiment_log.csv with full metadata per experiment
- [ ] Log: arch, git commit, mapping, seeds, hyperparams, augmentation, metrics, runtime, checkpoint path

### Phase 10 — Loss / Sampling
- [ ] Compute final class weights from actual manifest counts
      Recommended weights (class_id order 0-5): [1.0, 2.286, 32.0, 16.0, 9.412, 61.538]
      Adjust after FSD50K dev is ingested
- [ ] CrossEntropyLoss(weight=class_weights_tensor)
- [ ] Consider WeightedRandomSampler for minority classes

### Phase 11 — Hardware
- [ ] Check GPU availability: torch.cuda.is_available(), torch.cuda.get_device_name()
- [ ] If no GPU: prepare Google Colab notebook with identical config
      Upload splits/ manifests + extracted VOICe windows to Colab/Drive

### Phase 12 — Experiment Tracking
- [ ] Create experiments/experiment_log.csv
      Columns: exp_id, date, arch, num_classes, git_commit, class_mapping_version,
               seed, lr, batch_size, epochs, augmentation, train_samples,
               val_f1_macro, val_acc, test_f1_macro, test_acc,
               hazard_false_alarm_rate, hazard_miss_rate_per_class,
               model_size_mb, latency_ms, checkpoint_path, notes

### Phases 13-24 — Evaluation, Calibration, Multi-window, Hard Negatives, Phones...
- [ ] Phase 13: Full per-class metrics (P/R/F1/FP/FN, Macro F1, Weighted F1, Confusion Matrix)
- [ ] Phase 14: Threshold calibration on VAL only (precision-recall analysis)
- [ ] Phase 15: UNCERTAIN state (NORMAL / UNCERTAIN / HAZARD CANDIDATE)
- [ ] Phase 16: Multi-window overlapping inference (0.5s hop, temporal aggregation)
- [ ] Phase 17: Hard negative test suite (TV, appliances, speech, music, weather)
- [ ] Phase 18: Smartphone domain test (1m/3m/5m, quiet/noisy/outdoor)
- [ ] Phase 19: Media playback false positive test (action movies, YouTube)
- [ ] Phase 20: Failure analysis loop (val confusion → root cause → one change → new exp ID)
- [ ] Phase 21: Model selection, freeze, single final test set run
- [ ] Phase 22: Regression test suite
- [ ] Phase 23: Git hygiene, licenses, attribution
- [ ] Phase 24: Success criteria verification

---

## Key Decisions Made (LOCKED)

1. SCREAM + SHOUTING merged → human_distress (WARNING)
2. GUNSHOT and EXPLOSION are separate classes
3. GLASS_BREAKING: EXCLUDED this version
4. FIREWORKS: EXCLUDED (acoustic confusion with gunshot)
5. BABYCRY: EXCLUDED this version
6. Feature: plain 64-bin log-mel (NOT 192-bin Sobel/Laplacian)
7. Architecture: start with CNN baseline, then CRNN — NO Transformer until justified
8. Split: US8K fold-based (folds 1-8/9/10); FSD50K dev/eval; VOICe file-split
9. Normal cap: 4,000 training samples (adjust after FSD50K dev ingested)
10. Class weights (inv-freq, estimated): [1.0, 2.29, 32.0, 16.0, 9.41, 61.54]

---

## Files Created This Session

| File | Purpose |
|---|---|
| config/class_mapping.yaml | Authoritative class mapping with filter rules and split config |
| reports/dataset_audit.md | Phase 1 full dataset audit |
| reports/label_inventory.md | Phase 2 full label inventory with recommended mappings |
| reports/split_summary.txt | Phase 6 manifest summary |
| model/audit_leakage.py | Phase 5 leakage/quality audit script |
| model/build_manifests.py | Phase 6 manifest builder (fold-aware, VOICe extractor) |
| model/data/splits/train.csv | Training manifest (7,618 rows — 3 classes) |
| model/data/splits/val.csv | Validation manifest (1,669 rows — 3 classes) |
| model/data/splits/test.csv | TEST manifest (5,776 rows — 6 classes, UNTOUCHED) |
| model/data/voice_windows/ | Extracted VOICe gunshot windows (2,543 train / 853 val / 770 test) |

---

## Immediate Next Action for Next Agent

STEP 1 (10 min): Generate FSD50K download list
  Create model/generate_fsd50k_download_list.py that:
  - Reads model/data/raw/FSD50K/FSD50K.ground_truth/dev.csv
  - Applies fsd50k_label_to_echo() filter (same logic as build_manifests.py)
  - Outputs list of needed fnames to reports/fsd50k_needed_fnames.txt
  - Estimates total file count and approximate size

STEP 2 (ask user): Download FSD50K dev audio
  The FSD50K dev audio is on Zenodo record 4060432.
  There are 6 zip parts (FSD50K.dev_audio.zip parts 1-6).
  Ask user to download and place in: model/data/raw/FSD50K/FSD50K.dev_audio/
  OR: write a downloader script that downloads and filters on the fly.

STEP 3: Re-run build_manifests.py after FSD50K dev audio is available
  python model/build_manifests.py --voice-extract --normal-cap 4000

STEP 4: Build Phase 7 preprocessing pipeline
  Create model/preprocessing.py with canonical 64-bin log-mel pipeline.
  Ensure identical preprocessing between training and inference.

STEP 5: Create experiment tracking
  Create experiments/experiment_log.csv with correct column headers.

STEP 6: Train EXP_001_CNN_BASELINE
  3-class model first (normal/gunshot/siren) — these are the only classes
  with training data right now. Use this to validate the preprocessing
  pipeline and training framework before adding more classes.
