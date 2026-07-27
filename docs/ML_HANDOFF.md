# ECHO ML Handoff Document
**Single Source of Truth for ML Pipeline State**
**Branch:** ml/rebuild-audio-pipeline
**Last Updated:** 2026-07-27T14:20 IST (Phase 4 complete)

---

## Current Phase: PHASE 5 — Data Quality and Leakage Analysis (NEXT)

**Status:** Phase 4 complete. config/class_mapping.yaml written and locked.
Phase 5 (leakage analysis and quality checks) is the immediate next action.
DO NOT TRAIN YET.

---

## Phase 3 Gate — User Decisions (LOCKED)

| Class | Severity | Decision |
|---|---|---|
| GUNSHOT | DANGER | INCLUDE |
| EXPLOSION | DANGER | INCLUDE (separate from Gunshot) |
| HUMAN_DISTRESS (SCREAM + SHOUTING merged) | WARNING | INCLUDE |
| SIREN | WARNING | INCLUDE |
| FIRE_ALARM | DANGER | INCLUDE |
| GLASS_BREAKING | — | EXCLUDED by user |
| FIREWORKS | — | EXCLUDED (acoustic confusion with gunshot) |
| BABYCRY | — | EXCLUDED by user |
| NORMAL | — | INCLUDE (hard negatives) |

**Merge decisions:**
- SCREAM + SHOUTING → single class: `human_distress`
- GUNSHOT and EXPLOSION remain SEPARATE classes
- Final model has 6 output classes: normal, gunshot, explosion, human_distress, siren, fire_alarm

---

## Approved Class Mapping Summary

| class_id | Name | Severity | Primary Sources | Est. Samples |
|---|---|---|---|---|
| 0 | normal | NORMAL | US8K (8 bg classes), FSD50K (Speech/Music/Crowd/etc.) | ~8,000-11,000 |
| 1 | gunshot | DANGER | US8K gun_shot (374) + FSD50K Gunshot_and_gunfire (348) + VOICe gunshot (~3,200) | ~3,922 |
| 2 | explosion | DANGER | FSD50K Explosion (filtered, no Fireworks) | ~250-300 |
| 3 | human_distress | WARNING | FSD50K Screaming (254) + Shout (216) + Yell (139) | ~609 |
| 4 | siren | WARNING | US8K siren (929) + FSD50K Siren (77) | ~1,006 |
| 5 | fire_alarm | DANGER | FSD50K Alarm (strict-filtered, smoke/fire only) | ~120-180 |

**Known imbalance:** explosion (~250) and fire_alarm (~150) are severely under-represented.
Mitigation: class-weighted cross-entropy loss (inverse frequency weighting).

---

## Summary of Completed Work

| Phase | Status | Output |
|---|---|---|
| Phase 0: Baseline Preservation | DONE | Branch ml/rebuild-audio-pipeline created |
| Phase 1: Dataset Audit | DONE | reports/dataset_audit.md |
| Phase 2: Label Inventory | DONE | reports/label_inventory.md |
| Phase 3: User Gate | DONE | Class decisions received and locked |
| Phase 4: Class Mapping | DONE | config/class_mapping.yaml |
| Phase 5: Leakage Analysis | NEXT | — |
| Phase 6: Data Splits | TODO | — |
| Phase 7: Preprocessing | TODO | — |
| Phase 8: Augmentation | TODO | — |
| Phase 9: Baselines | TODO | — |
| Phases 10-24 | TODO | — |

---

## Baseline Metrics (Existing Pipeline — For Comparison Only)

| Metric | Value | Validity |
|---|---|---|
| Test Accuracy | 96.59% | INVALID (random split, fold leakage) |
| Macro F1 | 0.9358 | INVALID (3 of 8 classes had zero test samples) |
| Shouting FNR | 21.4% | Only 40 samples with wrong proxy label |
| Glass Breaking FNR | 12.5% | Only 40 samples |
| Classes actually trained | 5 of 8 | explosion/scream/fire_alarm = 0 samples |
| Training set size | 819 samples | Far too small for Transformer |
| Architecture | EchoTransformer (CNN+Transformer, 192 mel bins) | Overparameterized |

---

## Identified Failure Modes

| ID | Failure | Status |
|---|---|---|
| F1 | Zero data for 3 classes | Addressed: will ingest from FSD50K/VOICe in Phase 6 |
| F2 | Test/train leakage (random split) | Addressed: fold-aware splits in Phase 6 |
| F3 | Inflated accuracy | Addressed: rebuild test set in Phase 6 |
| F4 | Wrong proxy label (babycry→shouting) | Addressed: babycry excluded; human_distress from real data |
| F5 | Overparameterized model | Addressed: start with CNN baseline in Phase 9 |
| F6 | No hard negative testing | Pending: Phase 17 |
| F7 | No noise augmentation | Pending: Phase 8 |
| F8 | Multi-label FSD50K naive treatment | Addressed: filter rules in class_mapping.yaml |
| F9 | 192-bin Sobel/Laplacian unvalidated | Addressed: start with plain 64-bin mel |

---

## Environment Dependencies

```
Python: 3.10+
PyTorch: 2.x
torchaudio: 2.x
librosa: 0.10+
soundfile: 0.12+
scikit-learn: 1.x
pandas: 2.x
pyyaml: 6.x
numpy: 1.24+
```

Commands to resume work from Phase 5:
```powershell
cd C:\Users\nikhi\OneDrive\Documents\MINI_PROJECT
git checkout ml/rebuild-audio-pipeline

# Phase 5-6: leakage + splits
python model/audit_leakage.py         # (to be created in Phase 5)
python model/build_manifests.py       # (to be created in Phase 6)

# Phase 9: baseline training
python model/train_v2.py --exp-id EXP_001_CNN_BASELINE
```

---

## Pending Tasks

- [ ] PHASE 5: Audit for train/test leakage, duplicates, imbalance, silence-heavy clips
- [ ] PHASE 6: Build fold-aware TRAIN/VAL/TEST manifest CSVs
- [ ] PHASE 7: Canonical preprocessing pipeline (16kHz, mono, 64-bin log-mel, 2s window)
- [ ] PHASE 8: Training-only augmentations (noise mixing, SpecAugment, gain, reverb)
- [ ] PHASE 9: Train CNN baseline (EXP_001), then CRNN (EXP_002)
- [ ] PHASE 10: Class-weighted loss (explosion and fire_alarm badly imbalanced)
- [ ] PHASE 11: Benchmark local GPU/CPU; prepare Colab notebook if needed
- [ ] PHASE 12: experiments/experiment_log.csv
- [ ] PHASES 13-24: Full evaluation, calibration, multi-window, hard negatives, smartphones

---

## Immediate Next Action

Start Phase 5: Audit for leakage, duplicates, and quality issues.
Create model/audit_leakage.py that:
1. Cross-checks US8K clips across the proposed fold split (folds 1-8 vs 9 vs 10)
2. Cross-checks FSD50K dev vs eval by Freesound clip ID
3. Cross-checks VOICe files against official split lists
4. Reports class imbalance ratios and recommended class weights
5. Identifies clips shorter than 0.5 s or that are effectively silence
6. Checks for duplicate filenames across datasets

Then Phase 6: Build manifest CSVs (train.csv, val.csv, test.csv) with fold-aware splits.
