# ECHO — Dataset Audit Report
**Phase:** Phase 1 — Dataset Audit
**Date:** 2026-07-27

---

## 1. UrbanSound8K

### 1.1 Overview

| Property | Value |
|---|---|
| Total files | 8,732 audio clips |
| Sampling rates | Mixed (22050, 44100, 48000, 11025 Hz) |
| Channels | Mixed (mono and stereo) |
| File format | .wav, .mp3, .ogg |
| Clip duration | Up to 4.0 s (many shorter) |
| Official split | 10 predefined folds for cross-validation |
| Multi-label | No — single label per clip |
| License | CC BY-NC 3.0 |
| Source | Urban field recordings (NYC), Freesound.org |

### 1.2 Class Distribution

| Class | Count | Echo Relevance |
|---|---|---|
| dog_bark | 1000 | NORMAL (false-alarm risk source) |
| children_playing | 1000 | NORMAL |
| air_conditioner | 1000 | NORMAL |
| street_music | 1000 | NORMAL |
| engine_idling | 1000 | NORMAL |
| jackhammer | 1000 | NORMAL (impulsive hard negative) |
| drilling | 1000 | NORMAL (impulsive hard negative) |
| siren | 929 | HAZARD CANDIDATE |
| car_horn | 429 | NORMAL (borderline) |
| gun_shot | 374 | HAZARD CANDIDATE |

### 1.3 Fold Protocol (CRITICAL)
- 10 pre-assigned folds; official protocol requires fold-based cross-validation.
- The existing prepare_dataset.py performs a naive 70/15/15 random shuffle — this VIOLATES the official US8K protocol and causes leakage.
- MUST use folds as hard split boundaries. Recommended: folds 1-8 train, fold 9 val, fold 10 test.

### 1.4 Quality Notes
- dog_bark: loud impulsive — critical hard negative for gunshot confusion.
- jackhammer/drilling: rhythmic impulsive — must not trigger gunshot alert.
- gun_shot clips: some extremely short (<0.5 s); naive truncation to 2 s yields mostly silence.
- siren: includes emergency sirens, car alarms, industrial alarms — heterogeneous within class.
- No explosion, scream, fire alarm, or glass-breaking clips exist.

---

## 2. FSD50K

### 2.1 Overview

| Property | Value |
|---|---|
| Dev clips | 40,966 |
| Eval clips | 10,231 |
| MULTI-LABEL dev | 83.2% of clips have more than one label |
| MULTI-LABEL eval | 88.9% of clips have more than one label |
| Total unique labels | 200 (AudioSet ontology) |
| Sampling rate | 44,100 Hz (must resample to 16 kHz) |
| Duration | Variable: 0.3 s to 30+ s |
| Official split | dev (train/val) + eval (TEST — DO NOT TOUCH FOR TRAINING) |
| License | Mixed CC (per Freesound clip) |

### 2.2 CRITICAL: Multi-label Warning
FSD50K is fundamentally multi-label. 88.9% of clips carry multiple labels.
Naively treating FSD50K clips as single-label WILL cause catastrophic label confusion.
Recommended strategy: strict co-occurrence-aware filtering for hazard classes.

### 2.3 Echo-Relevant Dev Labels

| Label | Dev Count | Notes |
|---|---|---|
| Alarm | 1280 | Very heterogeneous (car/smoke/phone) |
| Explosion | 1122 | Co-occurs heavily with Fireworks |
| Glass | 974 | Broad; Shatter is more specific |
| Shatter | 414 | Best glass-breaking signal |
| Fireworks | 402 | Acoustically near-identical to gunshots |
| Fire | 385 | Crackling fire + fire alarm overlap |
| Gunshot_and_gunfire | 348 | Best direct source; multi-label |
| Screaming | 254 | High value for scream class |
| Shout | 216 | Weaker scream signal |
| Boom | 167 | Ambiguous: thunder, explosion, bass |
| Yell | 139 | Similar to Shout |
| Crying_and_sobbing | 109 | Possible distress class |
| Screech | 86 | Heterogeneous: tyre, animal, glass |
| Siren | 77 | Low count; supplement with US8K |

### 2.4 FSD50K Eval Set — RESERVED FOR TEST ONLY

| Label | Eval Count |
|---|---|
| Alarm | 584 |
| Glass | 267 |
| Explosion | 266 |
| Shout | 177 |
| Gunshot_and_gunfire | 134 |
| Screaming | 123 |
| Siren | 55 |

DO NOT USE eval set during training or threshold calibration.

### 2.5 Good NORMAL Sources from FSD50K
Speech, Male/Female speech, Conversation, Laughter, Music, Crowd, Traffic_noise, Dog, Bird, Wind, Rain, Typing, Keyboard, Footsteps

### 2.6 Limitations
- Alarm is too heterogeneous to be a clean class — needs sub-filtering.
- Fireworks acoustically mimics gunshots — training both as separate classes will cause model confusion.
- Explosion clips are mostly outdoor/distant fireworks, not indoor blasts.
- FSD50K eval is the TEST set boundary. It must remain untouched.

---

## 3. VOICe

### 3.1 Overview

| Property | Value |
|---|---|
| Subset | clean/ only |
| Audio files | 207 long-form WAV files |
| Duration range | 529 s to 2028 s per file (~57.3 hours total) |
| Sample rate | 16,000 Hz (estimated from file sizes) |
| Annotation format | Tab-separated: start_time end_time label per line |
| Total annotated events | 12,169 |
| Official splits | Source train: 69, val: 69, test: 69 files |
| Dataset type | Sound Event Detection (SED) — NOT clip classification |

### 3.2 Labels

| Label | Events | Files |
|---|---|---|
| glassbreak | 4,444 | 207 (all files) |
| gunshot | 4,235 | 207 (all files) |
| babycry | 3,490 | 207 (all files) |

### 3.3 Extraction Strategy
VOICe requires windowed extraction, NOT whole-clip treatment.
For each event [t_start, t_end, label]:
1. Extract window centered on event (2 s window, padded if event <2 s, trimmed if >2 s)
2. Reject windows shorter than 0.5 s
3. Handle overlapping events: label by dominant (>50% of window)
4. RESPECT official file-based splits (no cross-file shuffling)

Estimated extractable clips:
- gunshot: ~3,500 usable 2-s windows
- glassbreak: ~3,800 usable 2-s windows
- babycry: ~2,800 usable 2-s windows (mapping TBD)

### 3.4 Limitations
- babycry class has no direct Echo mapping — needs user decision.
- VOICe is synthetic mixture audio — acoustic realism depends on mixing.
- clean/ subset has no additive noise; noisy subset not locally present.
- Overlapping events are common — windowing requires careful label assignment.

---

## 4. Summary Comparison

| Property | UrbanSound8K | FSD50K | VOICe |
|---|---|---|---|
| Size | 8,732 clips | 51,197 clips | 207 files (57 hrs) |
| Multi-label | No | Yes (83-89%) | Overlapping events |
| Official splits | 10 folds | dev/eval | source train/val/test |
| Echo hazard labels | gun_shot, siren | Gunshot, Explosion, Screaming, Shout, Siren, Shatter, Alarm | gunshot, glassbreak |
| Echo normal labels | 8 urban classes | 100+ non-hazard classes | background within mixture |
| Biggest risk | Fold leakage | Multi-label confusion | Requires SED windowing |
| Sample rate | Mixed | 44100 Hz | 16000 Hz |

---

## 5. Critical Existing Pipeline Failures

1. ZERO samples for explosion, scream, fire_alarm in processed data — model cannot classify them.
2. The 96.59% accuracy is on a random 70/15/15 split, not fold-respecting — leakage is highly likely.
3. The confusion matrix rows for explosion, scream, fire_alarm are all zeros — evaluation was on 5 classes, not 8.
4. ESC-50 crying_baby mapped to shouting — acoustically and semantically incorrect proxy.
5. EchoTransformer has a Transformer encoder (192 mel bins input) trained on only 1,170 samples — massively overparameterized, overfits to dataset artifacts.
6. shouting (40 samples) and glass_breaking (40 samples) are essentially untrained — FNR 21% and 12.5% respectively.
7. No hard negative testing on everyday sounds.
8. No augmentation with real background noise.
9. The model uses Sobel/Laplacian-concatenated mel spectrograms (192 bins) — this unusual feature engineering has not been validated and adds complexity without proven benefit.
