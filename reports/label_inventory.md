# ECHO — Label Inventory
**Phase:** Phase 2 — Label Inventory
**Date:** 2026-07-27
**Purpose:** Structured mapping of all potentially relevant labels from all three datasets to candidate Echo classes. This is the decision-gate document. No mapping is finalized here — user approval required in Phase 3.

Legend — Confidence:
  HIGH = strong acoustic/semantic match, sufficient samples
  MEDIUM = reasonable match but some ambiguity or limited samples
  LOW = weak proxy or very few samples
  RISK = known confusion / leakage risk

---

## UrbanSound8K Labels

| Original Label | Dataset | Count | Possible Echo Meaning | Recommended Mapping | Confidence | Notes |
|---|---|---|---|---|---|---|
| gun_shot | US8K | 374 | Gunshot/Firearms | GUNSHOT | HIGH | Primary gunshot source. Field recordings. |
| siren | US8K | 929 | Emergency vehicle / alarm | SIREN | HIGH | Best siren source. Includes car alarms — needs filtering decision. |
| dog_bark | US8K | 1000 | Non-hazard, impulsive | NORMAL | HIGH | Critical hard negative. Loud, impulsive. |
| children_playing | US8K | 1000 | Non-hazard background | NORMAL | HIGH | Good background normal source. |
| air_conditioner | US8K | 1000 | Non-hazard background | NORMAL | HIGH | Continuous noise — good negative. |
| street_music | US8K | 1000 | Non-hazard background | NORMAL | HIGH | Good hard negative (contains rhythmic patterns). |
| engine_idling | US8K | 1000 | Non-hazard background | NORMAL | HIGH | Good continuous background. |
| jackhammer | US8K | 1000 | Non-hazard impulsive | NORMAL | HIGH | CRITICAL hard negative — rhythmic impulses near gunshot profile. |
| drilling | US8K | 1000 | Non-hazard impulsive | NORMAL | HIGH | Similar to jackhammer — hard negative. |
| car_horn | US8K | 429 | Borderline — alert sound but not emergency | NORMAL | MEDIUM | NOT an emergency signal. Exclude from SIREN class. Map to NORMAL. |

---

## FSD50K Labels — Hazard Candidates

| Original Label | Dataset | Dev Count | Possible Echo Meaning | Recommended Mapping | Confidence | Notes |
|---|---|---|---|---|---|---|
| Gunshot_and_gunfire | FSD50K | 348 | Gunshot | GUNSHOT | HIGH | Primary FSD50K gunshot source. Multi-label — filter carefully. |
| Explosion | FSD50K | 1122 | Explosion / blast | EXPLOSION | MEDIUM | Heavy co-occurrence with Fireworks. Mostly outdoor/distant. |
| Screaming | FSD50K | 254 | Human distress scream | SCREAM | HIGH | Best scream source available. |
| Shout | FSD50K | 216 | Human distress shout | SCREAM or SHOUTING | MEDIUM | Weaker than Screaming. May merge with SCREAM class. |
| Yell | FSD50K | 139 | Human distress yell | SCREAM or SHOUTING | MEDIUM | Similar to Shout. Merge with Screaming group. |
| Siren | FSD50K | 77 | Emergency siren | SIREN | HIGH | Low count; supplement with US8K siren (929 clips). |
| Shatter | FSD50K | 414 | Glass breaking | GLASS_BREAKING | HIGH | More specific than Glass label. Primary glass source. |
| Glass | FSD50K | 974 | Glass (broad) | GLASS_BREAKING | MEDIUM | Includes glass sound effects, clinking, not just breaking. Filter by co-occurrence with Shatter. |
| Alarm | FSD50K | 1280 | Various alarms | FIRE_ALARM (partial) | LOW | VERY heterogeneous. Includes car alarms, phone ringtones, doorbells. Requires per-clip filtering. Use only clips with no co-labels outside alarm family. |
| Fire | FSD50K | 385 | Fire sound | FIRE_ALARM (contextual) | LOW | Crackling fire ≠ fire alarm. Co-occurring with Alarm is needed for fire alarm context. |
| Fireworks | FSD50K | 402 | Fireworks (acoustic similar to gunshots) | EXCLUDE or NORMAL | RISK | Acoustically identical to distant gunshots. Training both as separate classes will cause severe confusion. |
| Boom | FSD50K | 167 | Low-freq impact | EXPLOSION (partial) | LOW | Too ambiguous. Co-occurs with thunder, bass, fireworks. |
| Screech | FSD50K | 86 | High-frequency screech | EXCLUDE | LOW | Heterogeneous: tyre squeal, animal, glass. Too ambiguous. |
| Crying_and_sobbing | FSD50K | 109 | Human distress (crying) | TBD | LOW | Not classic hazard. Possible DISTRESS class. Await user decision. |
| Boom | FSD50K | 167 | Low-frequency impact | EXCLUDE | LOW | Too ambiguous — thunder, fireworks, bass all tagged Boom. |

---

## FSD50K Labels — NORMAL Candidates (hard negatives)

| Original Label | Dataset | Dev Count | Recommended Mapping | Confidence | Notes |
|---|---|---|---|---|---|
| Speech | FSD50K | ~785 eval | NORMAL | HIGH | General speech — excellent hard negative. |
| Male_speech_and_man_speaking | FSD50K | 467 eval | NORMAL | HIGH | Specific speech hard negative. |
| Female_speech_and_woman_speaking | FSD50K | ~300 dev | NORMAL | HIGH | |
| Conversation | FSD50K | ~300 dev | NORMAL | HIGH | |
| Laughter | FSD50K | ~300 dev | NORMAL | HIGH | High-energy vocal — important hard negative. |
| Music | FSD50K | 1972 eval | NORMAL | HIGH | Contains drums, percussion — important hard negative. |
| Crowd | FSD50K | ~500 dev | NORMAL | HIGH | Contains cheering, shouting — critical hard negative. |
| Traffic_noise_and_roadway_noise | FSD50K | ~400 dev | NORMAL | HIGH | Urban noise. |
| Dog | FSD50K | ~500 dev | NORMAL | HIGH | Bark — hard negative. |
| Rain | FSD50K | ~300 dev | NORMAL | HIGH | Continuous noise. |
| Typing | FSD50K | ~300 dev | NORMAL | MEDIUM | Hard negative for impulsive sounds. |
| Walk_and_footsteps | FSD50K | ~400 dev | NORMAL | MEDIUM | Impulsive — relevant hard negative. |
| Applause | FSD50K | ~300 dev | NORMAL | MEDIUM | Grouped impulsive sounds. |
| Fireworks | FSD50K | 402 dev | NORMAL or EXCLUDE | RISK | If GUNSHOT class exists, Fireworks must be NORMAL to prevent confusion. |

---

## VOICe Labels

| Original Label | Dataset | Events | Files | Possible Echo Meaning | Recommended Mapping | Confidence | Notes |
|---|---|---|---|---|---|---|---|
| gunshot | VOICe | 4235 | 207 | Gunshot | GUNSHOT | HIGH | Very high event count. Synthetic mixture. Requires windowed extraction. |
| glassbreak | VOICe | 4444 | 207 | Glass breaking | GLASS_BREAKING | HIGH | Very high event count. Primary glass-breaking source. |
| babycry | VOICe | 3490 | 207 | Infant distress | TBD — await Phase 3 gate | LOW-MEDIUM | NOT a standard hazard. Possible DISTRESS sub-class. Acoustic profile ≠ adult scream. |

---

## Synthetic Data (existing, model/data/synthetic/)

| Class | Generated Samples | Quality | Recommended Action |
|---|---|---|---|
| gunshot | 50 | Poor — mathematical decay of white noise | EXCLUDE from rebuild training |
| explosion | 50 | Poor — sine + noise | EXCLUDE |
| scream | 50 | Very poor — modulated tone, not speech-like | EXCLUDE |
| glass_breaking | 50 | Poor — decaying sine impulses | EXCLUDE |
| fire_alarm | 50 | Passable — beeping pure tone at 3 kHz | May use for augmentation only |
| siren | 50 | Passable — FM sweep | May use for augmentation only |
| shouting | 50 | Very poor — modulated vocal simulation | EXCLUDE |
| normal | 50 | Poor — white noise only | EXCLUDE |

**Decision: Do NOT use synthetic data for training in the rebuild. Real data only.**

---

## Proposed Class Shortlist (for Phase 3 user decision)

| Candidate Class | Primary Source(s) | Est. Usable Samples | Strength |
|---|---|---|---|
| GUNSHOT | US8K gun_shot (374) + FSD50K Gunshot_and_gunfire (348) + VOICe gunshot (~3500) | ~4,200 | STRONG |
| SIREN | US8K siren (929) + FSD50K Siren (77) | ~1,000 | GOOD |
| GLASS_BREAKING | FSD50K Shatter (414) + VOICe glassbreak (~3800) | ~4,200 | STRONG |
| SCREAM | FSD50K Screaming (254) + Shout (216) + Yell (139) | ~600 | MODERATE |
| EXPLOSION | FSD50K Explosion (~400 filtered) | ~300-400 | WEAK (after Fireworks exclusion) |
| FIRE_ALARM | FSD50K Alarm (~200 filtered smoke/fire only) | ~150-200 | WEAK |
| SHOUTING | FSD50K Shout (216) + Yell (139) — or merge with SCREAM | ~350 | WEAK standalone |
| BABYCRY (new?) | VOICe babycry (~2800) | ~2,800 | MODERATE — mapping unclear |
| NORMAL | US8K 8 classes + FSD50K Speech/Music/Crowd + background extraction | ~3,000+ | STRONG |

---

## Summary of Class Viability

STRONGLY SUPPORTED (>1000 reliable samples, clear acoustic identity):
  - GUNSHOT, GLASS_BREAKING, SIREN, NORMAL

MODERATELY SUPPORTED (400-1000 samples, some ambiguity):
  - SCREAM / SHOUTING (may merge)

WEAKLY SUPPORTED (<400 reliable samples, high heterogeneity):
  - EXPLOSION (Fireworks problem)
  - FIRE_ALARM (Alarm heterogeneity)

INSUFFICIENT DATA or AMBIGUOUS:
  - BABYCRY (no clear Echo class mapping — user must decide)
  - SCREECH (excluded — too heterogeneous)
  - BOOM (excluded — too ambiguous)
