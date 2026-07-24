# ECHO — Decisions Log (Master Context Pack, Part 3 of 5)

> Append-only. Never delete an entry, even if reversed — add a new entry that supersedes it.
> Every AI-assisted session that changes an architectural/scope decision MUST result in a new
> entry here, written by the human who approved the change, same day. If it's not written down
> here, treat it as not decided — revert to what ARCHITECTURE.md/PROJECT_BRIEF.md already say.

Format per entry:
```
### #N — [short title]
Date:
Decided by:
What: (one line)
Why: (one line)
Affects: (which file(s) also needed updating)
```

---

### #1 — Session-based monitoring instead of 24/7 background
Date: [fill in]
Decided by: team
What: Monitoring runs while app is foregrounded/active service, not persistent 24/7 OS-level.
Why: Android 13+ background mic restrictions + Doze mode make true 24/7 unreliable to build
in timeframe; output/UX unaffected (same ON/OFF toggle).
Affects: PROJECT_BRIEF.md, ARCHITECTURE.md

### #2 — Single CRNN, two-pass, instead of two separate model architectures
Date: [fill in]
Decided by: team
What: Replaced AST/PANNs/CLAP "verification model" with the same CRNN run twice at different
window sizes/thresholds.
Why: Heavy transformer models need cloud inference, which conflicts with the no-continuous-
audio-upload privacy rule, and are impractical to train from scratch in timeframe.
Affects: ARCHITECTURE.md

### #3 — Keyword-spotter instead of full ASR
Date: [fill in]
Decided by: team
What: Replaced Whisper/general ASR with a small grammar-constrained keyword spotter for ~6
fixed phrases.
Why: Full ASR is a separate heavy subsystem for a "supporting evidence only" signal; not
worth the engineering cost at this scope. Tier 2 — cut first if behind schedule.
Affects: ARCHITECTURE.md, PROJECT_BRIEF.md

### #4 — Nearby-device alerts fully simulated
Date: [fill in]
Decided by: team
What: No real device-to-device networking; Demo Mode calls a mocked backend endpoint with
scripted "corroboration" data.
Why: Real implementation needs user density + background location infra not available at
prototype scale; explicitly Tier 3, disclosed in report.
Affects: ARCHITECTURE.md, TIER_TABLE.md

### #5 — Post-training quantization + OpenVINO instead of pruning/QAT
Date: [fill in]
Decided by: team
What: Use TFLite post-training dynamic-range quantization for mobile export; OpenVINO IR
export for laptop-side (Intel Arc iGPU) latency/size benchmarking.
Why: Pruning/QAT requires specialized ML-systems expertise beyond scope/timeline; OpenVINO
benchmarking on team's actual Intel hardware is a stronger, more specific interview story
than generic TFLite-only claims.
Affects: ARCHITECTURE.md

---

<!-- Add new entries below this line as the project progresses. -->
