# ECHO Model Training & Validation Run Trace

*Trace started at: 2026-07-24 06:20:26 UTC*


## 🏁 PHASE: INITIAL SETUP

### 📋 Task: Ingesting and Cleaning Dataset
Executing prepare_dataset.py to map raw files to the 5 active classes...
* **[SUCCESS]** Successfully mapped and prepared datasets.

## 🏁 PHASE: VALIDATION LOOP ITERATION 1 (LEARNING RATE = 0.001)

### 📋 Task: Model Training (Iteration 1)
Training CRNN model for 15 epochs with LR=0.001...
* **[SUCCESS]** Training completed in 83.9s.

### 📋 Task: Model Evaluation (Iteration 1)
Running evaluate.py on the test split...
* **[EVALUATION_METRICS]** Test Accuracy: 93.75%, Test F1 Score: 0.9195
* **[QUALITY_GATE_PASSED]** Quality gate F1 score >= 0.9 achieved!

---

## 📈 Final Summary Report
* **Duration**: 112.54 seconds
* **Start Time**: 2026-07-24T06:20:26.588376Z
* **End Time**: 2026-07-24T06:22:19.127908Z

### Metrics Details:
```
==================================================
AUTOMATED DEVELOPMENT LOOP METRICS REPORT
==================================================
Iteration 1:
  - Learning Rate   : 0.00100
  - Train Loss      : 0.0384
  - Val Loss        : 0.1025
  - Val Accuracy    : 96.57%
  - Test F1 Score   : 0.9195
  - Test Accuracy   : 93.75%
  - Elapsed Time    : 112.5s
  ----------------------------------------
*** BEST PERFORMANCE MODEL (Iteration 1) ***
  - Test F1 Score   : 0.9195
  - Test Accuracy   : 93.75%
  - Final Val Acc   : 96.57%
  - Final Val Loss  : 0.1025
Total duration: 112.54 seconds
==================================================

```

## 🏁 PHASE: GIT INTEGRATION

### 📋 Task: Committing Model
Staging checkpoints and reports, and creating git commit...
* **[GIT_COMMIT_SUCCESS]** Successfully committed model checkpoint: [master 010f33f] Automated Dev Loop Success: Test F1=0.9195, Accuracy=93.75% (Iteration 1)
 1 file changed, 57 insertions(+)
 create mode 100644 reports/training_run_trace.md
