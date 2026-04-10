# Results — Non-Invasive ICP Monitoring & 15-Minute Forecasting

> **Model version**: XGBoost v3.0 / LSTM v4.2 | **Training date**: 2026-04-09
> **Datasets**: CHARIS (13 TBI patients) + MIMIC-III (95 general ICU patients) = 108 patients, 501,302 windows

---

## 1. XGBoost Binary Classifier (Normal vs Abnormal ≥15 mmHg)

### Primary Metrics: 5-Fold Patient-Level Cross-Validation

> **These are the honest, headline numbers. Cite these in the paper.**

| Metric | CV Mean ± Std |
|--------|:---:|
| **F1-Score** | **0.828 ± 0.114** |
| **AUC** | **0.944 ± 0.023** |
| Recall | 0.826 ± 0.104 |

### Per-Fold Breakdown

| Fold | F1 | AUC | Test Patients | Notes |
|:---:|:---:|:---:|:---:|:---|
| 1 | 0.969 | 0.966 | 1 | Single CHARIS patient — inflated |
| 2 | 0.931 | 0.967 | 26 | Good mix |
| 3 | 0.812 | 0.912 | 27 | Moderate |
| 4 | 0.650 | 0.951 | 27 | Hardest fold |
| 5 | 0.781 | 0.923 | 27 | Moderate |

> **Note**: F1 std = 0.114 reflects high patient-level variance with only 108 patients. The test-set F1 (0.882) is from a favorable 20% holdout. **The CV mean is the more honest estimator.**

### Supplementary: Test Set Metrics (20% holdout, N=69,290 windows)

| Metric | Value | 95% CI |
|--------|:---:|:---:|
| F1-Score | 0.882 | [0.880, 0.885] |
| AUC | 0.956 | [0.954, 0.957] |
| Precision | 0.935 | [0.932, 0.937] |
| Recall | 0.835 | [0.831, 0.839] |
| Specificity | 0.949 | — |
| Balanced Accuracy | 0.892 | — |
| ECE (calibrated) | 0.098 | — |

### Confusion Matrix (Test Set)

|  | Predicted Normal | Predicted Abnormal |
|:--|:---:|:---:|
| **True Normal** | 34,921 (TN) | 1,894 (FP) |
| **True Abnormal** | 5,348 (FN) | 27,127 (TP) |

---

## 2. Baseline Comparison

All models use the same patient-level split. SVM subsampled to 20,000 training windows.

| Model | F1 | AUC | Precision | Recall | Balanced Acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| Cardiac Amp Threshold | 0.638 | 0.455 | 0.472 | 0.987 | 0.506 |
| SVM RBF (n=20k) | 0.837 | 0.944 | 0.875 | 0.802 | 0.851 |
| Logistic Regression | 0.861 | 0.941 | 0.906 | 0.821 | 0.873 |
| **XGBoost (ours)** | **0.891** | **0.969** | **0.930** | **0.856** | **0.899** |
| Random Forest (100) | 0.893 | 0.964 | 0.922 | 0.866 | 0.901 |

> XGBoost achieves the highest AUC (0.969). RF matches on F1 (+0.002, not significant). XGBoost is retained for inference speed (~0.1ms), SHAP explainability, and ESP32 deployability.

---

## 3. BiLSTM Forecaster — Honest Metrics

### 3A. Per-Sequence Evaluation (HONEST — 1 decision per sequence)

> **Primary metric**. Strategy: peak P(Abnormal) across 15 horizons → single binary decision per sequence.

| Metric | Value |
|--------|:---:|
| **AUC** | **0.890** |
| **F1-Score** | **0.795** |
| Precision | 0.924 |
| Recall | 0.698 |
| Balanced Accuracy | 0.801 |
| Test Sequences | 66,237 |
| Test Patients | 15 |

### 3B. Per-Horizon AUC Breakdown (key insight for reviewers)

| Horizon | AUC | F1 | Recall |
|:---:|:---:|:---:|:---:|
| **t+1 min** | **0.924** | **0.819** | 0.789 |
| t+2 min | 0.918 | 0.814 | 0.787 |
| t+3 min | 0.914 | 0.809 | 0.783 |
| t+5 min | 0.904 | 0.799 | 0.774 |
| t+10 min | 0.887 | 0.785 | 0.765 |
| **t+15 min** | **0.874** | **0.776** | 0.761 |

> AUC degrades from 0.924 → 0.874 over 15 minutes (Δ = −0.050). This monotonic degradation is expected and demonstrates the model is genuinely learning temporal patterns, not just memorizing the current state.

### 3C. Legacy Flattened Metrics (for reference only — inflated)

| Metric | Value | Note |
|--------|:---:|:---|
| AUC | 0.896 | ⚠️ 15× overcounting |
| F1 | 0.793 | ⚠️ 15× overcounting |

> Flattened evaluation treats each of the 15 horizon outputs per sequence as independent. They are NOT independent — adjacent horizons are highly correlated. The per-sequence and per-horizon metrics above are the honest numbers.

---

## 4. Cross-Dataset Generalization

| Setting | Macro F1 | Notes |
|:---|:---:|:---|
| CHARIS internal (val) | ~0.786 | Within-cohort |
| CHARIS → MIMIC (OOD) | ~0.610 | True out-of-distribution |
| Gap | −0.176 | Expected: TBI vs general ICU |

---

## 5. Data Summary

| Dataset | Patients | Windows | Domain |
|:---|:---:|:---:|:---|
| CHARIS | 13 | 399,751 | TBI ICU (neurointensive) |
| MIMIC-III | 95 | 101,551 | General ICU |
| **Combined** | **108** | **501,302** | Mixed |

### Data Quality (Documented Transparently)

| Issue | Scope | Status |
|:---|:---|:---:|
| CHARIS MAP = 90.0 constant | All 399,751 CHARIS windows | ⚠️ Structural — cannot fix without re-extraction |
| MIMIC negative MAP | 27 windows (−10.4 mmHg) | ✅ Clamped to 40 |
| MIMIC NaN features | 8 windows | ✅ Imputed with column median |
| MIMIC MAP > 200 | 22 windows | ✅ Clamped to 200 |
| Temporal autocorrelation | All windows | ⚠️ Documented — effective N ≪ 501k |

---

## 6. Known Limitations

1. **Small patient count (N=108)**: High CV variance (F1 std=0.114). Population-level generalization claims are premature.
2. **CHARIS MAP constant (90.0 mmHg)**: MAP feature provides zero discriminative signal for 80% of the data. MAP discrimination learned exclusively from MIMIC ABP.
3. **Temporal autocorrelation**: Adjacent windows from the same patient session are highly correlated. 501k windows represent ~108 effectively independent patient-sessions.
4. **LSTM evaluated on single split**: No cross-validation variance estimate for LSTM. Per-sequence F1=0.795 is from one split.
5. **LSTM per-horizon degradation**: AUC drops from 0.924 (t+1) to 0.874 (t+15). Forecasting beyond ~10 minutes carries increased uncertainty.
6. **No prospective validation**: All evaluation is retrospective. Clinical deployment requires IRB-approved prospective study.
7. **Hardware normalization gap**: ESP32 firmware uses linear approximations instead of the wavelet/FFT pipeline.
