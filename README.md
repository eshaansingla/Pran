# Non-Invasive ICP Monitoring — Capstone Project

Optical tympanic membrane (TM) sensor + machine learning pipeline for
non-invasive intracranial pressure (ICP) classification.
Trained on PhysioNet data (CHARIS + MIMIC-III).

---

## Project Overview

Intracranial pressure monitoring currently requires invasive neurosurgical procedures.
This project builds a non-invasive alternative using optical signals from the tympanic
membrane, extracting physiological features, and classifying ICP state using XGBoost.
A clinical web interface allows real-time monitoring and PDF report generation.

**Target:** IEEE EMBC submission / capstone committee review.

---

## Repository Structure

```
Pran/
├── src/
│   ├── data/
│   │   ├── download_physionet.py       # PhysioNet credential-based download
│   │   ├── segment_windows.py          # 10-second window extraction (125 Hz)
│   │   ├── extract_features.py         # 8-feature physiological extraction
│   │   ├── generate_labels.py          # ICP threshold labelling
│   │   └── save_processed_data.py      # Save to data/processed/
│   └── models/
│       ├── xgboost_classifier.py       # XGBoost training + SHAP + ablation
│       └── model_evaluation.py         # Confusion matrix, ROC, learning curves
│
├── build_mimic_features.py             # Stream-read MIMIC-III ICP via wfdb
├── combine_and_retrain.py              # CHARIS + MIMIC combined retraining
├── cross_dataset_eval.py               # Honest OOD generalisation test
├── train_final.py                      # Final model: dataset-stratified split
├── run_pipeline.py                     # End-to-end pipeline runner
├── download_charis.py                  # CHARIS dataset downloader
├── predict_from_hardware.py            # Live inference from sensor hardware
│
├── icp-monitor-web/                    # Clinical web interface
│   ├── backend/                        # FastAPI inference server
│   │   ├── main.py
│   │   ├── model_loader.py
│   │   ├── validation.py
│   │   ├── requirements.txt
│   │   ├── start.ps1                   # Windows PowerShell launcher
│   │   ├── start.bat                   # Windows cmd launcher
│   │   └── Dockerfile
│   ├── frontend/                       # React 18 + TypeScript + Tailwind
│   │   ├── src/
│   │   │   ├── components/             # UploadZone, PredictionCard, TrendChart,
│   │   │   │                           # FeatureExplainer, SessionSummary,
│   │   │   │                           # ReportExporter, LSTMPlaceholder
│   │   │   ├── pages/                  # Dashboard, Forecasting, ModelInfo
│   │   │   ├── types/index.ts
│   │   │   └── utils/                  # api.ts, formatters.ts
│   │   ├── tailwind.config.js
│   │   ├── vite.config.ts
│   │   └── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
│
├── data/
│   └── sample_hardware_data.csv        # Example hardware input
└── .gitignore
```

---

## Data

| Dataset | Patients | Windows | ICP Distribution |
|---|---|---|---|
| CHARIS (TBI ICU) | 13 | ~400k | Balanced: 38% N / 26% E / 36% C |
| MIMIC-III (Gen. ICU) | 36 | ~10k | Skewed: 87% N / 9% E / 4% C |
| **Combined** | **49** | **~409k** | 39% N / 25% E / 36% C |

**Label thresholds:** Normal < 15 mmHg · Elevated 15-20 mmHg · Critical >= 20 mmHg

Data is not committed (patient data, PhysioNet licence). To reproduce:

```bash
python download_charis.py
python build_mimic_features.py
```

---

## ML Pipeline

### Feature Extraction (8 features per 10-second window)

| Feature | Description | Unit |
|---|---|---|
| cardiac_amplitude | Peak-to-peak of 1-2 Hz bandpass | um |
| cardiac_frequency | Dominant frequency 0.7-2.5 Hz | Hz |
| respiratory_amplitude | Peak-to-peak of 0.1-0.5 Hz bandpass | um |
| slow_wave_power | Wavelet energy ratio (D5 level) | - |
| cardiac_power | Wavelet energy ratio (D4 level) | - |
| mean_arterial_pressure | Mean ABP or signal mean proxy | mmHg |
| head_angle | Patient head elevation | deg |
| motion_artifact_flag | Binary motion artifact indicator | - |

Phase-lag features were removed after ablation study (delta F1 = -0.026).

### Model

**XGBoost** with patient-stratified 70/10/20 split (GroupShuffleSplit by patient_id).
Dataset-stratified: CHARIS and MIMIC patients split independently then merged,
preventing test set from being accidentally dominated by one cohort's distribution.

| Hyperparameter | Value |
|---|---|
| learning_rate | 0.1 |
| max_depth | 4 |
| n_estimators | 180 (early stopping on CHARIS val) |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

```bash
python train_final.py
```

---

## Results

### Final Model — Combined CHARIS + MIMIC (dataset-stratified split)

| Metric | Value |
|---|---|
| Macro F1 | **0.7667** |
| Weighted F1 | 0.7978 |
| Balanced Accuracy | 0.7686 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.88 | 0.85 | 0.87 | 26,031 |
| Elevated | 0.57 | 0.59 | 0.58 | 12,725 |
| Critical | 0.84 | 0.86 | 0.85 | 17,350 |

### Cross-Dataset Generalisation (CHARIS -> MIMIC OOD)

Trained on CHARIS only (13 patients), tested on all 36 MIMIC patients never seen during training:

| Metric | CHARIS internal | MIMIC OOD |
|---|---|---|
| Macro F1 | 0.786 | **0.610** |
| Balanced Accuracy | - | 0.735 |

| Class | F1 on MIMIC |
|---|---|
| Normal | 0.825 |
| Elevated | 0.338 |
| Critical | 0.665 |

Generalisation gap: +0.176, driven by class distribution shift
(CHARIS: balanced 38/26/36% vs MIMIC: skewed 87/9/4%).

```bash
python cross_dataset_eval.py
```

---

## Clinical Web Interface

Hospital-grade React + FastAPI application. Clean EMR-style aesthetic,
WCAG AA compliant, keyboard accessible.

### Run locally

**Terminal 1 — Backend (PowerShell):**
```powershell
cd icp-monitor-web\backend
.\start.ps1
# Runs on http://localhost:8001
# API docs: http://localhost:8001/docs
```

**Terminal 2 — Frontend:**
```powershell
cd icp-monitor-web\frontend
npm install
npm run dev
# App: http://localhost:3000
```

**Or with Docker:**
```bash
cd icp-monitor-web
docker-compose up
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Single window — class + SHAP top-3 features |
| `/api/predict_batch` | POST | CSV upload — batch classification + summary |
| `/api/predict_forecast` | POST | LSTM stub (HTTP 501, planned v2.0) |
| `/api/model_info` | GET | Metrics, importances, feature ranges |
| `/api/example_csv` | GET | Downloadable sample CSV |

### Interface Tabs

| Tab | Status | Description |
|---|---|---|
| ICP Classification | Fully functional | CSV upload, trend chart, SHAP explainer, PDF export |
| ICP Forecasting | Placeholder (v2.0) | LSTM 15-min forecast — in development, Q3 2026 |
| Model Info | Fully functional | Metrics, feature importance, hyperparameters, disclaimer |

### CSV Format

```
cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,
cardiac_power,mean_arterial_pressure,head_angle,motion_artifact_flag
32.4,1.2,8.7,1.30,2.10,95.0,0.0,0
...
```

---

## Disclaimer

This is a **research prototype** developed as a capstone project.
NOT FDA-approved. NOT for autonomous diagnostic use.
All clinical decisions must be made by qualified medical professionals.
