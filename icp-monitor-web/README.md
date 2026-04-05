# ICP Monitor — Clinical Decision Support Web Application

Professional hospital-grade interface for non-invasive intracranial pressure (ICP)
classification using XGBoost binary classifier with isotonic probability calibration.

**Research prototype — NOT FDA-approved — NOT for diagnostic use.**

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# From the icp-monitor-web/ directory
docker-compose up

# App available at:
#   Frontend: http://localhost:3000
#   Backend API: http://localhost:8001
#   API docs: http://localhost:8001/docs
```

Requires the model file at `../models/xgboost_binary.pkl.gz`
(relative to `icp-monitor-web/`).

---

### Option 2: Local development

**Backend**
```bash
cd backend/
pip install -r requirements.txt
MODEL_PATH=../../models/xgboost_binary.pkl.gz uvicorn main:app --reload --port 8001
```

**Frontend**
```bash
cd frontend/
npm install
npm run dev       # http://localhost:3000  (proxies /api → :8001)
```

---

## Features

### Tab 1: ICP Classification (XGBoost) — Fully Functional

| Feature | Description |
|---|---|
| CSV Upload | Drag-drop or browse, 10 MB max, real-time validation |
| Prediction Card | Class (Normal / Abnormal), confidence, probability bars |
| Trend Chart | Step-wise time-series with colour-coded ICP zones |
| SHAP Explainer | Top-3 contributing features per prediction |
| Session Summary | Pie chart, critical episode timeline |
| PDF Export | One-click report with summary, event log, live model metadata |

**Keyboard shortcuts:**
- `Ctrl+U` — open file picker
- `Ctrl+E` — export report
- `Ctrl+D` — toggle dark/light mode
- `Ctrl+H` — keyboard shortcut help
- `Ctrl+1/2/3` — switch tabs

### Tab 2: ICP Forecasting (LSTM) — Placeholder

- Professional "in development" UI with mock chart
- API stub: `POST /api/predict_forecast` → HTTP 501
- Roadmap for v3.0 integration

### Tab 3: Model Information

- Performance metrics (F1, AUC, Precision, Recall, Specificity, Balanced Accuracy)
- Global feature importance bar chart
- Hyperparameter table
- Feature physiological ranges
- Training data provenance (live from `binary_meta.json`)

---

## API Reference

Auto-generated docs available at `http://localhost:8001/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service health check |
| `/api/predict` | POST | Single-window classification with SHAP |
| `/api/predict_batch` | POST | Batch CSV classification |
| `/api/predict_forecast` | POST | LSTM stub (501) |
| `/api/model_info` | GET | Model metadata and metrics |
| `/api/example_csv` | GET | Sample CSV content |

**Request format for `/api/predict`:**
```json
{
  "features": [32.4, 1.2, 8.7, 1.30, 2.10, 95.0, 0.0, 0]
}
```

**CSV column order (6 columns):**
```
cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
slow_wave_power, cardiac_power, mean_arterial_pressure
```

---

## Model

| Property | Value |
|---|---|
| Type | XGBoost Binary Classifier v2.2 + Isotonic Calibration |
| Input | 6 physiological features (noise features removed by ablation) |
| Output | Normal (<15 mmHg) vs Abnormal (≥15 mmHg) |
| F1-Score | **0.8770** |
| AUC-ROC | **0.9490** |
| Balanced Accuracy | **0.8848** |
| ECE (calibrated) | **0.0972** |
| Threshold | 0.5450 (F1-optimised on val set) |
| Model size | 45.9 KB (gzipped) |
| Training data | CHARIS (13 patients) + MIMIC-III (87 patients) |
| Total windows | 448,537 |
| Calibration | 5-fold patient-level CV isotonic regression |

---

## Disclaimer

This system is a clinical decision support research prototype developed as a capstone
project. It is NOT FDA-approved and is NOT intended for autonomous diagnostic use.
All clinical decisions must be made by qualified medical professionals.
