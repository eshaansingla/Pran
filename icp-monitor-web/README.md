# ICP Monitor — Clinical Decision Support Web Application

Professional hospital-grade interface for non-invasive intracranial pressure (ICP)
classification using XGBoost, with LSTM forecasting planned for v2.0.

**Research prototype — NOT FDA-approved — NOT for diagnostic use.**

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# From the icp-monitor-web/ directory
docker-compose up

# App available at:
#   Frontend: http://localhost:3000
#   Backend API: http://localhost:8000
#   API docs: http://localhost:8000/docs
```

Requires the model file at `../models/xgboost_final.pkl.gz`
(relative to `icp-monitor-web/`).

---

### Option 2: Local development

**Backend**
```bash
cd backend/
pip install -r requirements.txt
MODEL_PATH=../../models/xgboost_final.pkl.gz uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend/
npm install
npm run dev       # http://localhost:3000  (proxies /api → :8000)
```

---

## Features

### Tab 1: ICP Classification (XGBoost) — Fully Functional

| Feature | Description |
|---|---|
| CSV Upload | Drag-drop or browse, 10 MB max, real-time validation |
| Prediction Card | Class (Normal / Elevated / Critical), confidence, probability bars |
| Trend Chart | Step-wise time-series with colour-coded ICP zones |
| SHAP Explainer | Top-3 contributing features per prediction |
| Session Summary | Pie chart, critical episode timeline |
| PDF Export | One-click report with summary, event log, model metadata |

**Keyboard shortcuts:**
- `Ctrl+U` — open file picker
- `Ctrl+1/2/3` — switch tabs

### Tab 2: ICP Forecasting (LSTM) — Placeholder

- Professional "in development" UI with mock chart
- API stub: `POST /api/predict_forecast` → HTTP 501
- Roadmap for v2.0 integration documented in code

### Tab 3: Model Information

- Performance metrics (Macro F1, Balanced Accuracy, AUC per class)
- Global feature importance bar chart
- Hyperparameter table
- Feature physiological ranges
- Training data provenance

---

## API Reference

Auto-generated docs available at `http://localhost:8000/docs`.

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

**CSV column order:**
```
cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
slow_wave_power, cardiac_power, mean_arterial_pressure,
head_angle, motion_artifact_flag
```

---

## Model

- **Type:** XGBoost (Gradient Boosting, 180 trees)
- **Input:** 8 physiological features extracted from 10-second ICP windows
- **Output:** 3 classes — Normal (<15 mmHg), Elevated (15-20 mmHg), Critical (≥20 mmHg)
- **Macro F1:** 0.7667 | **Balanced Accuracy:** 0.7686
- **Training data:** CHARIS (13 TBI patients) + MIMIC-III (36 general ICU patients)

---

## Disclaimer

This system is a clinical decision support research prototype developed as a capstone
project. It is NOT FDA-approved and is NOT intended for autonomous diagnostic use.
All clinical decisions must be made by qualified medical professionals.
