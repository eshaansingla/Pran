# ICP Monitor — Clinical Decision Support Web Application

> **Dual-model clinical interface**: XGBoost real-time classification + BiLSTM 15-minute trend forecasting with Monte Carlo uncertainty estimation.

**Research prototype — NOT FDA-approved — NOT for diagnostic use.**

---

## Quick Start

### Option 1: Docker (recommended)

```bash
docker-compose up
# Frontend: http://localhost:3000
# Backend:  http://localhost:8001
# API docs: http://localhost:8001/docs
```

### Option 2: Local Development

```bash
# Backend
cd backend/
pip install -r requirements.txt
python -c "import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8001, reload=True)"

# Frontend (new terminal)
cd frontend/
npm install
npm run dev -- --port 3000
```

Requires model files at `../models/` (relative to `icp-monitor-web/`):
- `xgboost_binary.pkl.gz` — XGBoost classifier
- `best/lstm_forecast_v1.h5` + `best/lstm_meta.json` — LSTM forecaster

---

## Three-Tab Interface

### Tab 1: ICP Classification (XGBoost v2.2)

| Feature | Description |
|:---|:---|
| CSV Upload | Drag-drop or browse, 10 MB max, real-time validation |
| Alert Banner | Dynamic green "Normal" / red "CRITICAL" with severity |
| Stats Cards | Total windows, Abnormal %, Longest streak, Duration |
| Trend Chart | Step-wise time-series with Normal/Abnormal zones |
| Window Inspector | Click any window → modal with SHAP attribution |
| Clinical Summary | Auto-generated interpretive text |
| PDF Export | Structured medical report with clinical bounds flagging |
| Session History | Last 10 sessions stored locally |

### Tab 2: ICP Forecasting (BiLSTM v4.2) ✨

| Feature | Description |
|:---|:---|
| CSV Upload | Min 30 rows (5 min), same format as Classification |
| Example Data | One-click "Use Example Data" button (35 real CHARIS windows) |
| Early Warning | ⚠️ Clinical action items when Abnormal is predicted |
| Forecast Card | Class, P(Abnormal), 95% CI, confidence label |
| Forecast Chart | Historical (amber) + LSTM forecast (dashed) line chart |
| Attention Heatmap | Temporal feature importance across input windows |
| Window Analysis | Per-horizon breakdown of forecast probabilities |
| PDF Export | Structured medical report with forecast trajectory |

### Tab 3: Model Information

| Feature | Description |
|:---|:---|
| Dual-model metrics | F1, AUC, Precision, Recall for both XGBoost and LSTM |
| Feature importance | Horizontal bar chart (gain-based) |
| Hyperparameters | Full table for both models |
| Architecture details | BiLSTM topology, loss function, training config |

---

## API Endpoints

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/api/health` | GET | Service health check |
| `/api/predict` | POST | Single-window XGBoost classification + SHAP |
| `/api/predict_batch` | POST | Batch CSV XGBoost classification |
| `/api/predict_forecast` | POST | **LSTM 15-min forecast** with MC Dropout |
| `/api/model_info` | GET | Both models' metadata and metrics |
| `/api/example_csv` | GET | 35-row sample CSV (real abnormal CHARIS data) |

---

## Models

<table>
<tr>
<td>

**XGBoost v2.2**
| Metric | Value |
|:---|:---:|
| F1 | **0.877** |
| AUC | **0.949** |
| Threshold | 0.545 |
| Size | 45.9 KB |

</td>
<td>

**BiLSTM v4.2**
| Metric | Value |
|:---|:---:|
| F1 | **0.807** |
| AUC | **0.905** |
| Threshold | 0.525 |
| Horizon | 15 min |

</td>
</tr>
</table>

---

## Keyboard Shortcuts

| Shortcut | Action |
|:---|:---|
| `Ctrl+U` | Upload CSV |
| `Ctrl+E` | Export report |
| `Ctrl+D` | Toggle dark/light |
| `Ctrl+1/2/3` | Switch tabs |
| `← / →` | Navigate windows |

---

*See the [main README](../README.md) for full documentation, architecture diagrams, and training details.*
