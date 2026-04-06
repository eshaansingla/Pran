"""
main.py
=======
FastAPI application — ICP Classification Web Service
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import model_loader
from model_loader import (
    FEATURE_NAMES,
    get_model_info,
    load_model,
    predict_batch,
    predict_single,
)
from validation import ValidationError, parse_csv_bytes, validate_feature_vector
import lstm_predictor

app = FastAPI(
    title="ICP Monitor API",
    description=(
        "Clinical decision support API for intracranial pressure "
        "classification using XGBoost. "
        "NOT FOR DIAGNOSTIC USE. Research / capstone prototype only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to hospital intranet in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    load_model()   # warm up once


# ---------------------------------------------------------------------------
# Models (Pydantic)
# ---------------------------------------------------------------------------

class SinglePredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        from model_loader import FEATURE_NAMES
        n = len(FEATURE_NAMES)
        if len(v) != n:
            raise ValueError(f"Expected {n} features, got {len(v)}")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": _utcnow()}


@app.get("/api/model_info")
async def model_info() -> dict[str, Any]:
    return get_model_info()


@app.get("/api/lstm_info")
async def lstm_info() -> JSONResponse:
    """LSTM forecaster metadata — returns {} with 404 if not trained yet."""
    from pathlib import Path
    import json
    meta_path = Path(__file__).parent.parent.parent / "models" / "lstm_meta.json"
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "LSTM model not trained yet"})
    return JSONResponse(content=json.loads(meta_path.read_text()))


@app.post("/api/predict")
async def predict(req: SinglePredictRequest) -> dict[str, Any]:
    """
    Classify a single 10-second ICP monitoring window.

    Input: 6 extracted physiological features (cardiac_amplitude,
    cardiac_frequency, respiratory_amplitude, slow_wave_power,
    cardiac_power, mean_arterial_pressure).
    Output: ICP class (0=Normal, 1=Abnormal), probabilities,
            confidence, SHAP-based top contributing features.
    """
    errors = validate_feature_vector(req.features)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    result = predict_single(req.features)
    result["timestamp"] = _utcnow()
    return result


@app.post("/api/predict_batch")
async def predict_batch_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Classify multiple windows from a CSV upload.

    CSV must have 6 columns matching the feature specification.
    Returns per-window predictions plus session summary statistics.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail="Only .csv files are accepted"
        )

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    rows, parse_errors = parse_csv_bytes(raw)

    if not rows:
        raise HTTPException(
            status_code=422,
            detail={"validation_errors": parse_errors or ["No valid data rows found"]}
        )

    predictions = predict_batch(rows)

    counts = [0, 0]
    for p in predictions:
        counts[p["class"]] += 1

    return {
        "predictions": predictions,
        "parse_warnings": parse_errors,   # non-fatal range warnings
        "summary": {
            "total":        len(predictions),
            "normal":       counts[0],
            "abnormal":     counts[1],
            "normal_pct":   _pct(counts[0], len(predictions)),
            "abnormal_pct": _pct(counts[1], len(predictions)),
        },
        "calibrated":    model_loader._calibrated,
        "timestamp":     _utcnow(),
        "feature_names": FEATURE_NAMES,
    }


class ForecastRequest(BaseModel):
    sequence: list[list[float]]

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 30:
            raise ValueError(
                f"Sequence must contain at least 30 windows (got {len(v)}). "
                "Each window is a 10-second monitoring interval."
            )
        for i, row in enumerate(v):
            if len(row) != 6:
                raise ValueError(
                    f"Each window must have exactly 6 features (row {i} has {len(row)})."
                )
        return v


@app.post("/api/predict_forecast")
async def predict_forecast(req: ForecastRequest) -> JSONResponse:
    """
    LSTM-based ICP trend forecasting — 15 minutes ahead.

    Input: JSON body with 'sequence' key containing ≥30 windows (rows of 6 features).
    The model uses the last 30 windows as context.

    Output: Forecast class, calibrated probability, 95% CI, attention weights,
            and a plain-English interpretation.

    Requires models/lstm_forecast_v1.h5 to be present.
    Run  python src/models/lstm_forecaster.py  to train the model first.
    """
    if not lstm_predictor.lstm_available():
        err = lstm_predictor.get_load_error()
        return JSONResponse(
            status_code=501,
            content={
                "error": "LSTM forecasting model not available",
                "detail": err or "Model not loaded",
                "action": "Run  python src/models/lstm_forecaster.py  from the project root.",
            },
        )

    try:
        result = lstm_predictor.predict_forecast(req.sequence)
        result["timestamp"] = _utcnow()
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast inference failed: {exc}")


@app.get("/api/example_csv")
async def example_csv() -> JSONResponse:
    """Return CSV content and column documentation."""
    header = ",".join(FEATURE_NAMES)
    examples = [
        "32.4,1.2,8.7,1.30,2.10,95.0",
        "28.1,1.1,7.2,1.65,2.55,92.0",
        "45.6,1.3,12.3,1.80,3.20,98.0",
        "38.9,1.25,9.8,2.10,2.90,101.0",
        "52.3,1.4,14.5,2.80,3.60,105.0",
        "21.7,1.0,6.4,1.15,1.85,88.0",
        "61.2,1.5,18.2,3.20,4.10,110.0",
        "19.4,0.95,5.8,1.05,1.60,82.0",
        "47.8,1.35,11.6,2.40,3.35,97.0",
        "35.1,1.2,9.1,1.70,2.65,93.0",
    ]
    csv_content = header + "\n" + "\n".join(examples)
    return JSONResponse({"csv": csv_content, "header": FEATURE_NAMES})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0
