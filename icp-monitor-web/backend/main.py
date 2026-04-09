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
    """Return CSV content from the canonical example file.

    100 windows (≈ 16.7 min session) with sinusoidal physiological variation.
    Features are raw wavelet fractions as produced by the optical TM sensor.
    Supports both XGBoost (any rows) and LSTM (≥30 consecutive windows).
    """
    from pathlib import Path
    import os

    # Locate example CSV relative to this file
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "lstm_input_correct.csv",
        Path(__file__).parent.parent.parent / "data" / "sample_hardware_data.csv",
    ]
    example_path = next((p for p in candidates if p.exists()), None)

    if example_path and example_path.exists():
        csv_content = example_path.read_text()
    else:
        # Fallback: minimal 35-window example with physiologically reasonable raw fractions
        header = ",".join(FEATURE_NAMES)
        fallback_rows = [
            f"{20 + i * 1.1:.4f},{0.9 + (i % 5) * 0.1:.1f},{5 + i * 0.4:.4f},"
            f"{0.997 - i * 0.0003:.4f},{0.005 + i * 0.0005:.6f},{80 + i * 0.5:.4f}"
            for i in range(35)
        ]
        csv_content = header + "\n" + "\n".join(fallback_rows)

    return JSONResponse({"csv": csv_content, "header": FEATURE_NAMES})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0
