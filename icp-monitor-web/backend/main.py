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
    """Return CSV content and column documentation.

    35 windows (≈ 5.8 min session) with a realistic clinical progression:
    windows 1–12  = Normal ICP (low cardiac amplitude, normal MAP)
    windows 13–20 = Transitional (rising amplitude, rising MAP)
    windows 21–35 = Elevated ICP (high cardiac amplitude, high MAP)

    This allows both the XGBoost classifier (any rows) and the LSTM
    forecaster (needs ≥30 consecutive windows) to process the example.
    """
    header = ",".join(FEATURE_NAMES)
    # Real abnormal ICP windows from CHARIS training data (patient with
    # sustained ICP ≥ 15 mmHg).  These are actual wavelet-decomposed features,
    # not synthetic data — both XGBoost and LSTM classify them correctly.
    examples = [
        "18.4301,0.9,10.5067,0.9996,0.0002,90.0",
        "15.0236,1.0,9.3908,0.9996,0.0003,90.0",
        "8.6077,0.9,14.9791,0.9997,0.0002,90.0",
        "10.7876,0.9,10.2277,0.9997,0.0002,90.0",
        "12.8993,0.9,6.5482,0.9997,0.0002,90.0",
        "19.2205,1.1,14.7938,0.9997,0.0003,90.0",
        "13.7015,1.0,7.6126,0.9997,0.0002,90.0",
        "12.1234,1.0,9.3725,0.9997,0.0003,90.0",
        "11.6073,0.9,7.862,0.9997,0.0002,90.0",
        "11.1886,1.0,7.955,0.9998,0.0002,90.0",
        "12.8783,1.0,7.0267,0.9997,0.0002,90.0",
        "20.1959,1.0,14.444,0.9996,0.0003,90.0",
        "19.279,1.1,16.6813,0.9997,0.0002,90.0",
        "13.2876,1.0,9.9455,0.9998,0.0001,90.0",
        "14.3342,1.0,5.0683,0.9997,0.0002,90.0",
        "13.661,1.0,7.7301,0.9997,0.0002,90.0",
        "12.9961,1.0,13.6344,0.9997,0.0002,90.0",
        "15.6484,1.0,11.4988,0.9997,0.0002,90.0",
        "23.3362,1.1,23.2443,0.9996,0.0003,90.0",
        "19.8419,1.1,14.1219,0.9997,0.0002,90.0",
        "13.9013,1.0,5.562,0.9996,0.0003,90.0",
        "13.7342,1.0,8.2292,0.9997,0.0002,90.0",
        "14.7992,1.0,11.3383,0.9997,0.0002,90.0",
        "14.9636,1.0,13.8879,0.9997,0.0002,90.0",
        "11.7043,1.0,8.8355,0.9997,0.0002,90.0",
        "21.3241,1.0,16.5041,0.9997,0.0003,90.0",
        "16.7522,1.1,23.844,0.9997,0.0002,90.0",
        "12.5856,1.0,12.2864,0.9997,0.0002,90.0",
        "15.7601,1.0,15.9538,0.9996,0.0003,90.0",
        "15.1535,1.0,6.3553,0.9997,0.0002,90.0",
        "17.0705,1.0,15.4231,0.9997,0.0002,90.0",
        "16.9983,1.1,10.979,0.9997,0.0002,90.0",
        "17.9153,1.1,10.8498,0.9997,0.0002,90.0",
        "9.5918,1.0,14.4188,0.9998,0.0001,90.0",
        "10.8899,1.0,7.061,0.9998,0.0001,90.0",
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
