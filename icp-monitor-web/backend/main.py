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

from model_loader import (
    FEATURE_NAMES,
    get_model_info,
    load_model,
    predict_batch,
    predict_single,
)
from validation import ValidationError, parse_csv_bytes, validate_feature_vector

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
        if len(v) != 8:
            raise ValueError(f"Expected 8 features, got {len(v)}")
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


@app.post("/api/predict")
async def predict(req: SinglePredictRequest) -> dict[str, Any]:
    """
    Classify a single 10-second ICP monitoring window.

    Input: 8 extracted physiological features.
    Output: ICP class (0=Normal, 1=Elevated, 2=Critical), probabilities,
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

    CSV must have 8 columns matching the feature specification.
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

    counts = [0, 0, 0]
    for p in predictions:
        counts[p["class"]] += 1

    return {
        "predictions": predictions,
        "parse_warnings": parse_errors,   # non-fatal range warnings
        "summary": {
            "total": len(predictions),
            "normal":   counts[0],
            "elevated": counts[1],
            "critical": counts[2],
            "normal_pct":   _pct(counts[0], len(predictions)),
            "elevated_pct": _pct(counts[1], len(predictions)),
            "critical_pct": _pct(counts[2], len(predictions)),
        },
        "timestamp": _utcnow(),
        "feature_names": FEATURE_NAMES,
    }


@app.post("/api/predict_forecast")
async def predict_forecast() -> JSONResponse:
    """
    LSTM-based ICP trend forecasting.

    STATUS: Not yet implemented. Planned for v2.0 (Q3 2026).
    Will accept sequences of 30+ consecutive windows and return
    predicted ICP trajectory with 95% confidence intervals.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": "LSTM forecasting not yet available",
            "status": "in_development",
            "expected_release": "v2.0 (Q3 2026)",
            "description": (
                "Future endpoint will accept POST with multipart CSV "
                "containing >= 30 consecutive 10-second windows and return "
                "15-30 minute ICP trajectory predictions."
            ),
        },
    )


@app.get("/api/example_csv")
async def example_csv() -> JSONResponse:
    """Return CSV content and column documentation."""
    header = ",".join(FEATURE_NAMES)
    examples = [
        "32.4,1.2,8.7,1.30,2.10,95.0,0.0,0",
        "28.1,1.1,7.2,1.65,2.55,92.0,0.0,0",
        "45.6,1.3,12.3,1.80,3.20,98.0,0.0,0",
        "38.9,1.25,9.8,2.10,2.90,101.0,5.0,0",
        "52.3,1.4,14.5,2.80,3.60,105.0,0.0,0",
        "21.7,1.0,6.4,1.15,1.85,88.0,0.0,0",
        "61.2,1.5,18.2,3.20,4.10,110.0,0.0,0",
        "19.4,0.95,5.8,1.05,1.60,82.0,-5.0,0",
        "47.8,1.35,11.6,2.40,3.35,97.0,0.0,0",
        "35.1,1.2,9.1,1.70,2.65,93.0,0.0,0",
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
