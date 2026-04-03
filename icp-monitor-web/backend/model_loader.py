"""
model_loader.py
===============
XGBoost model loading, inference, and SHAP-based feature attribution.
"""
from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
    "head_angle",
    "motion_artifact_flag",
]
FEATURE_UNITS = {
    "cardiac_amplitude":      "μm",
    "cardiac_frequency":      "Hz",
    "respiratory_amplitude":  "μm",
    "slow_wave_power":        "",
    "cardiac_power":          "",
    "mean_arterial_pressure": "mmHg",
    "head_angle":             "°",
    "motion_artifact_flag":   "",
}
FEATURE_RANGES = {
    "cardiac_amplitude":      (10.0, 80.0),
    "cardiac_frequency":      (0.8, 2.5),
    "respiratory_amplitude":  (2.0, 30.0),
    "slow_wave_power":        (0.05, 5.0),
    "cardiac_power":          (0.1, 5.0),
    "mean_arterial_pressure": (50.0, 150.0),
    "head_angle":             (-20.0, 90.0),
    "motion_artifact_flag":   (0.0, 1.0),
}
CLASS_NAMES = ["Normal", "Elevated", "Critical"]

# Global model instance (loaded once at startup)
_model: xgb.Booster | None = None
_global_importances: dict[str, float] = {}


def _default_model_path() -> Path:
    """Resolve model path: env var > sibling 'models/' folder."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    # backend/  →  ../models/
    return Path(__file__).parent.parent / "models" / "xgboost_final.pkl.gz"


def load_model(path: Path | None = None) -> xgb.Booster:
    global _model, _global_importances
    if _model is not None:
        return _model

    model_path = path or _default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Set MODEL_PATH env var or place xgboost_final.pkl.gz in models/."
        )

    if model_path.suffix == ".gz":
        with gzip.open(model_path, "rb") as fh:
            _model = pickle.load(fh)
    else:
        with open(model_path, "rb") as fh:
            _model = pickle.load(fh)

    # Pre-compute global gain importances (feature index → name mapping)
    raw_scores = _model.get_score(importance_type="gain")
    fi_map = {f"f{i}": name for i, name in enumerate(FEATURE_NAMES)}
    total = sum(raw_scores.values()) or 1.0
    _global_importances = {
        fi_map.get(k, k): v / total
        for k, v in raw_scores.items()
    }

    return _model


def _make_dmatrix(features: list[float]) -> xgb.DMatrix:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    return xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(len(features))])


def predict_single(features: list[float]) -> dict[str, Any]:
    """Run inference for one feature vector (8 values)."""
    bst = load_model()
    dm = _make_dmatrix(features)

    proba = bst.predict(dm)[0]          # shape (3,)
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    # SHAP contributions: (1, 3, 9) → (3, 8) trimming bias column
    contribs = bst.predict(dm, pred_contribs=True)[0]   # (3, 9)
    shap_for_class = contribs[pred_class, :8]            # (8,)
    abs_shap = np.abs(shap_for_class)
    top3_idx = abs_shap.argsort()[::-1][:3]

    lo, hi = (
        [FEATURE_RANGES[FEATURE_NAMES[i]][0] for i in range(8)],
        [FEATURE_RANGES[FEATURE_NAMES[i]][1] for i in range(8)],
    )
    mid = [(lo[i] + hi[i]) / 2 for i in range(8)]
    rng = [(hi[i] - lo[i]) or 1.0 for i in range(8)]

    top_features = []
    for idx in top3_idx:
        val = features[idx]
        name = FEATURE_NAMES[idx]
        z = (val - mid[idx]) / (rng[idx] / 2)
        if z > 0.33:
            status = "HIGH"
        elif z < -0.33:
            status = "LOW"
        else:
            status = "NORMAL"
        top_features.append({
            "name": name,
            "value": round(float(val), 3),
            "unit": FEATURE_UNITS[name],
            "status": status,
            "shap": round(float(shap_for_class[int(idx)]), 4),
            "impact_pct": round(
                float(abs_shap[int(idx)]) / float(abs_shap.sum() or 1.0) * 100, 1
            ),
        })

    return {
        "class": int(pred_class),
        "class_name": CLASS_NAMES[pred_class],
        "probabilities": [round(float(p), 4) for p in proba.tolist()],
        "confidence": round(float(confidence), 4),
        "top_features": top_features,
    }


def predict_batch(feature_matrix: list[list[float]]) -> list[dict[str, Any]]:
    """Run inference for N rows. Returns list of prediction dicts."""
    bst = load_model()
    X = np.array(feature_matrix, dtype=np.float32)
    dm = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])

    proba = bst.predict(dm)                    # (N, 3)
    pred_classes = proba.argmax(axis=1)

    results = []
    for i, (cls, p) in enumerate(zip(pred_classes, proba)):
        results.append({
            "window_id": i + 1,
            "class": int(cls),
            "class_name": CLASS_NAMES[int(cls)],
            "probabilities": [round(float(v), 4) for v in p],
            "confidence": round(float(p[cls]), 4),
        })
    return results


def get_model_info() -> dict[str, Any]:
    load_model()
    return {
        "version": "1.0",
        "model_type": "XGBoost",
        "metrics": {
            "macro_f1": 0.7667,
            "weighted_f1": 0.7978,
            "balanced_accuracy": 0.7686,
            "auc_normal": 0.954,
            "auc_elevated": 0.871,
            "auc_critical": 0.943,
        },
        "training_date": "2026-04-03",
        "training_data": {
            "charis_patients": 13,
            "mimic_patients": 36,
            "total_windows": 409315,
        },
        "features": FEATURE_NAMES,
        "feature_units": FEATURE_UNITS,
        "feature_ranges": {k: list(v) for k, v in FEATURE_RANGES.items()},
        "classes": CLASS_NAMES,
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 4,
            "n_estimators": 180,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "global_importances": _global_importances,
    }
