"""
model_loader.py
===============
XGBoost binary model loading, inference, and SHAP-based feature attribution.

Binary classification: Normal (<15 mmHg) vs Abnormal (>=15 mmHg).

Rationale: the 3-class model's Elevated decision region collapsed to
max 3% probability under any input. Binary at the 15 mmHg clinical
intervention threshold is honest and well-calibrated (F1=0.88, AUC=0.96).
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
    "cardiac_amplitude":      "um",
    "cardiac_frequency":      "Hz",
    "respiratory_amplitude":  "um",
    "slow_wave_power":        "",
    "cardiac_power":          "",
    "mean_arterial_pressure": "mmHg",
    "head_angle":             "deg",
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
CLASS_NAMES    = ["Normal", "Abnormal"]
THRESHOLD      = 0.5   # probability threshold for Abnormal

_model: xgb.Booster | None = None
_global_importances: dict[str, float] = {}


def _default_model_path() -> Path:
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "models" / "xgboost_binary.pkl.gz"


def load_model(path: Path | None = None) -> xgb.Booster:
    global _model, _global_importances
    if _model is not None:
        return _model

    model_path = path or _default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Set MODEL_PATH env var or place xgboost_binary.pkl.gz in models/."
        )

    if model_path.suffix == ".gz":
        with gzip.open(model_path, "rb") as fh:
            _model = pickle.load(fh)
    else:
        with open(model_path, "rb") as fh:
            _model = pickle.load(fh)

    raw = _model.get_score(importance_type="gain")
    fi_map = {f"f{i}": name for i, name in enumerate(FEATURE_NAMES)}
    total = sum(raw.values()) or 1.0
    _global_importances = {
        fi_map.get(k, k): v / total for k, v in raw.items()
    }
    return _model


def _make_dmatrix(features: list[float]) -> xgb.DMatrix:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    return xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(len(features))])


def predict_single(features: list[float]) -> dict[str, Any]:
    bst = load_model()
    dm  = _make_dmatrix(features)

    prob_abnormal = float(bst.predict(dm)[0])
    pred_class    = int(prob_abnormal >= THRESHOLD)
    prob_normal   = 1.0 - prob_abnormal

    # SHAP contributions: (1, 9) for binary
    contribs      = bst.predict(dm, pred_contribs=True)[0]   # (9,)
    shap_vals     = contribs[:8]
    abs_shap      = np.abs(shap_vals)
    top3_idx      = abs_shap.argsort()[::-1][:3]

    lo  = [FEATURE_RANGES[FEATURE_NAMES[i]][0] for i in range(8)]
    hi  = [FEATURE_RANGES[FEATURE_NAMES[i]][1] for i in range(8)]
    mid = [(lo[i] + hi[i]) / 2 for i in range(8)]
    rng = [(hi[i] - lo[i]) or 1.0 for i in range(8)]

    top_features = []
    for idx in top3_idx:
        i   = int(idx)
        val = features[i]
        name = FEATURE_NAMES[i]
        z = (val - mid[i]) / (rng[i] / 2)
        status = "HIGH" if z > 0.33 else "LOW" if z < -0.33 else "NORMAL"
        top_features.append({
            "name":       name,
            "value":      round(float(val), 3),
            "unit":       FEATURE_UNITS[name],
            "status":     status,
            "shap":       round(float(shap_vals[i]), 4),
            "impact_pct": round(float(abs_shap[i]) / float(abs_shap.sum() or 1.0) * 100, 1),
        })

    return {
        "class":          pred_class,
        "class_name":     CLASS_NAMES[pred_class],
        "probability":    round(prob_abnormal, 4),
        "probabilities":  [round(prob_normal, 4), round(prob_abnormal, 4)],
        "confidence":     round(max(prob_normal, prob_abnormal), 4),
        "top_features":   top_features,
    }


def predict_batch(feature_matrix: list[list[float]]) -> list[dict[str, Any]]:
    bst = load_model()
    X   = np.array(feature_matrix, dtype=np.float32)
    dm  = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])

    probs = bst.predict(dm)   # (N,) — P(Abnormal)
    results = []
    for i, p in enumerate(probs):
        cls = int(p >= THRESHOLD)
        results.append({
            "window_id":    i + 1,
            "class":        cls,
            "class_name":   CLASS_NAMES[cls],
            "probability":  round(float(p), 4),
            "probabilities":[round(float(1-p), 4), round(float(p), 4)],
            "confidence":   round(float(max(1-p, p)), 4),
        })
    return results


def get_model_info() -> dict[str, Any]:
    load_model()
    return {
        "version":       "2.0",
        "model_type":    "XGBoost Binary",
        "classifier":    "Normal (<15 mmHg) vs Abnormal (>=15 mmHg)",
        "threshold_mmhg": 15.0,
        "metrics": {
            "f1":               0.8796,
            "auc":              0.9623,
            "precision":        0.9416,
            "recall":           0.8252,
            "specificity":      0.9409,
            "balanced_accuracy":0.8831,
        },
        "training_date": "2026-04-03",
        "training_data": {
            "charis_patients": 13,
            "mimic_patients":  36,
            "total_windows":   409315,
        },
        "features":          FEATURE_NAMES,
        "feature_units":     FEATURE_UNITS,
        "feature_ranges":    {k: list(v) for k, v in FEATURE_RANGES.items()},
        "classes":           CLASS_NAMES,
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth":     4,
            "n_estimators":  420,
            "subsample":     0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 0.4756,
        },
        "global_importances": _global_importances,
        "previous_model_note": (
            "3-class model retired: Elevated class max probability was 3% "
            "under any input — effectively binary already. This model is honest."
        ),
    }
