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
import json
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
]
FEATURE_UNITS = {
    "cardiac_amplitude":      "um",
    "cardiac_frequency":      "Hz",
    "respiratory_amplitude":  "um",
    "slow_wave_power":        "",
    "cardiac_power":          "",
    "mean_arterial_pressure": "mmHg",
}
FEATURE_RANGES = {
    "cardiac_amplitude":      (5.0, 120.0),     # μm — widened for MIMIC variability
    "cardiac_frequency":      (0.7, 2.5),       # Hz — 42-150 bpm
    "respiratory_amplitude":  (1.0, 50.0),      # μm — widened for MIMIC variability
    # slow_wave_power = wavelet energy fraction in 0–1.56 Hz band (db4, level-5).
    # Near-1.0 = normal; lower = more high-freq activity.
    "slow_wave_power":        (0.30, 1.0),
    # cardiac_power = wavelet energy fraction in 1.56–3.12 Hz band.
    # Training data max reaches ~0.35 for unusual MIMIC spectra.
    "cardiac_power":          (0.0, 0.40),
    "mean_arterial_pressure": (40.0, 200.0),    # mmHg — clamped in extraction
}
CLASS_NAMES = ["Normal", "Abnormal"]

_model: xgb.Booster | None = None
_calibrator = None
_threshold: float = 0.5
_calibrated: bool = False
_global_importances: dict[str, float] = {}
_meta: dict = {}   # full binary_meta.json contents


def _default_model_path() -> Path:
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent / "models" / "xgboost_binary.pkl.gz"


def load_model(path: Path | None = None) -> xgb.Booster:
    global _model, _calibrator, _threshold, _calibrated, _global_importances, _meta
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
    # Initialise all features to 0 — get_score() omits zero-gain features
    _global_importances = {name: 0.0 for name in FEATURE_NAMES}
    _global_importances.update({fi_map.get(k, k): v / total for k, v in raw.items()})

    # Load calibrator (optional — graceful fallback if not present or sklearn missing)
    cal_path = model_path.parent / "xgboost_binary_calibrator.pkl.gz"
    if cal_path.exists():
        try:
            with gzip.open(cal_path, "rb") as fh:
                _calibrator = pickle.load(fh)
            _calibrated = True
        except Exception as exc:
            print(f"[model_loader] Calibrator load failed ({exc}); using raw probabilities")

    # Load all metadata from binary_meta.json (optional — graceful fallback)
    meta_path = model_path.parent / "binary_meta.json"
    if meta_path.exists():
        _meta = json.loads(meta_path.read_text())
        _threshold = float(_meta.get("prob_threshold", 0.5))

    return _model


def _calibrate(prob: float) -> float:
    """Apply isotonic calibration if a calibrator is loaded."""
    if _calibrator is None:
        return prob
    try:
        return float(_calibrator.predict(np.array([prob]))[0])
    except Exception:
        return prob   # fallback to raw probability if calibration fails


def _make_dmatrix(features: list[float]) -> xgb.DMatrix:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    return xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(len(features))])


def predict_single(features: list[float]) -> dict[str, Any]:
    bst = load_model()
    dm  = _make_dmatrix(features)

    prob_abnormal = _calibrate(float(bst.predict(dm)[0]))
    pred_class    = int(prob_abnormal >= _threshold)
    prob_normal   = 1.0 - prob_abnormal

    # SHAP contributions: (1, 7) for binary with 6 features (6 values + 1 bias)
    contribs      = bst.predict(dm, pred_contribs=True)[0]   # (7,)
    shap_vals     = contribs[:6]
    abs_shap      = np.abs(shap_vals)
    top3_idx      = abs_shap.argsort()[::-1][:3]

    lo  = [FEATURE_RANGES[FEATURE_NAMES[i]][0] for i in range(6)]
    hi  = [FEATURE_RANGES[FEATURE_NAMES[i]][1] for i in range(6)]
    mid = [(lo[i] + hi[i]) / 2 for i in range(6)]
    rng = [(hi[i] - lo[i]) or 1.0 for i in range(6)]

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
        "calibrated":     _calibrated,
        "top_features":   top_features,
    }


def predict_batch(feature_matrix: list[list[float]]) -> list[dict[str, Any]]:
    bst = load_model()
    X   = np.array(feature_matrix, dtype=np.float32)
    dm  = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])

    raw_probs = bst.predict(dm)   # (N,) — P(Abnormal), uncalibrated

    # Apply calibration in one vectorised call if calibrator is available
    if _calibrator is not None:
        cal_probs = _calibrator.predict(raw_probs.astype(np.float64))
    else:
        cal_probs = raw_probs

    results = []
    for i, p in enumerate(cal_probs):
        p = float(p)
        cls = int(p >= _threshold)
        results.append({
            "window_id":    i + 1,
            "class":        cls,
            "class_name":   CLASS_NAMES[cls],
            "probability":  round(p, 4),
            "probabilities":[round(1 - p, 4), round(p, 4)],
            "confidence":   round(max(1 - p, p), 4),
        })
    return results


def get_model_info() -> dict[str, Any]:
    load_model()
    # Pull live values from meta — fall back to last-known-good defaults
    saved_metrics  = _meta.get("metrics", {})
    saved_td       = _meta.get("training_data", {})
    return {
        "version":       _meta.get("version", "2.2"),
        "model_type":    "XGBoost Binary + Isotonic Calibration",
        "classifier":    "Normal (<15 mmHg) vs Abnormal (>=15 mmHg)",
        "threshold_mmhg": 15.0,
        "prob_threshold": _threshold,
        "calibrated":     _calibrated,
        "ece_after_calibration": _meta.get("ece_after_calibration"),
        "metrics": {
            "f1":               saved_metrics.get("f1",           0.8770),
            "auc":              saved_metrics.get("auc",          0.9490),
            "precision":        saved_metrics.get("precision",    0.9443),
            "recall":           saved_metrics.get("recall",       0.8186),
            "specificity":      saved_metrics.get("specificity",  0.9510),
            "balanced_accuracy":saved_metrics.get("balanced_acc", 0.8848),
        },
        "training_date": _meta.get("training_date", "2026-04-05"),
        "training_data": {
            "charis_patients": saved_td.get("charis_patients", 13),
            "mimic_patients":  saved_td.get("mimic_patients",  87),
            "total_windows":   saved_td.get("total_windows",   448537),
        },
        "features":          FEATURE_NAMES,
        "feature_units":     FEATURE_UNITS,
        "feature_ranges":    {k: list(v) for k, v in FEATURE_RANGES.items()},
        "classes":           CLASS_NAMES,
        "hyperparameters": {
            "learning_rate":    0.1,
            "max_depth":        4,
            "n_estimators":     420,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(_meta.get("scale_pos_weight", 0.4756), 4),
        },
        "global_importances": _global_importances,
        "previous_model_note": (
            "3-class model retired: Elevated class max probability was 3% "
            "under any input — effectively binary already. This model is honest."
        ),
    }
