"""
lstm_predictor.py
=================
LSTM ICP forecasting inference — loaded lazily by main.py.

Expects:
    models/lstm_forecast_v1.h5   — trained Keras model
    models/lstm_attn_v1.h5       — attention sub-model
    models/lstm_meta.json        — metadata + scaler params

Provides:
    lstm_available()             — True if models are loaded
    predict_forecast(sequence)  — returns ForecastResult dict
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_model       = None
_attn_model  = None
_meta: dict  = {}
_loaded      = False
_load_error: str | None = None

MC_PASSES    = 30   # Monte Carlo dropout passes for uncertainty estimate (v4.2)


# ─── Model paths ──────────────────────────────────────────────────────────────

def _model_dir() -> Path:
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env).parent
    return Path(__file__).parent.parent.parent / "models"


def _try_load() -> None:
    global _model, _attn_model, _meta, _loaded, _load_error

    model_dir  = _model_dir()
    model_path = model_dir / "lstm_forecast_v1.h5"
    attn_path  = model_dir / "lstm_attn_v1.h5"
    meta_path  = model_dir / "lstm_meta.json"

    if not model_path.exists():
        _load_error = (
            f"LSTM model not found at {model_path}. "
            "Run  python src/models/lstm_forecaster.py  to train it first."
        )
        return

    try:
        import tensorflow as tf   # noqa: F401 — deferred import
        from tensorflow.keras.models import load_model

        _model      = load_model(str(model_path), compile=False)
        if attn_path.exists():
            _attn_model = load_model(str(attn_path), compile=False)
        if meta_path.exists():
            raw = meta_path.read_text()
            # json.loads handles NaN-free files only — guard against corrupt meta
            _meta = json.loads(raw)
        _loaded     = True
        print(f"[lstm_predictor] Loaded LSTM forecaster v{_meta.get('version','?')} "
              f"from {model_path}")
    except ImportError:
        _load_error = (
            "TensorFlow is not installed. "
            "Install it with:  pip install tensorflow>=2.12"
        )
    except Exception as exc:
        _load_error = f"Failed to load LSTM model: {exc}"


def lstm_available() -> bool:
    return _loaded


def get_load_error() -> str | None:
    return _load_error


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _scale(sequence: np.ndarray) -> np.ndarray:
    """Standardise using training-set statistics stored in lstm_meta.json."""
    mean_ = np.array(_meta.get("scaler_mean", [0.0] * 6), dtype=np.float64)
    std_  = np.array(_meta.get("scaler_std",  [1.0] * 6), dtype=np.float64)
    std_[std_ < 1e-8] = 1.0
    return ((sequence - mean_) / std_).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _attention_weights(lstm_out: np.ndarray) -> np.ndarray:
    """
    Derive per-timestep attention weights from BiLSTM hidden states.
    Uses L2 norm of hidden state at each step as an activity proxy,
    then softmax-normalises to produce weights that sum to 1.
    """
    norms = np.linalg.norm(lstm_out, axis=-1)   # (seq_len,)
    return _softmax(norms)                        # (seq_len,)


def _confidence_label(prob: float, std: float) -> str:
    if std > 0.12:
        return "Low"
    if prob > 0.8 or prob < 0.2:
        return "High"
    return "Medium"


def _interpretation(attn: np.ndarray, feat_vals_scaled: np.ndarray,
                    feature_names: list[str], pred_class: int) -> str:
    """
    Generate a plain-English explanation.
    Identifies the most attended timestep and the most active feature there.
    """
    top_t     = int(np.argmax(attn))           # most attended timestep (0-indexed)
    feat_row  = feat_vals_scaled[top_t]        # scaled feature values at that step
    top_f_idx = int(np.argmax(np.abs(feat_row)))
    feat_name = feature_names[top_f_idx].replace("_", " ")
    direction = "elevated" if feat_row[top_f_idx] > 0 else "reduced"
    minutes_ago = round((len(attn) - top_t) * 10 / 60, 1)

    if pred_class == 1:
        return (
            f"Forecast driven by {direction} {feat_name} "
            f"{minutes_ago} min ago (window t-{len(attn) - top_t}). "
            "Rising trend suggests possible ICP elevation ahead."
        )
    return (
        f"Stable {feat_name} noted {minutes_ago} min ago. "
        "Current trajectory suggests ICP likely to remain within normal range."
    )


# ─── Public predict function ──────────────────────────────────────────────────

def predict_forecast(sequence: list[list[float]]) -> dict[str, Any]:
    """
    Run LSTM inference on a 30-window sequence.

    Parameters
    ----------
    sequence : list of 30 rows, each row = [6 feature values]
               (backend trims/pads to exactly SEQ_LEN before calling this)

    Returns
    -------
    dict with forecast class, probability, confidence, attention weights, etc.
    """
    import tensorflow as tf

    seq_len      = _meta.get("seq_len",         30)
    horizon_min  = _meta.get("horizon_minutes", 15)
    feature_names = _meta.get("feature_names", [
        "cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
        "slow_wave_power",   "cardiac_power",     "mean_arterial_pressure",
    ])
    threshold    = float(_meta.get("threshold", 0.5))

    # Build (1, seq_len, 6) tensor
    arr_raw    = np.array(sequence[-seq_len:], dtype=np.float32)  # (30, 6)
    arr_scaled = _scale(arr_raw)
    x_batch    = arr_scaled[np.newaxis, ...]                      # (1, 30, 6)

    # MC Dropout: run MC_PASSES times with training=True (keeps dropout active)
    mc_preds = np.array([
        _model(x_batch, training=True).numpy()[0]
        for _ in range(MC_PASSES)
    ])
    # mc_preds shape: (MC_PASSES, horizon_min) — native from model output

    # Shape guard: meta and model must agree; warn and clip if legacy mismatch
    actual_outputs = mc_preds.shape[1]
    if actual_outputs != horizon_min:
        print(
            f"[lstm_predictor] WARNING: model outputs {actual_outputs} horizons "
            f"but meta says {horizon_min}. Clipping to {min(actual_outputs, horizon_min)}. "
            "Retrain with lstm_forecaster.py v4.1 to eliminate this warning."
        )
        horizon_min = min(actual_outputs, horizon_min)
        mc_preds    = mc_preds[:, :horizon_min]

    probs_30 = mc_preds.mean(axis=0)  # (horizon_min,)
    std_30   = mc_preds.std(axis=0)   # (horizon_min,)

    # Overarching classification: use the peak probability across the 30 horizons.
    # Peak (not last) because the worst-case within the forecast window is the
    # clinically relevant early-warning signal — a transient elevation at t=15 min
    # followed by recovery still warrants an Abnormal alert.
    peak_idx = int(np.argmax(probs_30))
    prob = float(probs_30[peak_idx])
    std  = float(std_30[peak_idx])

    pred_class  = int(prob >= threshold)
    class_names = ["Normal", "Abnormal"]

    # CI bounds per horizon — raw from MC dropout, no floor manipulation
    ci_lower_30 = np.clip(probs_30 - 1.96 * std_30, 0.0, 1.0)
    ci_upper_30 = np.clip(probs_30 + 1.96 * std_30, 0.0, 1.0)

    # Scalar CI at the peak horizon (for the summary card)
    ci_lower = float(ci_lower_30[peak_idx])
    ci_upper = float(ci_upper_30[peak_idx])

    # Attention weights
    if _attn_model is not None:
        lstm_out   = _attn_model(x_batch, training=False).numpy()[0]  # (30, 128)
        attn_w     = _attention_weights(lstm_out)                      # (30,)
    else:
        # Fallback: uniform weights
        attn_w = np.ones(seq_len, dtype=np.float32) / seq_len

    # Top-3 most attended timesteps
    top3_idx = attn_w.argsort()[::-1][:3]
    attention_highlights = [
        {
            "timestep":   int(-(seq_len - int(i))),   # negative = minutes ago
            "importance": round(float(attn_w[i]), 4),
        }
        for i in top3_idx
    ]

    # Per-feature importance at top-attended timestep
    top_t        = int(np.argmax(attn_w))
    feat_at_top  = arr_scaled[top_t]                 # (6,)
    abs_vals     = np.abs(feat_at_top)
    feat_total   = abs_vals.sum() or 1.0
    feature_highlights = [
        {
            "name":       feature_names[i],
            "importance": round(float(abs_vals[i] / feat_total), 4),
        }
        for i in abs_vals.argsort()[::-1][:3]
    ]

    interpretation = _interpretation(
        attn_w, arr_scaled, feature_names, pred_class
    )

    return {
        "class":              pred_class,
        "class_name":         class_names[pred_class],
        "probability":        round(prob, 4),
        "probabilities":      [round(1.0 - prob, 4), round(prob, 4)],
        "confidence_label":   _confidence_label(prob, std),
        "ci_lower":           round(ci_lower, 4),
        "ci_upper":           round(ci_upper, 4),
        "std":                round(std, 4),
        "horizon_minutes":    horizon_min,
        "interpretation":     interpretation,
        "attention_weights":  [round(float(w), 5) for w in attn_w],   # (30,)
        "attention_highlights": attention_highlights,
        "feature_highlights": feature_highlights,
        "model_version":      _meta.get("version", "2.0"),
        "seq_len":            seq_len,
        "threshold":          round(threshold, 4),
        "forecast_probabilities": [round(float(p), 4) for p in probs_30],
        "forecast_ci_lower":      [round(float(p), 4) for p in ci_lower_30],
        "forecast_ci_upper":      [round(float(p), 4) for p in ci_upper_30],
    }


# ─── Initialise on import ─────────────────────────────────────────────────────

_try_load()
