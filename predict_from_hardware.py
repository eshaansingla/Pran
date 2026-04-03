"""
predict_from_hardware.py
========================
Load a trained XGBoost ICP model and predict ICP class from hardware sensor CSV.

CSV format (with or without header row):
    cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,
    cardiac_power,mean_arterial_pressure,head_angle,motion_artifact_flag,
    phase_lag_mean,phase_lag_std,phase_coherence

Each row = one 10-second window captured by the BPW34 photodiode + ESP32.

Usage:
    python predict_from_hardware.py --input hardware_data.csv
    python predict_from_hardware.py --input hardware_data.csv --model models/xgboost/xgboost_best.pkl
    python predict_from_hardware.py --input hardware_data.csv --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = Path("models/xgboost/xgboost_best.pkl")

FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
    "head_angle",
    "motion_artifact_flag",
    "phase_lag_mean",
    "phase_lag_std",
    "phase_coherence",
]
NUM_FEATURES = len(FEATURE_NAMES)

CLASS_NAMES   = ["Normal", "Elevated", "Critical"]
CLASS_COLORS  = ["\033[92m", "\033[93m", "\033[91m"]  # green, yellow, red
RESET_COLOR   = "\033[0m"
CRITICAL_WARN = "  ⚠️  ALERT — Notify medical staff immediately!"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    """Load XGBClassifier from pickle file."""
    import pickle
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'bash run_full_pipeline.sh' first to train the model.")
        sys.exit(1)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


# ── Input loading ──────────────────────────────────────────────────────────────

def load_hardware_csv(csv_path: Path) -> np.ndarray:
    """
    Load sensor CSV into a feature matrix.

    Accepts files with or without a header row. If the first row contains
    any of the known feature names, it is treated as a header. Otherwise
    the file is read as raw numbers.

    Parameters
    ----------
    csv_path : Path

    Returns
    -------
    X : np.ndarray, shape (N, 11), float32
    """
    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    # Peek at first row to detect header
    with open(csv_path, encoding="utf-8") as fh:
        first_line = fh.readline().strip()

    has_header = any(name in first_line for name in FEATURE_NAMES)
    df = pd.read_csv(csv_path, header=0 if has_header else None)

    if df.shape[1] != NUM_FEATURES:
        print(f"ERROR: Expected {NUM_FEATURES} columns, got {df.shape[1]}.")
        print(f"       Required: {', '.join(FEATURE_NAMES)}")
        sys.exit(1)

    X = df.values.astype(np.float32)

    # Sanity-check for NaN
    if np.any(np.isnan(X)):
        n_nan = int(np.isnan(X).sum())
        print(f"WARNING: {n_nan} NaN value(s) detected — imputing with column median.")
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j]

    return X


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return (predicted_classes, probabilities).

    Parameters
    ----------
    model : XGBClassifier
    X : np.ndarray, shape (N, 11)

    Returns
    -------
    classes : np.ndarray, shape (N,), int
    probs   : np.ndarray, shape (N, 3), float
    """
    classes = model.predict(X).astype(int)
    probs   = model.predict_proba(X)
    return classes, probs


# ── Output formatting ──────────────────────────────────────────────────────────

def print_results(classes: np.ndarray, probs: np.ndarray, use_color: bool = True) -> None:
    """Pretty-print prediction results to stdout."""
    n = len(classes)
    width = len(str(n))
    sep = "-" * 72

    print()
    print(sep)
    print("  ICP Classification Results (XGBoost model)")
    print(sep)
    print(f"  {'Window':<8}  {'ICP Class':<12}  {'Confidence':>10}  Probabilities (N/E/C)")
    print(sep)

    for i in range(n):
        cls   = int(classes[i])
        label = CLASS_NAMES[cls]
        conf  = probs[i, cls] * 100
        p_str = " / ".join(f"{p*100:5.1f}%" for p in probs[i])

        color = CLASS_COLORS[cls] if use_color else ""
        reset = RESET_COLOR if use_color else ""

        print(f"  {i+1:>{width}}       {color}{label:<12}{reset}  {conf:>9.1f}%  {p_str}")

        if cls == 2:
            print(f"  *** ALERT: Critical ICP -- Notify medical staff immediately! ***")

    print(sep)

    # Summary
    critical_count = int((classes == 2).sum())
    elevated_count = int((classes == 1).sum())
    normal_count   = int((classes == 0).sum())
    print(f"\n  Summary ({n} windows):")
    print(f"    Normal   : {normal_count}")
    print(f"    Elevated : {elevated_count}")
    print(f"    Critical : {critical_count}" + ("  *** ALERT ***" if critical_count > 0 else ""))
    print()


def print_json_results(classes: np.ndarray, probs: np.ndarray) -> None:
    """Print machine-readable JSON output."""
    out = []
    for i in range(len(classes)):
        cls = int(classes[i])
        out.append({
            "window": i + 1,
            "predicted_class": cls,
            "predicted_label": CLASS_NAMES[cls],
            "confidence": round(float(probs[i, cls]), 4),
            "probabilities": {
                CLASS_NAMES[c]: round(float(probs[i, c]), 4)
                for c in range(3)
            },
        })
    print(json.dumps({"predictions": out}, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict ICP class from hardware sensor CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="CSV file with 11 feature columns per row (one row = one 10-s window).",
    )
    parser.add_argument(
        "--model", "-m", type=Path, default=DEFAULT_MODEL,
        help="Path to trained XGBClassifier pickle.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (for programmatic use).",
    )
    parser.add_argument(
        "--no_color", action="store_true",
        help="Disable ANSI colour codes in terminal output.",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    X     = load_hardware_csv(args.input)
    classes, probs = predict(model, X)

    if args.json:
        print_json_results(classes, probs)
    else:
        print_results(classes, probs, use_color=not args.no_color)


if __name__ == "__main__":
    main()
