"""
predict_from_hardware.py
========================
Load the trained XGBoost v2.2 binary ICP model and predict ICP class
from hardware sensor CSV output.

CSV format (with or without header row) — 6 columns:
    cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
    slow_wave_power, cardiac_power, mean_arterial_pressure

Each row = one 10-second window captured by the BPW34 photodiode + ESP32.

Note: head_angle and motion_artifact_flag were removed in v2.2 following
ablation study that confirmed 0% gain importance and noise-level impact.

Usage:
    python predict_from_hardware.py --input hardware_data.csv
    python predict_from_hardware.py --input hardware_data.csv --json
    python predict_from_hardware.py --input data.csv --model models/xgboost_binary.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = Path("models/xgboost_binary.pkl.gz")
DEFAULT_CAL   = Path("models/xgboost_binary_calibrator.pkl.gz")

FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
]
NUM_FEATURES = len(FEATURE_NAMES)

FEATURE_RANGES = {
    "cardiac_amplitude":      (10.0, 80.0),
    "cardiac_frequency":      (0.8, 2.5),
    "respiratory_amplitude":  (2.0, 30.0),
    "slow_wave_power":        (0.05, 5.0),
    "cardiac_power":          (0.1, 5.0),
    "mean_arterial_pressure": (50.0, 150.0),
}

CLASS_NAMES  = ["Normal", "Abnormal"]
CLASS_COLORS = ["\033[92m", "\033[91m"]   # green, red
RESET_COLOR  = "\033[0m"

# Optimal threshold from v2.2 training (F1-sweep on calibrated val probs)
DEFAULT_THRESHOLD = 0.4250


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'python train_binary.py' to train the model first.")
        sys.exit(1)
    if model_path.suffix == ".gz":
        with gzip.open(model_path, "rb") as fh:
            return pickle.load(fh)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def load_calibrator(cal_path: Path):
    if not cal_path.exists():
        return None
    with gzip.open(cal_path, "rb") as fh:
        return pickle.load(fh)


# ── Input loading ──────────────────────────────────────────────────────────────

def load_hardware_csv(csv_path: Path) -> np.ndarray:
    """
    Load sensor CSV into a feature matrix (N, 6).

    Accepts files with or without a header row. If the first row contains
    any known feature name, it is treated as a header. Columns beyond the
    first 6 are silently dropped for forward-compatibility.
    """
    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as fh:
        first_line = fh.readline().strip()

    has_header = any(name in first_line for name in FEATURE_NAMES)
    df = pd.read_csv(csv_path, header=0 if has_header else None)

    # Accept 6-column files; also accept legacy 8-column files (drop cols 6-7)
    if df.shape[1] == 8:
        print("INFO: 8-column CSV detected — dropping head_angle and motion_artifact_flag (v2.2 uses 6 features).")
        df = df.iloc[:, :6]
    elif df.shape[1] < NUM_FEATURES:
        print(f"ERROR: Expected {NUM_FEATURES} columns, got {df.shape[1]}.")
        print(f"       Required: {', '.join(FEATURE_NAMES)}")
        sys.exit(1)
    elif df.shape[1] > NUM_FEATURES:
        df = df.iloc[:, :NUM_FEATURES]

    X = df.values.astype(np.float32)

    if np.any(np.isnan(X)):
        n_nan = int(np.isnan(X).sum())
        print(f"WARNING: {n_nan} NaN value(s) detected — imputing with column median.")
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j]

    # Warn on out-of-range values
    for j, name in enumerate(FEATURE_NAMES):
        lo, hi = FEATURE_RANGES[name]
        bad = np.where((X[:, j] < lo) | (X[:, j] > hi))[0]
        if len(bad):
            print(f"WARNING: {name} out of range [{lo}, {hi}] in rows: {bad[:5].tolist()}"
                  + (f" (+{len(bad)-5} more)" if len(bad) > 5 else ""))

    return X


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(bst, calibrator, X: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])
    raw_probs = bst.predict(dm)
    if calibrator is not None:
        probs = calibrator.predict(raw_probs.astype(np.float64))
    else:
        probs = raw_probs
    classes = (probs >= threshold).astype(int)
    return classes, probs


# ── Output formatting ──────────────────────────────────────────────────────────

def print_results(classes: np.ndarray, probs: np.ndarray, use_color: bool = True) -> None:
    n = len(classes)
    sep = "-" * 68
    print()
    print(sep)
    print("  ICP Classification Results — XGBoost Binary v2.2")
    print(sep)
    print(f"  {'Window':<8}  {'ICP Class':<12}  {'P(Abnormal)':>11}  {'Confidence':>10}")
    print(sep)
    for i in range(n):
        cls   = int(classes[i])
        label = CLASS_NAMES[cls]
        prob  = float(probs[i])
        conf  = max(prob, 1 - prob) * 100
        color = CLASS_COLORS[cls] if use_color else ""
        reset = RESET_COLOR if use_color else ""
        alert = "  *** ALERT: Notify medical staff! ***" if cls == 1 else ""
        print(f"  {i+1:<8}  {color}{label:<12}{reset}  {prob:>11.4f}  {conf:>9.1f}%{alert}")
    print(sep)
    n_abnormal = int((classes == 1).sum())
    n_normal   = int((classes == 0).sum())
    print(f"\n  Summary ({n} windows):")
    print(f"    Normal   : {n_normal}")
    print(f"    Abnormal : {n_abnormal}" + ("  *** ALERT ***" if n_abnormal > 0 else ""))
    print()


def print_json_results(classes: np.ndarray, probs: np.ndarray) -> None:
    out = []
    for i in range(len(classes)):
        cls = int(classes[i])
        prob = float(probs[i])
        out.append({
            "window":          i + 1,
            "class":           cls,
            "class_name":      CLASS_NAMES[cls],
            "probability":     round(prob, 4),
            "probabilities":   [round(1 - prob, 4), round(prob, 4)],
            "confidence":      round(max(prob, 1 - prob), 4),
        })
    print(json.dumps({"predictions": out, "model_version": "2.2"}, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict ICP class from hardware sensor CSV (XGBoost Binary v2.2, 6 features)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",    "-i", type=Path, required=True,
                        help="CSV file with 6 feature columns per row (one row = one 10-s window).")
    parser.add_argument("--model",    "-m", type=Path, default=DEFAULT_MODEL,
                        help="Path to trained XGBoost Booster pickle (.gz).")
    parser.add_argument("--cal",            type=Path, default=DEFAULT_CAL,
                        help="Path to isotonic calibrator pickle (.gz).")
    parser.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD,
                        help="Decision threshold for Abnormal classification.")
    parser.add_argument("--json",           action="store_true",
                        help="Output results as JSON (for programmatic use).")
    parser.add_argument("--no_color",       action="store_true",
                        help="Disable ANSI colour codes in terminal output.")
    args = parser.parse_args()

    bst        = load_model(args.model)
    calibrator = load_calibrator(args.cal)
    X          = load_hardware_csv(args.input)
    classes, probs = predict(bst, calibrator, X, args.threshold)

    if args.json:
        print_json_results(classes, probs)
    else:
        print_results(classes, probs, use_color=not args.no_color)


if __name__ == "__main__":
    main()
