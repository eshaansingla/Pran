"""CHARIS window-by-window classifier: upload a CSV, see every 10-second window scored by the trained XGBoost model.

    python charis/web_app.py            # then open http://127.0.0.1:5001  (opens automatically)
    python charis/web_app.py --port 8000 --no-browser

Accepted CSVs
  1. Feature CSV: one row per 10 s window with the columns cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
     slow_wave_power, cardiac_power. Optional: patient_id, label_elevated (adds accuracy metrics).
     This is the format of results/charis_patient1_all_rows.csv.
  2. Raw hardware CSV: columns ir_raw, disp_raw (and optionally artifact_flag), 50 Hz. It is cut into 10 s windows
     (no overlap) and turned into the same five features before scoring.

Model: models/charis_compare/XGBoost (trained on all 13 CHARIS patients), with its scaler and Youden threshold.
"""
from __future__ import annotations
import argparse
import io
import json
import pickle
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import xgboost as xgb  # noqa: F401  (pickled model needs it importable)
from flask import Flask, jsonify, request, send_from_directory

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
MODEL_DIR = ROOT / "models" / "charis_compare" / "XGBoost"
LOPO_DIR = ROOT / "results" / "charis_lopo"
SAMPLE = ROOT / "results" / "charis_patient1_all_rows.csv"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

if not (MODEL_DIR / "model.pkl").exists():
    sys.exit(f"Trained model not found in {MODEL_DIR}. Run `python charis/compare_models.py` first.")
MODEL = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
QT = pickle.load(open(MODEL_DIR / "qt_scaler.pkl", "rb"))
INFO = json.loads((MODEL_DIR / "metrics.json").read_text())
THRESHOLD = float(INFO["threshold"])


def windows_from_raw(df: pd.DataFrame):
    """Raw 50 Hz hardware recording -> features using the same extractor as the training pipeline."""
    import full_pipeline_qt as F
    if "artifact_flag" in df.columns:
        df = df[df["artifact_flag"] == 0]
    ir, disp = df["ir_raw"].to_numpy(np.float32), df["disp_raw"].to_numpy(np.float32)
    feats = []
    for w in range(len(df) // F.WIN):          # consecutive, non-overlapping 10 s windows
        s = w * F.WIN
        f = F.extract_hw_window(ir[s:s + F.WIN], disp[s:s + F.WIN])
        if f is not None:
            feats.append(f)
    return np.array(feats, dtype=np.float32).reshape(-1, len(FEATURES)), None


def parse(df: pd.DataFrame):
    df.columns = [str(c).strip().lower() for c in df.columns]
    if all(c in df.columns for c in FEATURES):
        d = df.dropna(subset=FEATURES).reset_index(drop=True)
        if "minutes_from_start" in d and len(d) > 2 and np.median(np.diff(d["minutes_from_start"].to_numpy(float))) * 60 < 7:
            d = d.iloc[::2].reset_index(drop=True)   # stored with a ~5 s hop: every 2nd row = consecutive non-overlapping 10 s windows
        X = d[FEATURES].to_numpy(np.float32)
        lab = d["label_elevated"].to_numpy(int) if "label_elevated" in d else None
        return X, lab, "feature windows"
    if {"ir_raw", "disp_raw"} <= set(df.columns):
        X, lab = windows_from_raw(df)
        return X, lab, "raw recording, windowed here"
    raise ValueError("CSV needs either the five feature columns (" + ", ".join(FEATURES) + ") or raw columns ir_raw and disp_raw.")


def classify(df: pd.DataFrame, name: str):
    X, lab, kind = parse(df)
    if len(X) == 0:
        raise ValueError("No usable windows found (too short, flat signal, or all rows flagged as artifacts).")
    p = MODEL.predict_proba(QT.transform(X).astype(np.float32))[:, 1]
    out = dict(
        name=name, kind=kind, n=len(p), threshold=THRESHOLD,
        p=np.round(p, 4).tolist(),
        features={f: np.round(X[:, i].astype(float), 5).tolist() for i, f in enumerate(FEATURES)},
        labels=lab.tolist() if lab is not None else None,
        model=dict(name="XGBoost", auc_mean=INFO["auc_mean"], sensitivity=INFO["sensitivity"],
                   specificity=INFO["specificity"], worst_patient_auc=INFO["auc_worst_patient"]),
    )
    return out


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent / "web", "index.html")


@app.post("/api/classify")
def api_classify():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file uploaded."), 400
    try:
        df = pd.read_csv(io.BytesIO(f.read()), comment="#", low_memory=False)
        return jsonify(classify(df, f.filename))
    except Exception as e:  # surface parse problems to the UI instead of a 500 page
        return jsonify(error=str(e)), 400


@app.get("/api/sample")
def api_sample():
    if not SAMPLE.exists():
        return jsonify(error=f"Sample file {SAMPLE.name} not found."), 404
    return jsonify(classify(pd.read_csv(SAMPLE), SAMPLE.name))


@app.get("/api/lopo")
def api_lopo():
    f = LOPO_DIR / "lopo_windows.csv"
    if not f.exists():
        return jsonify(error="Run `python charis/lopo_windows_report.py` first."), 404
    return jsonify(pd.read_csv(f).to_dict("records"))


@app.get("/lopo/<path:name>")
def lopo_image(name):
    return send_from_directory(LOPO_DIR, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=5001); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(); url = f"http://127.0.0.1:{a.port}"
    print(f"CHARIS classifier (XGBoost, threshold {THRESHOLD:.3f}) -> {url}")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=a.port, debug=False)
