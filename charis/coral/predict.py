"""Score hardware feature CSVs with the trained CORAL-aligned CHARIS model.
Loads the artifacts saved by train.py (same idea as charis/web_app.py loading
models/charis_compare/XGBoost, just domain-adapted).

Usage:
  python charis/coral/predict.py <one_or_more_csv_or_glob>
  python charis/coral/predict.py hw_data/*.csv
  python charis/coral/predict.py C:\\Users\\asus\\Downloads\\tes
"""
from __future__ import annotations
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]

MODEL = pickle.load(open(HERE / "model.pkl", "rb"))
QT_HW = pickle.load(open(HERE / "qt_hw.pkl", "rb"))
METRICS = json.loads((HERE / "metrics.json").read_text())
THRESHOLD = METRICS["threshold"]


def score_csv(path: str):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  [skip] {path}: missing columns {missing}")
        return None
    X = df[FEATURES].to_numpy(np.float64)
    # hardware data is the CORAL target domain: normalize with its own domain QT only,
    # no further alignment needed (CORAL only warps the CHARIS source at train time)
    Xq = QT_HW.transform(X)
    p = MODEL.predict_proba(Xq.astype(np.float32))[:, 1]
    df = df.copy()
    df["p_elevated"] = np.round(p, 4)
    df["flagged"] = (p >= THRESHOLD).astype(int)
    return df


def main(paths: list[str]):
    files = []
    for p in paths:
        matched = glob.glob(p)
        files.extend(matched if matched else [p])
    if not files:
        sys.exit("no files matched")

    print(f"CORAL CHARIS model | threshold={THRESHOLD:.3f} | "
          f"domain-classifier residual AUC={METRICS['domain_classifier_auc_after_coral']} "
          f"(see metrics.json for caveats)\n")
    print(f"{'file':>32} {'windows':>8} {'flagged%':>9} {'mean P':>7}")
    for f in sorted(files):
        df = score_csv(f)
        if df is None:
            continue
        name = Path(f).name
        print(f"{name:>32} {len(df):>8,} {100*df['flagged'].mean():>8.1f}% {df['p_elevated'].mean():>7.3f}")
        out = Path(f).with_name(Path(f).stem + "_coral_scored.csv")
        df.to_csv(out, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python charis/coral/predict.py <csv_or_glob> [...]")
    main(sys.argv[1:])
