"""Live terminal test: score every hardware subject in hw_data/ with the
CORAL-aligned CHARIS model and print full per-subject, per-session results.
No files written -- terminal output only.

    python charis/coral/live_test_all.py
    python charis/coral/live_test_all.py --sort flagged     # sort by flagged% desc (default: subject id)
"""
from __future__ import annotations
import argparse
import glob
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
SESS = {0: "supine", 1: "head-up-30", 2: "head-down-10", 3: "valsalva"}

ap = argparse.ArgumentParser()
ap.add_argument("--sort", choices=["id", "flagged"], default="id")
args = ap.parse_args()

model = pickle.load(open(HERE / "model.pkl", "rb"))
qt_hw = pickle.load(open(HERE / "qt_hw.pkl", "rb"))
metrics = json.loads((HERE / "metrics.json").read_text())
thr = metrics["threshold"]

files = {}
for f in sorted((ROOT / "hw_data").glob("icp_*_features.csv")):
    m = re.match(r"icp_(\d+)_(\d+)_([MF])", f.name)
    if m:
        files[int(m[1])] = (f, int(m[2]), m[3])

print("=" * 108)
print(f"  CORAL-aligned CHARIS XGBoost -- LIVE scoring of {len(files)} hardware subjects")
print(f"  threshold={thr:.3f}  |  domain-classifier residual AUC vs CHARIS={metrics['domain_classifier_auc_after_coral']:.3f} (not fully domain-blind)")
print(f"  NO ICP ground truth exists for these volunteers -- flag rates are a screening number, not a diagnosis")
print("=" * 108)

hdr = f"{'Subj':>4} {'age/sex':>8} {'windows':>8} {'flagged%':>9} {'mean P':>7} | " + \
      " ".join(f"{n:>12}" for n in SESS.values())
print(hdr)
print("-" * len(hdr))

rows = []
for s in sorted(files):
    f, age, sex = files[s]
    df = pd.read_csv(f)
    X = df[FEATURES].to_numpy(np.float64)
    sess = df["session_label"].to_numpy() if "session_label" in df.columns else np.full(len(df), -1)
    p = model.predict_proba(qt_hw.transform(X).astype(np.float32))[:, 1]
    fl = p >= thr
    r = dict(subject=s, age=age, sex=sex, windows=len(p), flagged_pct=100 * fl.mean(), mean_p=p.mean())
    for k, n in SESS.items():
        r[n] = 100 * fl[sess == k].mean() if (sess == k).any() else float("nan")
    rows.append(r)

df_all = pd.DataFrame(rows)
order = df_all.sort_values("flagged_pct", ascending=False) if args.sort == "flagged" else df_all

for _, r in order.iterrows():
    tag = "  <- age 83, prior haemorrhage" if r.subject == 4 else ""
    sess_str = " ".join(f"{r[n]:>11.1f}%" for n in SESS.values())
    print(f"{int(r.subject):>4} {f'{int(r.age)}/{r.sex}':>8} {int(r.windows):>8,} {r.flagged_pct:>8.1f}% {r.mean_p:>7.3f} | {sess_str}{tag}")

print("-" * len(hdr))
print(f"\n{'SUMMARY':^40}")
print(f"  subjects scored               : {len(df_all)}")
print(f"  flagged% -- min/median/max    : {df_all.flagged_pct.min():.1f}% / {df_all.flagged_pct.median():.1f}% / {df_all.flagged_pct.max():.1f}%")
print(f"  degenerate (0% or 100% output): {((df_all.flagged_pct==0)|(df_all.flagged_pct==100)).sum()}/{len(df_all)}")
print(f"  subjects >50% flagged (unvalidated cutoff, illustrative only): {(df_all.flagged_pct>50).sum()}/{len(df_all)}")

print(f"\n{'PER-SESSION MEAN FLAGGED%':^40}")
for n in SESS.values():
    col = n.replace("-", "_")
    print(f"  {n:>13}: {df_all[n].mean():.1f}%")

print(f"\n{'WITHIN-SUBJECT DIRECTION CHECKS':^40}")
for a_, b_, expect in [("valsalva", "supine", "expect higher"),
                        ("head-up-30", "supine", "expect LOWER"),
                        ("head-down-10", "supine", "expect higher")]:
    both = df_all.dropna(subset=[a_, b_])
    up = (both[a_] > both[b_]).sum()
    print(f"  {a_:>13} > {b_:<7}: {up}/{len(both)} subjects ({100*up/len(both):.1f}%)   [{expect}]")

print("\nNote: no subject-level 'elevated person' threshold has been validated -- only the per-window")
print("threshold (0.522) is calibrated against real CHARIS labels. Treat flagged% as a screening number.")
