"""Score randomly chosen hardware volunteers with the CHARIS-trained XGBoost model. The 83-year-old case (Subject 4,
prior haemorrhage) is always included.

    python charis/hw_random_test.py                 # 10 subjects: Subject 4 + 9 random others (new random draw each run)
    python charis/hw_random_test.py --seed 7        # repeat a previous draw
    python charis/hw_random_test.py --n 10 --subjects 4 12 111   # force particular subjects (rest random)

Model: models/charis_compare/XGBoost (trained on all 13 CHARIS patients, never on hardware data), scaler and threshold
from the same folder. Prints per-subject and per-session results and writes results/hw_random_test.png / .csv.
There is NO ICP ground truth for hardware volunteers: the only labels are the recorded manoeuvres.
"""
from __future__ import annotations
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb  # noqa: F401  (pickled model needs it importable)

import full_pipeline_qt as F

ALWAYS = 4
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--subjects", type=int, nargs="*", default=[]); a = ap.parse_args()
seed = a.seed if a.seed is not None else int(time.time()) % 100000; rng = np.random.RandomState(seed)

files = {}
for f in Path("hw-tests").glob("icp_*.csv"):
    m = re.match(r"icp_(\d+)_(\d+)_([MF])", f.name)
    if m: files[int(m[1])] = (f, int(m[2]), m[3])
must = list(dict.fromkeys([ALWAYS] + a.subjects)); assert all(s in files for s in must), "subject file not found"
pool = [s for s in sorted(files) if s not in must]; pick = must + list(rng.choice(pool, max(a.n - len(must), 0), replace=False))
print(f"Random draw seed {seed} (repeat with --seed {seed}). Subjects: {sorted(int(s) for s in pick)}  (Subject {ALWAYS}, age {files[ALWAYS][1]}, is always included)")

d = Path("models/charis_compare/XGBoost"); model = pickle.load(open(d / "model.pkl", "rb")); qt = pickle.load(open(d / "qt_scaler.pkl", "rb"))
thr = json.loads((d / "metrics.json").read_text())["threshold"]
print(f"Model: CHARIS-trained XGBoost, threshold {thr:.3f}. Hardware data was never used for training.\n")
names = {0: "supine", 1: "head-up 30", 2: "head-down 10", 3: "Valsalva"}
print(f"{'Subj':>4} {'age/sex':>8} {'windows':>8} {'flagged%':>9} {'mean P':>7} | " + " ".join(f"{n:>13}" for n in names.values()) + "  (flagged% per session)")
rows = []
for s in sorted(int(x) for x in pick):
    f, age, sex = files[s]; X, sess = F.load_hw_csv(f)
    if len(X) == 0: print(f"{s:>4}  no usable windows"); continue
    p = model.predict_proba(qt.transform(X).astype(np.float32))[:, 1]; fl = p >= thr
    r = dict(subject=s, age=age, sex=sex, windows=len(p), flagged_pct=100 * fl.mean(), mean_p=p.mean())
    for k, n in names.items(): r[n] = 100 * fl[sess == k].mean() if (sess == k).any() else np.nan
    rows.append(r)
    print(f"{s:>4} {f'{age}/{sex}':>8} {len(p):>8,} {r['flagged_pct']:>8.1f}% {p.mean():>7.3f} | " + " ".join(f"{r[n]:>12.1f}%" for n in names.values())
          + ("   <- age 83, prior haemorrhage" if s == ALWAYS else ""))
df = pd.DataFrame(rows); Path("results").mkdir(exist_ok=True); df.to_csv("results/hw_random_test.csv", index=False)
up = (df["Valsalva"] > df["supine"]).sum()
print(f"\nValsalva flagged more than supine in {up}/{len(df)} of these subjects.")
print("How to read this: the model was trained on brain-injury patients with a probe; there is no ICP measurement for these volunteers, so flag rates are not accuracy.")
print("Known limits: on all 146 volunteers this model separates Valsalva from supine only weakly (AUC about 0.65) and flags older people more (39% of windows over 65).")

cols = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]; fig, ax = plt.subplots(figsize=(12, 5)); w = 0.2; x = np.arange(len(df))
for j, (n, c) in enumerate(zip(names.values(), cols)): ax.bar(x + (j - 1.5) * w, df[n].fillna(0), w, color=c, label=n)
ax.set_xticks(x); ax.set_xticklabels([f"S{r.subject}\n{r.age}{r.sex}" for r in df.itertuples()]); ax.set_ylabel("windows flagged (%)")
ax.set_title(f"CHARIS XGBoost on {len(df)} hardware volunteers (seed {seed}); S{ALWAYS} = 83-year-old with prior haemorrhage", loc="left", fontweight="bold")
ax.axhline(0, color="#e6e5e1"); ax.grid(axis="y", color="#e6e5e1"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16)); fig.text(0.01, 0.005, "No ICP ground truth exists for these volunteers; sessions are the recorded manoeuvres.", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("results/hw_random_test.png", dpi=140); print("Figure: results/hw_random_test.png  Table: results/hw_random_test.csv")
