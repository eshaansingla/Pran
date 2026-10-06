"""Score randomly chosen hardware volunteers with the CORAL-aligned CHARIS model
(charis/coral/model.pkl) -- domain-adapted counterpart of charis/hw_random_test.py.

    python charis/coral/hw_random_test.py                 # 10 subjects: Subject 4 + 9 random others
    python charis/coral/hw_random_test.py --seed 7         # repeat a previous draw
    python charis/coral/hw_random_test.py --n 10 --subjects 4 12 111

Model: charis/coral/model.pkl (CHARIS-trained, CORAL-aligned onto the hardware
background's covariance -- see charis/coral/README.md). Reads precomputed
per-window features from hw_data/*.csv (extracted by
support/code/extract_hw_features_csv.py with the exact same formula as the
model). There is NO ICP ground truth for hardware volunteers: the only labels
are the recorded manoeuvres (session_label).
"""
from __future__ import annotations
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
ALWAYS = 4  # 83-year-old, prior haemorrhage -- always included, same convention as the original script

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--subjects", type=int, nargs="*", default=[])
a = ap.parse_args()
seed = a.seed if a.seed is not None else int(time.time()) % 100000
rng = np.random.RandomState(seed)

files = {}
for f in (ROOT / "hw_data").glob("icp_*_features.csv"):
    m = re.match(r"icp_(\d+)_(\d+)_([MF])", f.name)
    if m:
        files[int(m[1])] = (f, int(m[2]), m[3])

must = list(dict.fromkeys([ALWAYS] + a.subjects))
assert all(s in files for s in must), "subject file not found in hw_data/"
pool = [s for s in sorted(files) if s not in must]
pick = must + list(rng.choice(pool, max(a.n - len(must), 0), replace=False))
print(f"Random draw seed {seed} (repeat with --seed {seed}). Subjects: {sorted(int(s) for s in pick)}  "
      f"(Subject {ALWAYS}, age {files[ALWAYS][1]}, is always included)")

model = pickle.load(open(HERE / "model.pkl", "rb"))
qt_hw = pickle.load(open(HERE / "qt_hw.pkl", "rb"))
metrics = json.loads((HERE / "metrics.json").read_text())
thr = metrics["threshold"]
print(f"Model: CORAL-aligned CHARIS XGBoost, threshold {thr:.3f}. Hardware data was never used as training "
      f"rows/labels -- only as an unsupervised covariance-alignment target (see README).\n")

names = {0: "supine", 1: "head-up", 2: "head-down", 3: "Valsalva"}
print(f"{'Subj':>4} {'age/sex':>8} {'windows':>8} {'flagged%':>9} {'mean P':>7} | "
      + " ".join(f"{n:>13}" for n in names.values()) + "  (flagged% per session)")

rows = []
for s in sorted(int(x) for x in pick):
    f, age, sex = files[s]
    df = pd.read_csv(f)
    X = df[FEATURES].to_numpy(np.float64)
    sess = df["session_label"].to_numpy() if "session_label" in df.columns else np.full(len(df), -1)
    if len(X) == 0:
        print(f"{s:>4}  no usable windows")
        continue
    p = model.predict_proba(qt_hw.transform(X).astype(np.float32))[:, 1]
    fl = p >= thr
    r = dict(subject=s, age=age, sex=sex, windows=len(p), flagged_pct=100 * fl.mean(), mean_p=p.mean())
    for k, n in names.items():
        r[n] = 100 * fl[sess == k].mean() if (sess == k).any() else np.nan
    rows.append(r)
    print(f"{s:>4} {f'{age}/{sex}':>8} {len(p):>8,} {r['flagged_pct']:>8.1f}% {p.mean():>7.3f} | "
          + " ".join(f"{r[n]:>12.1f}%" for n in names.values())
          + ("   <- age 83, prior haemorrhage" if s == ALWAYS else ""))

df_out = pd.DataFrame(rows)
(ROOT / "charis" / "coral" / "results").mkdir(exist_ok=True, parents=True)
df_out.to_csv(HERE / "results" / "hw_random_test.csv", index=False)
up = (df_out["Valsalva"] > df_out["supine"]).sum()
print(f"\nValsalva flagged more than supine in {up}/{len(df_out)} of these subjects.")
print("How to read this: the model is CHARIS-trained (brain-injury patients with an invasive probe); there is")
print("no ICP measurement for these volunteers, so flag rates are not accuracy against ground truth.")
print(f"Known limits (see metrics.json): a domain classifier can still tell CHARIS windows from hardware windows")
print(f"at AUC~{metrics['domain_classifier_auc_after_coral']:.3f} even after CORAL alignment. On the full 146-subject set,")
print("head-up scores HIGHER than supine (should be lower) and head-down is flat vs supine (should be higher) --")
print("likely the PPG-derived cardiac features picking up postural cardiovascular change, not pure ICP signal.")
print("Only Valsalva-vs-supine direction has been independently validated end-to-end (hardware-vasalva-method/).")

cols = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]
fig, ax = plt.subplots(figsize=(12, 5))
w = 0.2
x = np.arange(len(df_out))
for j, (n, c) in enumerate(zip(names.values(), cols)):
    ax.bar(x + (j - 1.5) * w, df_out[n].fillna(0), w, color=c, label=n)
ax.set_xticks(x)
ax.set_xticklabels([f"S{r.subject}\n{r.age}{r.sex}" for r in df_out.itertuples()])
ax.set_ylabel("windows flagged (%)")
ax.set_title(f"CORAL-aligned CHARIS XGBoost on {len(df_out)} hardware volunteers (seed {seed}); "
             f"S{ALWAYS} = 83-year-old with prior haemorrhage", loc="left", fontweight="bold")
ax.axhline(0, color="#e6e5e1")
ax.grid(axis="y", color="#e6e5e1")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.text(0.01, 0.005, "No ICP ground truth exists for these volunteers; sessions are the recorded manoeuvres. "
                      "Domain classifier AUC vs CHARIS after alignment: ~0.82-0.87 (not fully domain-blind).",
         fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "results" / "hw_random_test.png", dpi=140)
print(f"\nFigure: {HERE / 'results' / 'hw_random_test.png'}\nTable: {HERE / 'results' / 'hw_random_test.csv'}")
