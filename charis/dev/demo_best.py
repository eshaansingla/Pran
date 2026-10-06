"""Live demo: score one CHARIS patient with the best model, trained WITHOUT that patient.

    python charis/demo_best.py --patient 4                 # best model from models/charis_best/metrics.json
    python charis/demo_best.py --patient 6 --model XGBoost # force a model
    python charis/demo_best.py --table                     # just print the saved ranking of all models

Uses the same fold recipe as charis/compare_models.py (scaler fit on the fit patients, threshold from 2 validation
patients), so the numbers match models/charis_compare/<Model>/per_patient.csv. Writes results/demo_patient<N>_<Model>.png.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import compare_models as C

ap = argparse.ArgumentParser()
ap.add_argument("--patient", type=int, default=6)
ap.add_argument("--model", default=None)
ap.add_argument("--table", action="store_true")
a = ap.parse_args()

if a.table:
    print(pd.read_csv(C.ROOT / "summary.csv").round(3).to_string(index=False)); sys.exit()

name = a.model or json.loads((C.BEST / "metrics.json").read_text())["model"]
X, y, pid = (np.load(C.CACHE / f"{n}.npy") for n in ("X", "y", "pid")); pats = sorted(np.unique(pid))
if a.patient not in pats: sys.exit(f"patient must be one of {[int(p) for p in pats]}")
i = pats.index(a.patient); others = [q for q in pats if q != a.patient]
te = pid == a.patient; vm = np.isin(pid, [others[(i * 2) % 12], others[(i * 2 + 1) % 12]]); fm = ~te & ~vm
print(f"Model: {name} | test patient {a.patient} ({te.sum():,} ten-second windows, {100 * y[te].mean():.1f}% truly elevated)")
print(f"Training on {fm.sum():,} windows from 10 other patients (+2 validation patients); patient {a.patient} is never seen ...")
qt = QuantileTransformer(output_distribution="normal", random_state=C.SEED, n_quantiles=1000, subsample=200_000).fit(X[fm])
Xf, Xv, Xt = (qt.transform(X[m]).astype(np.float32) for m in (fm, vm, te)); yf, yv, yt = y[fm], y[vm], y[te]
if name in C.BOOST:
    m, _ = C.fit_boost(name, Xf, yf, Xv, yv); Xv_, yv_ = Xv, yv
else:
    sub = np.random.RandomState(C.SEED + i).permutation(len(yf))[:C.CAP]; vs = np.random.RandomState(i).permutation(len(yv))[:100_000]
    m = C.SK[name]().fit(Xf[sub], yf[sub]); Xv_, yv_ = Xv[vs], yv[vs]
thr = C.youden(yv_, C.score(m, Xv_)); s = C.score(m, Xt); pred = (s >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
print(f"\nThreshold (chosen on validation patients): {thr:.3f}")
print(f"AUC {roc_auc_score(yt, s):.3f} | avg precision {average_precision_score(yt, s):.3f} | sensitivity {tp / max(tp + fn, 1):.3f} | "
      f"specificity {tn / max(tn + fp, 1):.3f} | precision {tp / max(tp + fp, 1):.3f}")
print(f"Correctly flagged elevated: {tp:,} | missed: {fn:,} | false alarms: {fp:,} | correctly cleared: {tn:,}")
print("Mean predicted score: elevated windows %.3f vs normal windows %.3f" % (s[yt == 1].mean() if (yt == 1).any() else float("nan"), s[yt == 0].mean()))

k = max(len(s) // 600, 1); t = np.arange(len(s) // k) * k * 5 / 60   # windows step 5 s; block-average for a readable plot
sm = s[:len(t) * k].reshape(-1, k).mean(1); tr = yt[:len(t) * k].reshape(-1, k).mean(1)
fig, ax = plt.subplots(figsize=(12, 4.2)); ax.fill_between(t, 0, tr, color="#eb6834", alpha=0.25, step="mid", label="truly elevated (probe, ICP >= 20 mmHg)")
ax.plot(t, sm, color="#2a78d6", lw=1.6, label="model score"); ax.axhline(thr, color="#52514e", ls="--", lw=1, label=f"threshold {thr:.2f}")
ax.set_xlabel("minutes into recording"); ax.set_ylabel("score / fraction elevated"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
ax.set_title(f"{name} on patient {a.patient} (model never saw this patient): AUC {roc_auc_score(yt, s):.3f}", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); Path("results").mkdir(exist_ok=True)
out = Path(f"results/demo_patient{a.patient}_{name}.png"); fig.savefig(out, dpi=140); print(f"\nFigure: {out}")
