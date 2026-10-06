"""Live XGBoost pipeline on CHARIS (for presenting): leave-one-patient-out, one fold at a time, printed as it runs.

    python charis/run_xgb_pipeline.py                  # all 13 patients (about 4 to 5 minutes)
    python charis/run_xgb_pipeline.py --patients 4 6 9 # only some folds (quick demo)

For each test patient: the scaler and the model are trained WITHOUT that patient (2 other patients pick the
early-stopping round and the decision threshold), then the unseen patient is scored.
Writes results/xgb_live/ (per_patient.csv, summary.json, results.png). Same recipe and seed as charis/compare_models.py.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import compare_models as C
import plot_compare as P

ap = argparse.ArgumentParser(); ap.add_argument("--patients", type=int, nargs="*"); a = ap.parse_args()
X, y, pid = (np.load(C.CACHE / f"{n}.npy") for n in ("X", "y", "pid")); pats = [int(p) for p in sorted(np.unique(pid))]
todo = a.patients or pats
out = Path("results/xgb_live"); out.mkdir(parents=True, exist_ok=True)

print("=" * 96)
print("XGBoost on CHARIS: leave-one-patient-out")
print(f"Data: {len(y):,} ten-second windows, {len(pats)} patients, 5 waveform features. Label: ICP >= 20 mmHg (from the invasive probe).")
print("Each row below is a patient the model has NEVER seen in training.")
print("=" * 96)
print(f"{'Patient':>7} {'windows':>8} {'%elev':>6} {'AUC':>6} {'sens':>6} {'spec':>6} {'prec':>6} {'F1':>6} {'flagged/missed/false alarm':>30} {'sec':>5}")
rows = []
for p in todo:
    t0 = time.time(); i = pats.index(p); others = [q for q in pats if q != p]
    te = pid == p; vm = np.isin(pid, [others[(i * 2) % 12], others[(i * 2 + 1) % 12]]); fm = ~te & ~vm
    qt = QuantileTransformer(output_distribution="normal", random_state=C.SEED, n_quantiles=1000, subsample=200_000).fit(X[fm])
    Xf, Xv, Xt = (qt.transform(X[m]).astype(np.float32) for m in (fm, vm, te)); yf, yv, yt = y[fm], y[vm], y[te]
    m, n_it = C.fit_boost("XGBoost", Xf, yf, Xv, yv)
    thr = C.youden(yv, C.score(m, Xv)); s = C.score(m, Xt)
    tn, fp, fn, tp = confusion_matrix(yt, (s >= thr).astype(int), labels=[0, 1]).ravel()
    se, sp, pc = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1); f1 = 2 * pc * se / max(pc + se, 1e-9)
    auc = roc_auc_score(yt, s)
    rows.append(dict(model="XGBoost", patient=p, n_windows=int(te.sum()), abnormal_pct=100 * yt.mean(), auc=auc, avg_precision=average_precision_score(yt, s),
                     sensitivity=se, specificity=sp, precision=pc, f1=f1, threshold=thr, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
    print(f"{p:>7} {te.sum():>8,} {100 * yt.mean():>6.1f} {auc:>6.3f} {se:>6.3f} {sp:>6.3f} {pc:>6.3f} {f1:>6.3f} {f'{tp:,} / {fn:,} / {fp:,}':>30} {time.time() - t0:>5.0f}", flush=True)

d = pd.DataFrame(rows); d.to_csv(out / "per_patient.csv", index=False); a_ = d.auc.values
rng = np.random.RandomState(0); bs = [a_[rng.randint(0, len(a_), len(a_))].mean() for _ in range(5000)]
summ = dict(model="XGBoost", n_patients=len(d), auc_mean=a_.mean(), auc_std=a_.std(ddof=1) if len(a_) > 1 else 0.0, auc_ci_lo=np.percentile(bs, 2.5), auc_ci_hi=np.percentile(bs, 97.5),
            auc_worst_patient=a_.min(), patients_auc_ge_0_90=int((a_ >= .9).sum()), avg_precision=d.avg_precision.mean(), sensitivity=d.sensitivity.mean(),
            specificity=d.specificity.mean(), precision=d.precision.mean(), f1=d.f1.mean())
(out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print("-" * 96)
print(f"MEAN over {len(d)} patients: AUC {summ['auc_mean']:.3f} (95% CI {summ['auc_ci_lo']:.3f} to {summ['auc_ci_hi']:.3f}) | sensitivity {summ['sensitivity']:.3f} | "
      f"specificity {summ['specificity']:.3f} | precision {summ['precision']:.3f} | F1 {summ['f1']:.3f}")
print(f"Patients with AUC >= 0.90: {summ['patients_auc_ge_0_90']}/{len(d)}   worst patient AUC: {summ['auc_worst_patient']:.3f}")
if len(d) > 1:
    P.model_figure("XGBoost", d.sort_values("patient"), pd.Series(summ), out); print(f"Figure: {out / 'results.png'}")
print("Note: features still contain mean-level pressure information; with it removed the AUC is about 0.69 (see README).")
