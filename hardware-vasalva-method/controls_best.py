"""Extra honesty checks on the winning hardware model. Uses DEVELOPMENT subjects only (the final-test subjects are never touched here).

    python hardware-vasalva-method/controls_best.py

  1. label-shuffle null   : labels shuffled inside each subject -> AUC should fall to ~0.5 (a pipeline that leaks would not)
  2. learning curve       : AUC vs number of training subjects (still rising = data-limited/underfit, flat = saturated)
  3. feature-group ablation: which signals carry the effect (base 5 / IR / displacement / red)
  4. position-only baseline: AUC of "how late in the recording is this window" alone (shows how strong the fixed-order confound could be)
  5. drift control        : does the model separate early-supine from late-supine windows? (should be ~0.5)
  6. session-pair AUCs    : Valsalva vs each other session (head-down is the closest in time)
  7. age / sex breakdown  : does the model work in every age band and both sexes (development + final test, out-of-sample scores only)
"""
from __future__ import annotations
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

def draw(out):
    best, base_auc = out["model"], out["reference_dev_auc"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    ax = axs[0]; ks = list(out["learning_curve"]); ax.plot([int(k) for k in ks], list(out["learning_curve"].values()), "o-", color="#2a78d6", lw=2); ax.set_xscale("log", base=2); ax.set_xlabel("training subjects"); ax.set_ylabel("held-out AUC")
    ax.set_title("Learning curve (flat = not data-limited)"); ax.grid(color="#e6e5e1"); ax.set_ylim(min(.9, min(out["learning_curve"].values()) - .01), 1.0)
    ax = axs[1]; lab = list(out["feature_groups"]); ax.barh(lab[::-1], list(out["feature_groups"].values())[::-1], color="#2a78d6", height=.6); ax.set_xlim(.5, 1.0); ax.set_title("Which signals carry it (development AUC)")
    for i, v in enumerate(list(out["feature_groups"].values())[::-1]): ax.text(v + .003, i, f"{v:.3f}", va="center", fontsize=9)
    ax = axs[2]; nm = ["real\nlabels", "shuffled\nlabels", "position\nonly", "early vs late\nsupine"]; v = [base_auc, out["label_shuffle_auc"], out["position_only_auc"], out["drift_supine_early_vs_late_auc_mean"]]
    ax.bar(nm, v, color=["#2a78d6", "#52514e", "#eb6834", "#eb6834"], width=.6); ax.axhline(.5, color="#0b0b0b", ls="--", lw=1); ax.set_ylim(0, 1.05); ax.set_title("Controls (0.5 = chance)"); ax.tick_params(axis="x", labelsize=9)
    for i, x_ in enumerate(v): ax.text(i, x_ + .015, f"{x_:.3f}", ha="center", fontsize=9)
    for a in axs: a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{best}: honesty checks on development subjects", x=.01, ha="left", fontweight="bold", fontsize=13); fig.tight_layout(); fig.savefig(C.OUT / "controls_best.png", dpi=130)


if "--replot" in sys.argv:
    draw(json.loads((C.OUT / "controls_best.json").read_text())); sys.exit()

X, y, P, S, age, names = C.load(); dev, test = C.split_subjects(P, age); folds = C.dev_folds(dev); dev_i = np.isin(P, dev)
best = pd.read_csv(C.OUT / "summary.csv").model[0]; print("winner:", best, flush=True); out = {"model": best}
sex_of = {i: re.match(r"icp_\d+_\d+_([MF])", n).group(1) for i, n in enumerate(names)}


def cv_auc(ycv, cols=None, n_train_subjects=None, seed=0):
    oof = np.full(len(y), np.nan)
    for i in range(C.N_FOLDS):
        fit, va, te = C.fold_split(P, dev, folds, i)
        if n_train_subjects is not None:
            keep = np.random.default_rng(seed + i).choice(np.unique(P[fit]), min(n_train_subjects, len(np.unique(P[fit]))), replace=False); fit = fit[np.isin(P[fit], keep)]
        m, qt, thr, it = C.fit_scaled(best, X, ycv, fit, va, cols); oof[te] = C.prob(m, qt, X[te], cols)
    return float(roc_auc_score(ycv[dev_i], oof[dev_i])), oof


base_auc, oof = cv_auc(y); out["reference_dev_auc"] = base_auc; print("reference dev AUC", round(base_auc, 4), flush=True)
rng = np.random.default_rng(7); ys = y.copy()
for k in np.unique(P): idx = np.where(P == k)[0]; ys[idx] = rng.permutation(y[idx])
out["label_shuffle_auc"] = cv_auc(ys)[0]; print("label-shuffle null AUC", round(out["label_shuffle_auc"], 4), flush=True)
out["learning_curve"] = {str(n): cv_auc(y, n_train_subjects=n)[0] for n in (8, 16, 32, 64, 92)}; print("learning curve", {k: round(v, 4) for k, v in out["learning_curve"].items()}, flush=True)
groups = {"all 32 features": None, "5 CHARIS-style base only": list(range(0, 5)), "IR channel only (9)": list(range(5, 14)), "displacement channel only (9)": list(range(14, 23)),
          "red channel only (9)": list(range(23, 32)), "IR + red (18)": list(range(5, 14)) + list(range(23, 32)), "no base features (27)": list(range(5, 32))}
out["feature_groups"] = {k: (base_auc if v is None else cv_auc(y, cols=v)[0]) for k, v in groups.items()}
for k, v in out["feature_groups"].items(): print(f"  {k:32} {v:.4f}", flush=True)
pos = np.zeros(len(y))
for k in np.unique(P): m = P == k; pos[m] = np.linspace(0, 1, m.sum())
out["position_only_auc"] = float(roc_auc_score(y[dev_i], pos[dev_i]))
sc = pd.read_csv(C.OUT / best / "predictions.csv"); d = sc[sc.set == "development"].reset_index(drop=True); d["pos"] = 0
early = []; pooled_y, pooled_s = [], []
for k, g in d.groupby("subject", sort=False):
    g = g[g.session == 0]
    if len(g) < 20: continue
    h = len(g) // 2; yy = np.r_[np.zeros(h), np.ones(len(g) - h)]; ss = g.score.values; early.append(roc_auc_score(yy, ss)); pooled_y += list(yy); pooled_s += list(ss - ss[:h].mean())
out["drift_supine_early_vs_late_auc_mean"] = float(np.mean(early)); out["drift_supine_early_vs_late_auc_sd"] = float(np.std(early))
print("position-only AUC", round(out["position_only_auc"], 3), "| supine early-vs-late (model score) per-subject AUC", round(out["drift_supine_early_vs_late_auc_mean"], 3), flush=True)
allp = sc.copy(); pair = {}
for a, nm in [(0, "supine"), (1, "head-up"), (2, "head-down")]:
    for which in ("development", "final_test"):
        g = allp[(allp.set == which) & (allp.session.isin([a, 3]))]; pair[f"Valsalva vs {nm} ({which})"] = float(roc_auc_score(g.session == 3, g.score))
out["session_pair_auc"] = pair
for k, v in pair.items(): print(f"  {k:44} {v:.4f}")
age_of_subj = {int(s): int(age[P == s][0]) for s in np.unique(P)}; allp["age"] = allp.subject.map(age_of_subj)
rows = []
for lo, hi, nm in [(0, 18, "<=18"), (19, 25, "19-25"), (26, 45, "26-45"), (46, 64, "46-64"), (65, 200, "65+")]:
    g = allp[(allp.age >= lo) & (allp.age <= hi)]; rows.append((nm, g.subject.nunique(), float(roc_auc_score(g.label, g.score)), float(np.mean(list(C.subject_auc(g.label.values, g.score.values, g.subject.values).values())))))
out["by_age"] = [dict(band=a, subjects=n, pooled_auc=p, within_subject_auc=w) for a, n, p, w in rows]
allp["sex"] = allp.subject.map(sex_of); out["by_sex"] = [dict(sex=s, subjects=int(g.subject.nunique()), pooled_auc=float(roc_auc_score(g.label, g.score))) for s, g in allp.groupby("sex")]
for r in out["by_age"]: print(f"  age {r['band']:6} n={r['subjects']:3} pooled AUC {r['pooled_auc']:.3f} within-subject {r['within_subject_auc']:.3f}")
for r in out["by_sex"]: print(f"  sex {r['sex']} n={r['subjects']:3} pooled AUC {r['pooled_auc']:.3f}")
(C.OUT / "controls_best.json").write_text(json.dumps(out, indent=1))
draw(out); print("saved controls")
