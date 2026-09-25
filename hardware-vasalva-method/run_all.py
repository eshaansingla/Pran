"""Train and evaluate 14 models on the hardware data: Valsalva windows = abnormal (1), all other windows = normal (0).

    python hardware-vasalva-method/run_all.py

Protocol (see common.py for the leakage controls):
  1. ~20% of subjects (age-stratified, seed 42) are set aside as the FINAL TEST set and are not looked at until step 4.
  2. Development subjects: 10-fold subject-grouped CV. Per fold: scaler + model fit on the fitting subjects, early stopping and the
     Youden threshold on 14 separate validation subjects, scoring on the held-out subjects. No test-fold information is used.
  3. The model with the highest pooled out-of-fold AUC on the development subjects is the pre-declared winner.
  4. Every model is then refit once (on the development subjects minus 14 validation subjects) and scored ONCE on the final test subjects.
Outputs: hardware-vasalva-method/<Model>/ (model.pkl, qt_scaler.pkl, metrics.json, per_fold.csv, per_subject.csv, predictions.csv) and summary.csv.
Resumable: finished models are cached in hardware-vasalva-method/_partial/.
"""
from __future__ import annotations
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import common as C

X, y, P, S, age, names = C.load(); dev, test = C.split_subjects(P, age); folds = C.dev_folds(dev)
dev_i, test_i = np.isin(P, dev), np.isin(P, test)
print(f"{len(X):,} windows | {len(np.unique(P))} subjects | Valsalva {y.mean():.1%} | development {len(dev)} subjects ({dev_i.sum():,} windows) | FINAL TEST {len(test)} subjects ({test_i.sum():,} windows)")
print("final-test subjects:", [int(s) for s in test], flush=True)
(C.OUT / "_partial").mkdir(exist_ok=True)


def run_model(name):
    part = C.OUT / "_partial" / f"{name}.pkl"
    if part.exists(): return pickle.load(open(part, "rb"))
    t0 = time.time(); oof = np.full(len(y), np.nan); thr_w = np.full(len(y), np.nan); rows = []
    for i in range(C.N_FOLDS):
        fit, va, te = C.fold_split(P, dev, folds, i); t1 = time.time()
        m, qt, thr, it = C.fit_scaled(name, X, y, fit, va)
        s = C.prob(m, qt, X[te]); oof[te] = s; thr_w[te] = thr
        sub = np.random.default_rng(i).choice(fit, min(15000, len(fit)), replace=False); tr_auc = roc_auc_score(y[sub], C.prob(m, qt, X[sub]))
        rows.append(dict(fold=i, held_out_subjects=len(folds[i]), held_out_windows=len(te), auc=roc_auc_score(y[te], s), train_auc=tr_auc, threshold=thr, iterations=it, seconds=time.time() - t1))
    va_s = np.random.default_rng(999).choice(dev, C.N_VAL, replace=False)
    va = np.where(np.isin(P, va_s))[0]; fit = np.where(np.isin(P, np.setdiff1d(dev, va_s)))[0]
    assert not set(P[fit]) & set(P[test_i]) and not set(P[va]) & set(P[test_i])
    m, qt, thr_f, it_f = C.fit_scaled(name, X, y, fit, va); s_test = C.prob(m, qt, X[test_i])           # the ONE scoring of the final test set
    res = dict(oof=oof, thr_w=thr_w, folds=rows, test_scores=s_test, thr_final=thr_f, iters_final=it_f, model=m, qt=qt, minutes=(time.time() - t0) / 60)
    pickle.dump(res, open(part, "wb")); print(f"{name:18} dev CV AUC {roc_auc_score(y[dev_i], oof[dev_i]):.4f} | final test AUC {roc_auc_score(y[test_i], s_test):.4f} | {res['minutes']:.1f} min", flush=True)
    return res


def write_model(name, r):
    d = C.OUT / name; d.mkdir(exist_ok=True); yd, sd, td = y[dev_i], r["oof"][dev_i], r["thr_w"][dev_i]; yt, st = y[test_i], r["test_scores"]
    dm, tm = C.metrics(yd, sd, td), C.metrics(yt, st, r["thr_final"]); lo, hi = C.cluster_ci(yd, sd, P[dev_i], 500); tlo, thi = C.cluster_ci(yt, st, P[test_i], 1000)
    f = pd.DataFrame(r["folds"]); gap = float(f.train_auc.mean() - f.auc.mean())
    verdict = "overfit" if gap > 0.05 else ("underfit" if f.auc.mean() < 0.80 else "ok")
    sa_d, sa_t = C.subject_auc(yd, sd, P[dev_i]), C.subject_auc(yt, st, P[test_i]); td_top, tt_top = C.session_top(sd, P[dev_i], S[dev_i]), C.session_top(st, P[test_i], S[test_i])
    meta = dict(model=name, task="Valsalva windows = abnormal (1), all other windows = normal (0)", n_dev_subjects=len(dev), n_test_subjects=len(test), n_features=len(C.FEATURE_NAMES),
                dev_cv={**dm, "auc_ci95_subject_bootstrap": [lo, hi], "fold_auc_mean": float(f.auc.mean()), "fold_auc_std": float(f.auc.std(ddof=1)), "within_subject_auc_mean": float(np.mean(list(sa_d.values()))),
                        "valsalva_session_highest_of_4": f"{td_top[0]}/{td_top[1]}"},
                final_test={**tm, "auc_ci95_subject_bootstrap": [tlo, thi], "threshold": r["thr_final"], "within_subject_auc_mean": float(np.mean(list(sa_t.values()))),
                            "valsalva_session_highest_of_4": f"{tt_top[0]}/{tt_top[1]}"},
                overfit_check={"train_auc_mean": float(f.train_auc.mean()), "heldout_auc_mean": float(f.auc.mean()), "gap": gap, "verdict": verdict,
                               "rule": "overfit if train-heldout AUC gap > 0.05; underfit if held-out fold AUC < 0.80"},
                protocol=dict(cv="10-fold subject-grouped on development subjects", scaler="QuantileTransformer fit on fitting subjects only", threshold="Youden on 14 separate validation subjects",
                              early_stopping="validation subjects (boosters only)", final_test="scored once, never used for choices"), minutes=r["minutes"])
    (d / "metrics.json").write_text(json.dumps(meta, indent=1, default=float)); f.to_csv(d / "per_fold.csv", index=False)
    pd.DataFrame([dict(subject=k, set="development", within_subject_auc=v) for k, v in sa_d.items()] + [dict(subject=k, set="final_test", within_subject_auc=v) for k, v in sa_t.items()]).to_csv(d / "per_subject.csv", index=False)
    pd.concat([pd.DataFrame(dict(subject=P[dev_i], session=S[dev_i], label=yd, score=sd, threshold=td, set="development")),
               pd.DataFrame(dict(subject=P[test_i], session=S[test_i], label=yt, score=st, threshold=r["thr_final"], set="final_test"))]).to_csv(d / "predictions.csv", index=False)
    pickle.dump(r["model"], open(d / "model.pkl", "wb")); pickle.dump(r["qt"], open(d / "qt_scaler.pkl", "wb"))
    return dict(model=name, dev_auc=dm["auc"], dev_auc_lo=lo, dev_auc_hi=hi, dev_ap=dm["avg_precision"], dev_sens=dm["sensitivity"], dev_spec=dm["specificity"], dev_prec=dm["precision"], dev_f1=dm["f1"],
                dev_bal_acc=dm["balanced_accuracy"], dev_mcc=dm["mcc"], dev_within_subject_auc=meta["dev_cv"]["within_subject_auc_mean"], dev_valsalva_top1=meta["dev_cv"]["valsalva_session_highest_of_4"],
                test_auc=tm["auc"], test_auc_lo=tlo, test_auc_hi=thi, test_ap=tm["avg_precision"], test_sens=tm["sensitivity"], test_spec=tm["specificity"], test_prec=tm["precision"], test_f1=tm["f1"],
                test_bal_acc=tm["balanced_accuracy"], test_mcc=tm["mcc"], train_auc=float(f.train_auc.mean()), heldout_fold_auc=float(f.auc.mean()), train_heldout_gap=gap, fit_verdict=verdict, minutes=r["minutes"]), sa_d


rows, sub_aucs = [], {}
for n in C.MODELS:
    row, sa = write_model(n, run_model(n)); rows.append(row); sub_aucs[n] = sa
S_ = pd.DataFrame(rows).sort_values("dev_auc", ascending=False).reset_index(drop=True); best = S_.model[0]
common = sorted(set.intersection(*[set(v) for v in sub_aucs.values()])); p = []
for n in S_.model:
    a, b = [sub_aucs[best][k] for k in common], [sub_aucs[n][k] for k in common]
    p.append(1.0 if n == best else (wilcoxon(a, b).pvalue if any(np.array(a) != np.array(b)) else 1.0))
order = np.argsort(p); holm = np.empty(len(p)); run = 0.0
for rank, idx in enumerate(order): run = max(run, min(1.0, p[idx] * (len(p) - rank))); holm[idx] = run
S_["p_vs_best_within_subject_auc_holm"] = holm; S_.to_csv(C.OUT / "summary.csv", index=False)
(C.OUT / "best_model.txt").write_text(f"Pre-declared rule: highest pooled out-of-fold AUC on the development subjects.\nBest: {best}\n")
pd.set_option("display.width", 250); print("\n" + S_.round(3).to_string(index=False)); print("\nBEST (development CV):", best)
