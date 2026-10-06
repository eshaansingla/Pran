"""CHARIS-only model comparison: 14 classifiers, one leave-one-patient-out protocol, one folder per model.

    python charis/compare_models.py

Per fold (test = 1 patient): 2 of the other 12 patients are held out for early stopping and the Youden threshold,
the QuantileTransformer is fit on the remaining 10, imbalance is handled with class weights (no SMOTE).
Sklearn models train on a 200k-row random subsample per fold (speed; identical for all of them).

Outputs (models/ and results/ are git-ignored):
  models/charis_compare/<Model>/   model.pkl, qt_scaler.pkl, metrics.json, per_patient.csv
  models/charis_compare/summary.csv
  models/charis_best/              the model with the highest mean leave-one-patient-out AUC (selection rule fixed in advance)
Resumable: finished patients are cached in models/charis_compare/_partial.pkl.
"""
from __future__ import annotations
import json, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED, CAP = 42, 200_000
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
CACHE = Path("results/audit/cache")
ROOT = Path("models/charis_compare")
BEST = Path("models/charis_best")


# name -> (factory(n_iter or None), kind). kind "boost" = early stopping on val patients, full fit set; "sk" = 200k subsample.
def _xgb(n):
    return lambda spw: xgb.XGBClassifier(
        n_estimators=n or 500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", eval_metric="auc", random_state=SEED, n_jobs=16, verbosity=0,
        **({} if n else {"early_stopping_rounds": 50}))


def _lgb(n):
    return lambda spw: lgb.LGBMClassifier(n_estimators=n or 500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                          subsample_freq=1, colsample_bytree=0.8, is_unbalance=True,
                                          random_state=SEED, n_jobs=16, verbose=-1)


def _cat(n):
    return lambda spw: CatBoostClassifier(iterations=n or 500, learning_rate=0.1, depth=6, auto_class_weights="Balanced",
                                          eval_metric="AUC", random_seed=SEED, thread_count=16, verbose=0, allow_writing_files=False,
                                          **({} if n else {"early_stopping_rounds": 50}))


SK = {
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
    "RandomForest": lambda: RandomForestClassifier(200, min_samples_leaf=20, class_weight="balanced", n_jobs=16, random_state=SEED),
    "ExtraTrees": lambda: ExtraTreesClassifier(200, min_samples_leaf=20, class_weight="balanced", n_jobs=16, random_state=SEED),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
    "HistGradBoost": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    "LogisticRegression": lambda: LogisticRegression(class_weight="balanced", max_iter=500),
    "LinearSVM": lambda: LinearSVC(class_weight="balanced", dual=False),
    "NaiveBayes": lambda: GaussianNB(),
    "KNN_k50": lambda: KNeighborsClassifier(50, n_jobs=16),
    "MLP_64_32": lambda: MLPClassifier((64, 32), early_stopping=True, max_iter=100, random_state=SEED),
}
BOOST = {"XGBoost": _xgb, "LightGBM": _lgb, "CatBoost": _cat}
NAMES = list(BOOST) + list(SK)


def score(m, Z):
    return m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Z)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    return float(th[np.argmax(t - f)])


def fit_boost(name, Xf, yf, Xv, yv):
    spw = (yf == 0).sum() / max((yf == 1).sum(), 1)
    m = BOOST[name](None)(spw)
    if name == "XGBoost":
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False); it = m.best_iteration + 1
    elif name == "LightGBM":
        m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)]); it = m.best_iteration_
    else:
        m.fit(Xf, yf, eval_set=(Xv, yv)); it = m.get_best_iteration() + 1
    return m, int(it)


def main():
    X, y, pid = (np.load(CACHE / f"{n}.npy") for n in ("X", "y", "pid"))
    pats = sorted(np.unique(pid)); ROOT.mkdir(parents=True, exist_ok=True)
    part_p = ROOT / "_partial.pkl"
    rows, iters = pickle.load(open(part_p, "rb")) if part_p.exists() else ([], {})
    done = {r["patient"] for r in rows}
    for i, p in enumerate(pats):
        if int(p) in done:
            continue
        t_fold = time.time()
        te = pid == p; others = [q for q in pats if q != p]
        vm = np.isin(pid, [others[(i * 2) % 12], others[(i * 2 + 1) % 12]]); fm = ~te & ~vm
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=1000, subsample=200_000).fit(X[fm])
        Xf, Xv, Xt = (qt.transform(X[m]).astype(np.float32) for m in (fm, vm, te)); yf, yv, yt = y[fm], y[vm], y[te]
        sub = np.random.RandomState(SEED + i).permutation(len(yf))[:CAP]
        vsub = np.random.RandomState(i).permutation(len(yv))[:100_000]
        for name in NAMES:
            if name in BOOST:
                m, it = fit_boost(name, Xf, yf, Xv, yv); iters.setdefault(name, []).append(it); Xv_, yv_ = Xv, yv
            else:
                m = SK[name]().fit(Xf[sub], yf[sub]); Xv_, yv_ = Xv[vsub], yv[vsub]
            thr = youden(yv_, score(m, Xv_)); s = score(m, Xt)
            tn, fp, fn, tp = confusion_matrix(yt, (s >= thr).astype(int), labels=[0, 1]).ravel()
            se, sp, pc = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
            rows.append(dict(model=name, patient=int(p), n_windows=int(te.sum()), abnormal_pct=100 * yt.mean(), auc=roc_auc_score(yt, s),
                             avg_precision=average_precision_score(yt, s), sensitivity=se, specificity=sp, precision=pc,
                             f1=2 * pc * se / max(pc + se, 1e-9), threshold=thr, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
        pickle.dump((rows, iters), open(part_p, "wb"))
        print(f"patient {p} done in {time.time() - t_fold:.0f}s", flush=True)

    df = pd.DataFrame(rows); rng = np.random.RandomState(0); summ = []
    for name in NAMES:
        d = df[df.model == name].sort_values("patient"); a = d.auc.values
        bs = [a[rng.randint(0, len(a), len(a))].mean() for _ in range(5000)]
        summ.append(dict(model=name, auc_mean=a.mean(), auc_std=a.std(ddof=1), auc_ci_lo=np.percentile(bs, 2.5), auc_ci_hi=np.percentile(bs, 97.5),
                         auc_worst_patient=a.min(), patients_auc_ge_0_90=int((a >= .9).sum()), avg_precision=d.avg_precision.mean(),
                         sensitivity=d.sensitivity.mean(), specificity=d.specificity.mean(), precision=d.precision.mean(), f1=d.f1.mean()))
    S = pd.DataFrame(summ).sort_values("auc_mean", ascending=False).reset_index(drop=True); S.to_csv(ROOT / "summary.csv", index=False)
    best = S.model[0]; ba = df[df.model == best].sort_values("patient").auc.values
    S["wilcoxon_p_vs_best"] = [1.0 if n == best else wilcoxon(ba, df[df.model == n].sort_values("patient").auc.values).pvalue for n in S.model]
    S.to_csv(ROOT / "summary.csv", index=False)

    # final models: trained on all 13 patients, own folder each; threshold = median of the fold thresholds
    qt_all = QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=1000, subsample=200_000).fit(X)
    Xa = qt_all.transform(X).astype(np.float32); sub = np.random.RandomState(SEED).permutation(len(y))[:CAP]
    for name in NAMES:
        d = df[df.model == name].sort_values("patient"); out = ROOT / name; out.mkdir(exist_ok=True)
        if name in BOOST:
            n_it = int(np.median(iters[name])); spw = (y == 0).sum() / (y == 1).sum(); m = BOOST[name](n_it)(spw).fit(Xa, y)
        else:
            m = SK[name]().fit(Xa[sub], y[sub])
        thr = float(np.median(d.threshold))
        pickle.dump(m, open(out / "model.pkl", "wb")); pickle.dump(qt_all, open(out / "qt_scaler.pkl", "wb"))
        d.to_csv(out / "per_patient.csv", index=False)
        row = S[S.model == name].iloc[0].to_dict()
        (out / "metrics.json").write_text(json.dumps({**row, "threshold": thr, "features": FEATURES, "protocol": "leave-one-patient-out, 13 folds, CHARIS only"}, indent=1, default=float))
        if name == best:
            BEST.mkdir(exist_ok=True)
            for f in ("model.pkl", "qt_scaler.pkl", "metrics.json", "per_patient.csv"):
                (BEST / f).write_bytes((out / f).read_bytes())
            (BEST / "README.txt").write_text(f"Best CHARIS-only model: {name}\nSelected by highest mean leave-one-patient-out AUC (rule fixed before running).\n"
                                              f"Input: 5 features {FEATURES}, transformed with qt_scaler.pkl. Output: P(ICP >= 20 mmHg window). Decision threshold in metrics.json.\n")
    pd.set_option("display.width", 250); print(S.round(3).to_string(index=False)); print("BEST:", best)


if __name__ == "__main__":
    main()
