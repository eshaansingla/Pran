"""Shared code for the hardware Valsalva-vs-rest study (Valsalva windows = abnormal = 1, all other windows = normal = 0).

Leakage / overfitting controls built in here:
  * every split is by SUBJECT (no subject ever appears on two sides of a split; asserted)
  * a final test set of subjects is drawn once (age-stratified, seed 42) and is never used for model choice
  * the scaler is fit on the fitting subjects only; early stopping and the decision threshold use separate validation subjects
  * no sklearn built-in early stopping (it splits by window, which would leak inside a subject)
  * hyper-parameters are fixed in advance (no tuning on any test data)
  * features are per-window (each window is detrended and normalised on its own); the session label is never a feature
"""
from __future__ import annotations
import os
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO); sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from pran.features import FEATURE_NAMES
from pran.hardware_model import build_dataset

SEED, N_FOLDS, N_VAL, TEST_FRAC = 42, 10, 14, 0.20
OUT = REPO / "hardware-vasalva-method"
MODELS = ["XGBoost", "LightGBM", "CatBoost", "HistGradBoost", "GradientBoosting", "RandomForest", "ExtraTrees", "AdaBoost",
          "DecisionTree", "LogisticRegression", "LinearSVM", "NaiveBayes", "KNN_k50", "MLP_64_32"]


def load():
    """X [n,32], y (1 = Valsalva), subject id, session, age per window (windows are in time order within a subject)."""
    X, P, S, names = build_dataset()
    age_of = {i: int(re.match(r"icp_\d+_(\d+)_", n).group(1)) for i, n in enumerate(names)}
    return X.astype(np.float32), (S == 3).astype(int), P, S, np.array([age_of[p] for p in P]), names


def split_subjects(P, age):
    """Age-stratified final-test subjects (~20%) and the development subjects. Deterministic (seed 42)."""
    rng = np.random.default_rng(SEED); subj = np.unique(P); sage = np.array([age[P == s][0] for s in subj]); test = []
    for lo, hi in [(0, 18), (19, 25), (26, 45), (46, 64), (65, 200)]:
        grp = subj[(sage >= lo) & (sage <= hi)]; test += list(rng.choice(grp, max(1, round(TEST_FRAC * len(grp))), replace=False))
    test = np.array(sorted(test)); dev = np.setdiff1d(subj, test)
    assert not set(test) & set(dev)
    return dev, test


def dev_folds(dev):
    return np.array_split(np.random.default_rng(SEED).permutation(dev), N_FOLDS)


def fold_split(P, dev, folds, i):
    """(fit windows, validation windows, held-out windows) for CV fold i; three disjoint subject sets."""
    f = folds[i]; va_s = np.random.default_rng(i).choice(np.setdiff1d(dev, f), N_VAL, replace=False)
    te = np.where(np.isin(P, f))[0]; va = np.where(np.isin(P, va_s))[0]; fit = np.where(np.isin(P, np.setdiff1d(dev, np.r_[f, va_s])))[0]
    assert not (set(P[te]) & set(P[fit])) and not (set(P[va]) & set(P[fit])) and not (set(P[te]) & set(P[va]))
    return fit, va, te


def make(name, spw):
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 reg_alpha=0.1, scale_pos_weight=spw, tree_method="hist", eval_metric="auc", early_stopping_rounds=50, random_state=SEED, n_jobs=16, verbosity=0)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  is_unbalance=True, random_state=SEED, n_jobs=16, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=500, learning_rate=0.08, depth=6, auto_class_weights="Balanced", eval_metric="AUC", random_seed=SEED, thread_count=16,
                                  verbose=0, early_stopping_rounds=50, allow_writing_files=False)
    if name == "HistGradBoost":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3, subsample=0.8, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(300, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=16, random_state=SEED)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(300, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=16, random_state=SEED)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=SEED)
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    if name == "LinearSVM":
        return LinearSVC(C=0.5, dual=False, class_weight="balanced")
    if name == "NaiveBayes":
        return GaussianNB()
    if name == "KNN_k50":
        return KNeighborsClassifier(50, n_jobs=16)
    if name == "MLP_64_32":
        return MLPClassifier((64, 32), alpha=1e-3, max_iter=100, early_stopping=False, random_state=SEED)
    raise KeyError(name)


def score(m, Z):
    return m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Z)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    return float(th[np.argmax(t - f)])


def fit_model(name, Zf, yf, Zv, yv):
    """Fit on the fitting subjects; validation subjects are used only for early stopping (boosters)."""
    spw = (yf == 0).sum() / max((yf == 1).sum(), 1); m = make(name, spw); it = None
    if name == "XGBoost":
        m.fit(Zf, yf, eval_set=[(Zv, yv)], verbose=False); it = m.best_iteration + 1
    elif name == "LightGBM":
        m.fit(Zf, yf, eval_set=[(Zv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)]); it = m.best_iteration_
    elif name == "CatBoost":
        m.fit(Zf, yf, eval_set=(Zv, yv)); it = m.get_best_iteration() + 1
    elif name in ("AdaBoost", "GradientBoosting"):
        m.fit(Zf, yf, sample_weight=np.where(yf == 1, spw, 1.0))
    else:
        m.fit(Zf, yf)
    return m, it


def fit_scaled(name, X, y, fit, va, cols=None):
    """Scaler fit on FIT subjects only -> fit model -> threshold on VALIDATION subjects. Returns (model, scaler, threshold, iters)."""
    c = slice(None) if cols is None else cols
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED, subsample=10**6).fit(X[fit][:, c])
    Zf, Zv = qt.transform(X[fit][:, c]), qt.transform(X[va][:, c])
    m, it = fit_model(name, Zf, y[fit], Zv, y[va])
    return m, qt, youden(y[va], score(m, Zv)), it


def prob(m, qt, X, cols=None):
    return score(m, qt.transform(X[:, slice(None) if cols is None else cols]))


def metrics(y, s, thr):
    """thr may be a scalar or a per-window array (each fold's own threshold)."""
    pred = (s >= thr).astype(int); tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    se, sp, pc = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1); den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(auc=float(roc_auc_score(y, s)), avg_precision=float(average_precision_score(y, s)), avg_precision_chance=float(y.mean()), sensitivity=se, specificity=sp,
                precision=pc, f1=2 * pc * se / max(pc + se, 1e-9), balanced_accuracy=(se + sp) / 2, accuracy=(tp + tn) / len(y), accuracy_always_normal=float(1 - y.mean()),
                mcc=(tp * tn - fp * fn) / den if den else 0.0, TP=tp, FP=fp, FN=fn, TN=tn)


def subject_auc(y, s, P):
    """Within-subject AUC (Valsalva vs the same person's other windows) for every subject with both classes."""
    return {int(k): float(roc_auc_score(y[P == k], s[P == k])) for k in np.unique(P) if 0 < y[P == k].sum() < (P == k).sum()}


def cluster_ci(y, s, P, n=500, seed=0):
    """95% CI of the pooled AUC, resampling SUBJECTS (not windows) with replacement."""
    rng = np.random.default_rng(seed); subj = np.unique(P); idx = {k: np.where(P == k)[0] for k in subj}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(subj, len(subj))])
        if 0 < y[ii].sum() < len(ii): out.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def session_top(s, P, S):
    """Number of subjects whose Valsalva session has the highest mean score among their four sessions."""
    top = tot = 0
    for k in np.unique(P):
        mean = {j: s[(P == k) & (S == j)].mean() for j in range(4) if ((P == k) & (S == j)).sum() > 5}
        if len(mean) == 4: tot += 1; top += max(mean, key=mean.get) == 3
    return int(top), int(tot)
