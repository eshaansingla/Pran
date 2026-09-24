"""Hardware model: recognises Valsalva windows (head 1) and the ICP-ladder state (head 2) from optical features.

Train (needs the private recordings in hw-tests/):   python -m pran.hardware_model
Labels are the recorded manoeuvres, so this is a *state-recognition* model, not an ICP measurement.
Head 1 "valsalva": Valsalva windows = 1, all other windows = 0.
Head 2 "ladder"  : head-down + Valsalva = 1, supine + head-up = 0 (the higher-ICP half of the protocol).
All reported metrics are from subject-grouped 13-fold cross-validation (no subject in both train and test);
thresholds are chosen on 14 validation subjects inside each training set (never on the test subjects).
"""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .features import FEATURE_NAMES, REQUIRED_COLUMNS, recording_windows

MODEL_DIR = Path("models/hardware")
FEATURE_CACHE = Path("results/hardware_features.npz")
SEED, N_FOLDS, N_VAL, TRAIL = 42, 13, 14, 6   # TRAIL = trailing windows averaged (~35 s)
PARAMS = {"objective": "binary:logistic", "eta": 0.05, "max_depth": 5, "min_child_weight": 3, "subsample": 0.8,
          "colsample_bytree": 0.8, "lambda": 1.0, "alpha": 0.1, "tree_method": "hist", "seed": SEED, "verbosity": 0}


def _weights(y):
    return np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((y == 0).sum(), 1)) * len(y)


def _fit(X, y):
    return xgb.train(PARAMS, xgb.DMatrix(X, label=y, weight=_weights(y), feature_names=FEATURE_NAMES), 300)


def _predict(booster, X):
    return booster.predict(xgb.DMatrix(np.asarray(X, np.float32), feature_names=FEATURE_NAMES))


def trailing_mean(scores, k=TRAIL):
    """Causal moving average over the last k windows (past only)."""
    c = np.cumsum(np.r_[0.0, scores]); out = np.empty(len(scores))
    for t in range(len(scores)):
        lo = max(0, t - k + 1); out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def build_dataset(hw_dir="hw-tests"):
    """Features for every recording -> X [n,32], subject index, session label, file names (cached)."""
    if FEATURE_CACHE.exists():
        z = np.load(FEATURE_CACHE, allow_pickle=True)
        return z["X"], z["P"], z["S"], [str(n) for n in z["names"]]
    X, P, S, names = [], [], [], []
    for f in sorted(Path(hw_dir).glob("icp_*.csv")):
        df = pd.read_csv(f, comment="#", low_memory=False)
        if not (REQUIRED_COLUMNS | {"session_label"}).issubset(df.columns):
            continue
        Xi, Si = recording_windows(df)
        if len(Xi) == 0:
            continue
        X.append(Xi); S.append(Si); P.append(np.full(len(Xi), len(names))); names.append(f.name)
    X, P, S = np.vstack(X), np.concatenate(P), np.concatenate(S)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURE_CACHE, X=X, P=P, S=S, names=np.array(names))
    return X, P, S, names


def _smooth_by_subject(scores, pids, k):
    if k == 1:
        return scores
    out = np.empty(len(scores))
    for s in np.unique(pids):
        m = pids == s
        out[m] = trailing_mean(scores[m], k)
    return out


def _cv_head(X, P, y, folds, subjects):
    oof = {1: np.zeros(len(y)), TRAIL: np.zeros(len(y))}; thr = {1: np.zeros(len(y)), TRAIL: np.zeros(len(y))}; all_thr = {1: [], TRAIL: []}
    for i, f in enumerate(folds):
        te = np.where(np.isin(P, f))[0]
        va_s = np.random.default_rng(i).choice(np.setdiff1d(subjects, f), N_VAL, replace=False)
        va = np.where(np.isin(P, va_s))[0]; tr = np.where(~np.isin(P, np.r_[f, va_s]))[0]
        assert not (set(P[te]) & set(P[tr])) and not (set(P[va]) & set(P[tr]))
        b = _fit(X[tr], y[tr]); pv, pt = _predict(b, X[va]), _predict(b, X[te])
        for k in (1, TRAIL):
            fpr, tpr, t = roc_curve(y[va], _smooth_by_subject(pv, P[va], k)); th = float(t[np.argmax(tpr - fpr)]); all_thr[k].append(th)
            oof[k][te] = _smooth_by_subject(pt, P[te], k); thr[k][te] = th
    return oof, thr, all_thr


def _metrics(y, o, th):
    pred = (o >= th).astype(int)
    tp, fp, fn, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1)
    r = lambda v: round(float(v), 3)
    return {"auc": r(roc_auc_score(y, o)), "avg_precision": r(average_precision_score(y, o)), "avg_precision_chance": r(y.mean()),
            "sensitivity": r(sens), "specificity": r(spec), "precision": r(prec), "f1": r(2 * prec * sens / max(prec + sens, 1e-9)),
            "balanced_accuracy": r((sens + spec) / 2), "accuracy": r((tp + tn) / len(y)), "accuracy_always_negative": r(1 - y.mean())}


def train(hw_dir="hw-tests", out=MODEL_DIR):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    X, P, S, names = build_dataset(hw_dir); subjects = np.unique(P)
    folds = np.array_split(np.random.default_rng(SEED).permutation(subjects), N_FOLDS)
    print(f"{len(X):,} windows | {len(subjects)} subjects | {X.shape[1]} features")
    meta = {"trained": dt.date.today().isoformat(), "n_windows": int(len(X)), "n_subjects": int(len(subjects)),
            "n_features": int(X.shape[1]), "feature_names": FEATURE_NAMES}
    # ---- head 1: Valsalva
    y = (S == 3).astype(int); oof, thr, all_thr = _cv_head(X, P, y, folds, subjects)
    meta["thresholds"] = {"window": float(np.median(all_thr[1])), "trailing": float(np.median(all_thr[TRAIL]))}
    meta["valsalva_window"] = _metrics(y, oof[1], thr[1]); meta["valsalva_trailing"] = _metrics(y, oof[TRAIL], thr[TRAIL])
    w = [roc_auc_score(y[P == s], oof[1][P == s]) for s in subjects if 0 < y[P == s].sum() < (P == s).sum()]
    meta["valsalva_within_subject_auc"] = round(float(np.mean(w)), 3)
    m = np.isin(S, [0, 3]); meta["valsalva_vs_supine_auc"] = round(float(roc_auc_score(y[m], oof[1][m])), 3)
    units, ulab, top1 = [], [], []
    for s in subjects:
        mm = P == s; mean = {k: oof[1][mm & (S == k)].mean() for k in range(4) if (mm & (S == k)).sum() > 5}
        if len(mean) == 4:
            units += list(mean.values()); ulab += [int(k == 3) for k in mean]; top1.append(max(mean, key=mean.get) == 3)
    meta["valsalva_session_level"] = {"auc": round(float(roc_auc_score(ulab, units)), 3), "highest_of_4_sessions": f"{int(np.sum(top1))}/{len(top1)}"}
    _fit(X, y).save_model(str(out / "valsalva.json"))
    # ---- head 2: ICP ladder
    y2 = np.isin(S, [2, 3]).astype(int); oof2, _, _ = _cv_head(X, P, y2, folds, subjects); sc = oof2[1]
    rho, mono, hu = [], [], []
    for s in subjects:
        mm = P == s; mean = [sc[mm & (S == k)].mean() if (mm & (S == k)).sum() > 5 else np.nan for k in (1, 0, 2, 3)]
        if not np.isnan(mean).any():
            rho.append(spearmanr(range(4), mean)[0]); mono.append(all(mean[i] < mean[i + 1] for i in range(3))); hu.append(mean[0] < mean[1])
    meta["ladder"] = {"window_auc": round(float(roc_auc_score(y2, sc)), 3), "mean_within_subject_spearman": round(float(np.mean(rho)), 3),
                      "strictly_monotone": f"{int(np.sum(mono))}/{len(mono)}", "head_up_below_supine": f"{int(np.sum(hu))}/{len(hu)}"}
    _fit(X, y2).save_model(str(out / "ladder.json"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("thresholds", "valsalva_window", "valsalva_session_level", "ladder")}, indent=1))
    return meta


class HardwareModel:
    """Loads the trained heads and scores windows."""

    def __init__(self, model_dir=MODEL_DIR):
        d = Path(model_dir)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No trained hardware model in {d}. Run: python -m pran.hardware_model")
        self.meta = json.loads((d / "meta.json").read_text())
        self.valsalva, self.ladder = xgb.Booster(), xgb.Booster()
        self.valsalva.load_model(str(d / "valsalva.json")); self.ladder.load_model(str(d / "ladder.json"))
        self.threshold = self.meta["thresholds"]["window"]

    def score(self, X):
        return _predict(self.valsalva, X), _predict(self.ladder, X)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    train()
