"""Train + save the production CORAL-aligned CHARIS classifier.

Mirrors models/charis_compare/XGBoost's layout (model.pkl, qt_scaler.pkl,
metrics.json) but domain-adapted: CHARIS features are CORAL-aligned onto
the hardware background's covariance structure before the classifier is
fit, so the decision boundary is less dependent on CHARIS-only geometry.

Trained ONLY on CHARIS features + real CHARIS labels. Hardware data is used
ONLY, unsupervised, to define the covariance target for CORAL -- no hardware
rows or labels ever enter clf.fit(). See charis/coral/README.md.

Run from repo root: python charis/coral/train.py
Outputs -> charis/coral/{model.pkl, qt_charis.pkl, qt_hw.pkl, coral.npz, metrics.json}
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import QuantileTransformer

OUT = Path(__file__).parent
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]


def sqrtm(cov):
    u, s, _ = np.linalg.svd(cov)
    return u @ np.diag(np.sqrt(s + 1e-8)) @ u.T


def invsqrtm(cov):
    u, s, _ = np.linalg.svd(cov)
    return u @ np.diag(1 / np.sqrt(s + 1e-8)) @ u.T


def coral_align(X_source_qt, cov_source, cov_target):
    return (X_source_qt @ invsqrtm(cov_source)) @ sqrtm(cov_target)


def youden_threshold(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    Xc = np.load("results/audit/cache/X.npy").astype(np.float64)
    yc = np.load("results/audit/cache/y.npy").astype(int)
    pidc = np.load("results/audit/cache/pid.npy").astype(int)
    Xh_bg = np.load("support/results/hw_features_cache.npz")["X"].astype(np.float64)

    print(f"CHARIS: {len(Xc):,} windows, 13 patients | hardware background: {len(Xh_bg):,} windows (unlabeled, alignment target only)")

    qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    Xc_qt = qt_c.fit_transform(Xc)
    Xh_qt = qt_h.fit_transform(Xh_bg)

    cov_source = np.cov(Xc_qt.T)
    cov_target = np.cov(Xh_qt.T)
    Xc_al = coral_align(Xc_qt, cov_source, cov_target)

    n_pos, n_neg = yc.sum(), len(yc) - yc.sum()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=100, reg_lambda=10,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="auc", random_state=0, n_jobs=-1,
    )
    clf.fit(Xc_al, yc)
    p_all = clf.predict_proba(Xc_al)[:, 1]
    threshold = youden_threshold(yc, p_all)
    train_auc = roc_auc_score(yc, p_all)
    print(f"fit AUC (in-sample, all 13 patients): {train_auc:.4f}  threshold={threshold:.3f}")

    pickle.dump(clf, open(OUT / "model.pkl", "wb"))
    pickle.dump(qt_c, open(OUT / "qt_charis.pkl", "wb"))
    pickle.dump(qt_h, open(OUT / "qt_hw.pkl", "wb"))
    np.savez(OUT / "coral.npz", cov_source=cov_source, cov_target=cov_target)

    metrics = {
        "features": FEATURES,
        "trained_on": "13 CHARIS patients (real labels); hardware background used only, "
                       "unsupervised, as the CORAL alignment target -- no hardware rows/labels in clf.fit()",
        "n_charis_windows": int(len(Xc)),
        "n_hw_background_windows": int(len(Xh_bg)),
        "threshold": threshold,
        "fit_auc_in_sample": float(train_auc),
        "lopo_mean_auc_intact": 0.8967,
        "lopo_mean_auc_demeaned": 0.8451,
        "lopo_note": "see charis/coral_lopo.py and charis/coral_lopo_demean.py for the leak-checked "
                     "per-patient CV this model's design was validated with (patient 5 unreliable: "
                     "AUC 0.745 intact -> 0.464 de-meaned, i.e. baseline-level artifact, not real signal)",
        "domain_classifier_auc_after_coral": 0.824,
        "domain_classifier_note": "CHARIS vs hardware still distinguishable at AUC~0.82-0.87 after CORAL; "
                                   "this is a structural ceiling (CHARIS features derive from 1 ICP channel, "
                                   "hardware features derive from 2 independent sensors), not fixable by more scaling",
    }
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    print(f"saved model.pkl, qt_charis.pkl, qt_hw.pkl, coral.npz, metrics.json -> {OUT}")


if __name__ == "__main__":
    main()
