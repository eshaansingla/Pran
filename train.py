"""
train.py
========
Full training pipeline: hybrid dataset → XGBoost binary classifier.

Steps:
  1. Load hybrid_features/labels/patient_ids
  2. Patient-level GroupShuffleSplit (80/20)
  3. SMOTE on training split only
  4. XGBoost with scale_pos_weight
  5. Evaluate: AUC, F1, confusion matrix, bootstrap CI (1000 iterations)
  6. Save model to models/xgboost_hybrid.pkl

Usage:
  python train.py
  python train.py --test_size 0.25 --n_estimators 300
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import (
    FEATURE_NAMES, HYBRID_FEATURES, HYBRID_LABELS, HYBRID_IDS,
    MODEL_DIR, RESULTS_DIR,
)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, y_prob, patient_ids, n_iter=1000, ci=0.95):
    """
    Patient-level bootstrap CI.
    Resamples patient IDs with replacement, then takes all windows for each
    sampled patient. Prevents correlated windows from inflating confidence.
    """
    auc_scores, f1_scores = [], []
    rng = np.random.default_rng(42)
    unique_pids = np.unique(patient_ids)

    for _ in range(n_iter):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([np.where(patient_ids == p)[0] for p in sampled_pids])
        yt, yp, yprob = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_scores.append(roc_auc_score(yt, yprob))
        f1_scores.append(f1_score(yt, yp, zero_division=0))

    alpha = (1 - ci) / 2
    def _ci(arr):
        return float(np.mean(arr)), float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha))

    return {
        "auc":  _ci(auc_scores),
        "f1":   _ci(f1_scores),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size",     type=float, default=0.20)
    parser.add_argument("--n_estimators",  type=int,   default=200)
    parser.add_argument("--max_depth",     type=int,   default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--smote_k",       type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    for path in [HYBRID_FEATURES, HYBRID_LABELS, HYBRID_IDS]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run build_hybrid_dataset.py first.")
            sys.exit(1)

    X   = np.load(HYBRID_FEATURES)
    y   = np.load(HYBRID_LABELS)
    ids = np.load(HYBRID_IDS)
    print(f"Loaded: {len(X)} windows, {len(np.unique(ids))} patients")
    print(f"  Normal (0): {(y==0).sum()}  |  Abnormal (1): {(y==1).sum()}")

    # ── Patient-level split ──────────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=ids))

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    train_patients = len(np.unique(ids[train_idx]))
    test_patients  = len(np.unique(ids[test_idx]))
    print(f"\nSplit: train={len(X_tr)} windows ({train_patients} patients) | "
          f"test={len(X_te)} windows ({test_patients} patients)")

    # ── SMOTE on training split only ─────────────────────────────────────────
    # Cap oversampling at 3:1 max to avoid extreme synthetic-sample overfitting.
    # If minority/majority > 1:3 already, SMOTE will only reach 1:3 ratio.
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    minority_count = min(neg, pos)
    majority_count = max(neg, pos)
    # sampling_strategy = target minority / majority ratio after resampling
    max_ratio = 1.0   # 1:1 balance (change to 0.5 for 1:2 if SMOTE over-inflates)
    target_minority = min(int(max_ratio * majority_count), majority_count)
    sampling_strategy = target_minority / majority_count

    print(f"\nApplying SMOTE (k={args.smote_k}, target ratio={sampling_strategy:.2f}) ...")
    k = min(args.smote_k, minority_count - 1)
    if k < 1:
        print("  Not enough minority samples for SMOTE — skipping.")
        X_tr_sm, y_tr_sm = X_tr, y_tr
    else:
        sm = SMOTE(k_neighbors=k, sampling_strategy=sampling_strategy, random_state=args.seed)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    print(f"  After SMOTE: {(y_tr_sm==0).sum()} normal | {(y_tr_sm==1).sum()} abnormal")

    # ── scale_pos_weight: set to 1 after SMOTE (SMOTE already balanced classes)
    # Using pre-SMOTE ratio here would re-introduce imbalance correction on a
    # balanced dataset, which would catastrophically down-weight the abnormal class.
    spw = 1.0
    print(f"\nscale_pos_weight = {spw} (SMOTE already balanced; SPW=1 preserves equal weighting)")

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print("\nTraining XGBoost ...")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_tr_sm, y_tr_sm,
              eval_set=[(X_te, y_te)],
              verbose=50)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)

    auc  = roc_auc_score(y_te, y_prob)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n── Test Results ────────────────────────────")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    test_ids = ids[test_idx]
    print("\nBootstrap CI (1000 iterations, patient-level resample) ...")
    ci_results = bootstrap_ci(y_te, y_pred, y_prob, patient_ids=test_ids)
    auc_mean, auc_lo, auc_hi = ci_results["auc"]
    f1_mean,  f1_lo,  f1_hi  = ci_results["f1"]
    print(f"  AUC: {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  F1:  {f1_mean:.4f}  [{f1_lo:.4f},  {f1_hi:.4f}]")

    # Feature importance
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:<28} {imp:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_hybrid.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    report = {
        "train_windows": int(len(X_tr)),
        "test_windows":  int(len(X_te)),
        "train_patients": int(train_patients),
        "test_patients":  int(test_patients),
        "auc":   round(auc, 4),
        "f1":    round(f1,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
        "f1_ci":  [round(f1_lo,  4), round(f1_hi,  4)],
        "confusion_matrix": cm.tolist(),
        "feature_importances": dict(zip(FEATURE_NAMES,
                                        [round(float(v), 4)
                                         for v in model.feature_importances_])),
    }
    report_path = RESULTS_DIR / "hybrid_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
