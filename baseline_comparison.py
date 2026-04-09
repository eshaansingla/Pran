"""
baseline_comparison.py
======================
Compare XGBoost binary ICP classifier against simple baselines:
  1. Logistic Regression
  2. Random Forest (n=100)
  3. SVM (RBF kernel)
  4. Single-feature threshold (cardiac_amplitude only)
  5. XGBoost (current model)

Uses identical patient-level split as train_binary.py.

Usage:
    python baseline_comparison.py
    python baseline_comparison.py --processed_dir data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

SEED = 42
FEATURE_NAMES = [
    "cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
    "slow_wave_power", "cardiac_power", "mean_arterial_pressure",
]
CLASS_NAMES = ["Normal", "Abnormal"]


# ---------------------------------------------------------------------------
# Data + Split (identical to train_binary.py)
# ---------------------------------------------------------------------------

def load_binary(processed_dir: Path):
    KEEP = [0, 1, 2, 3, 4, 5]
    ch_feat = np.load(processed_dir / "features.npy").astype(np.float32)[:, KEEP]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int64)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float32)[:, KEEP]
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int64)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)

    X   = np.vstack([ch_feat, mi_feat])
    y   = (np.concatenate([ch_lab, mi_lab]) >= 1).astype(np.int64)
    pid = np.concatenate([ch_pid, mi_pid])
    return X, y, pid


def _split_cohort(X, y, pid):
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(X, y, groups=pid))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr, va = next(gss2.split(X[tv], y[tv], groups=pid[tv]))
    return tv[tr], tv[va], te


def patient_split(X, y, pid):
    ch = pid <= 13
    mi = ~ch
    ch_idx = np.where(ch)[0]
    mi_idx = np.where(mi)[0]
    ch_tr, ch_va, ch_te = _split_cohort(X[ch], y[ch], pid[ch])
    mi_tr, mi_va, mi_te = _split_cohort(X[mi], y[mi], pid[mi])
    tr = np.concatenate([ch_idx[ch_tr], mi_idx[mi_tr]])
    va = np.concatenate([ch_idx[ch_va], mi_idx[mi_va]])
    te = np.concatenate([ch_idx[ch_te], mi_idx[mi_te]])
    return (X[tr], y[tr], X[va], y[va], X[te], y[te])


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

def eval_model(name, y_te, y_pred, y_prob=None):
    """Return dict of metrics."""
    f1   = f1_score(y_te, y_pred, zero_division=0)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_te, y_pred)
    auc_val = roc_auc_score(y_te, y_prob) if y_prob is not None else 0.0
    return {
        "model": name, "f1": round(f1, 4), "auc": round(auc_val, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "balanced_acc": round(bacc, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline comparison for binary ICP classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir",       type=Path, default=Path("results/binary"))
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON — Binary ICP Classifier")
    print("=" * 70)

    X, y, pid = load_binary(args.processed_dir)

    # Impute NaN values with column median (MIMIC has ~8 NaN values)
    # XGBoost handles NaN natively, but sklearn models cannot.
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values found — imputing with column median")
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j]

    X_tr, y_tr, X_va, y_va, X_te, y_te = patient_split(X, y, pid)
    print(f"  Train: {len(y_tr):,}  Val: {len(y_va):,}  Test: {len(y_te):,}\n")

    # Scale for linear models
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    results = []

    # 1. Logistic Regression
    print("  Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_pred = lr.predict(X_te_s)
    results.append(eval_model("Logistic Regression", y_te, lr_pred, lr_prob))

    # 2. Random Forest
    print("  Training Random Forest (n=100) ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_pred = rf.predict(X_te)
    results.append(eval_model("Random Forest (100)", y_te, rf_pred, rf_prob))

    # 3. SVM (RBF)
    print("  Training SVM (RBF, probability=True) ...")
    # Subsample for SVM speed (>100k samples is too slow)
    n_svm = min(20000, len(X_tr_s))
    rng = np.random.RandomState(SEED)
    svm_idx = rng.choice(len(X_tr_s), n_svm, replace=False)
    svc = SVC(kernel="rbf", probability=True, random_state=SEED,
              class_weight="balanced")
    svc.fit(X_tr_s[svm_idx], y_tr[svm_idx])
    svm_prob = svc.predict_proba(X_te_s)[:, 1]
    svm_pred = svc.predict(X_te_s)
    results.append(eval_model(f"SVM RBF (n={n_svm})", y_te, svm_pred, svm_prob))

    # 4. Single-feature threshold (cardiac_amplitude)
    print("  Training single-feature threshold (cardiac_amplitude) ...")
    ca_tr = X_tr[:, 0]
    ca_te = X_te[:, 0]
    # Find optimal threshold on training data
    best_t, best_f1 = 0, 0
    for t in np.linspace(ca_tr.min(), ca_tr.max(), 200):
        pred = (ca_tr >= t).astype(int)
        f = f1_score(y_tr, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    sf_pred = (ca_te >= best_t).astype(int)
    sf_prob = (ca_te - ca_tr.min()) / (ca_tr.max() - ca_tr.min() + 1e-8)
    results.append(eval_model(f"Cardiac Amp Threshold ({best_t:.1f})", y_te, sf_pred, sf_prob))

    # 5. XGBoost
    print("  Training XGBoost ...")
    n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n0 / max(n1, 1), 4)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    dte    = xgb.DMatrix(X_te)
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "eta": 0.1, "max_depth": 4, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "lambda": 1.0, "alpha": 0.1,
        "seed": SEED, "tree_method": "hist", "verbosity": 0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=500,
                     evals=[(dval, "val")],
                     early_stopping_rounds=30, verbose_eval=0)
    xgb_prob = bst.predict(dte)
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    results.append(eval_model("XGBoost (ours)", y_te, xgb_pred, xgb_prob))

    # --- Report ---
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  BASELINE COMPARISON RESULTS")
    print(SEP)
    print(f"  {'Model':<30} {'F1':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'BalAcc':>7}")
    print(f"  {'-'*73}")
    for r in results:
        marker = "  ***" if r["model"] == "XGBoost (ours)" else ""
        print(f"  {r['model']:<30} {r['f1']:>6.4f}  {r['auc']:>6.4f}  "
              f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['balanced_acc']:>7.4f}{marker}")
    print(SEP)

    # --- Statistical significance vs XGBoost ---
    # McNemar's test: does XGBoost make significantly different errors than each baseline?
    # Null hypothesis: both models make the same number of errors.
    # p < 0.05 → XGBoost is significantly different (better or worse).
    from scipy.stats import chi2

    def mcnemar_test(y_true, pred_a, pred_b):
        """McNemar's mid-P test comparing model A vs model B predictions."""
        b = int(((pred_a == y_true) & (pred_b != y_true)).sum())  # A right, B wrong
        c = int(((pred_a != y_true) & (pred_b == y_true)).sum())  # A wrong, B right
        if b + c == 0:
            return 1.0  # no discordant pairs — identical behaviour
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - chi2.cdf(chi2_stat, df=1))

    # Reconstruct per-window predictions for significance testing
    # XGBoost predictions (last entry in results)
    xgb_pred_vec = (bst.predict(dte) >= 0.5).astype(int)
    baseline_preds = {
        "Logistic Regression":   lr.predict(X_te_s),
        f"Random Forest (100)":  rf.predict(X_te),
        f"SVM RBF (n={n_svm})":  svc.predict(X_te_s),
        f"Cardiac Amp Threshold ({best_t:.1f})": sf_pred,
    }

    SEP2 = "=" * 70
    print(f"\n{SEP2}")
    print("  STATISTICAL SIGNIFICANCE vs XGBoost (McNemar's test)")
    print(f"  {'Baseline':<35} {'p-value':>10}  {'Significant?':>14}")
    print(f"  {'-'*62}")
    sig_results = {}
    for bname, bpred in baseline_preds.items():
        p = mcnemar_test(y_te, xgb_pred_vec, bpred)
        sig = "Yes (p<0.05)" if p < 0.05 else "No"
        print(f"  {bname:<35} {p:>10.4f}  {sig:>14}")
        sig_results[bname] = {"mcnemar_p": round(p, 4), "significant": p < 0.05}
    print(SEP2)

    for r in results:
        if r["model"] in sig_results:
            r["mcnemar_vs_xgboost"] = sig_results[r["model"]]

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "baseline_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
