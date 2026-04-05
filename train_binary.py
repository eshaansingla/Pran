"""
train_binary.py
===============
Binary ICP classifier: Normal (<15 mmHg) vs Abnormal (>=15 mmHg).

Rationale: the 3-class model's Elevated decision region collapsed to
max 3% probability under any input — it effectively predicted only
Normal vs Critical. Converting to binary at the 15 mmHg clinical
intervention threshold produces an honest, well-calibrated model.

Split: dataset-stratified 70/10/20 (CHARIS and MIMIC split separately
       then merged — prevents one cohort dominating the test set).

Usage:
    python train_binary.py
    python train_binary.py --processed_dir data/processed --out_dir results/binary
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

SEED        = 42
CLASS_NAMES = ["Normal", "Abnormal"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_binary(processed_dir: Path):
    # Keep only the 6 validated features; drop head_angle (col 6) and
    # motion_artifact_flag (col 7) — ablation confirmed 0% gain and noise.
    KEEP = [0, 1, 2, 3, 4, 5]

    ch_feat = np.load(processed_dir / "features.npy").astype(np.float32)[:, KEEP]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int64)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float32)
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int64)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)
    mi_feat = mi_feat[:, KEEP]

    X   = np.vstack([ch_feat, mi_feat])
    y3  = np.concatenate([ch_lab, mi_lab])
    pid = np.concatenate([ch_pid, mi_pid])

    # Relabel: 0=Normal, 1=Abnormal (Elevated+Critical)
    y = (y3 >= 1).astype(np.int64)
    return X, y, pid


def print_dist(y, tag):
    n = (y == 0).sum()
    a = (y == 1).sum()
    t = len(y)
    print(f"  [{tag:8s}] Normal: {n:,} ({100*n/t:.1f}%)  Abnormal: {a:,} ({100*a/t:.1f}%)")


# ---------------------------------------------------------------------------
# Split — dataset-stratified (CHARIS and MIMIC independently)
# ---------------------------------------------------------------------------

def _split_cohort(X, y, pid):
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(X, y, groups=pid))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr, va = next(gss2.split(X[tv], y[tv], groups=pid[tv]))
    return tv[tr], tv[va], te


def patient_split(X, y, pid):
    # CHARIS patients have IDs 1–13 (assigned during CHARIS extraction).
    # MIMIC patients now use their real subject_id (7-digit numbers >> 13),
    # so this boundary is stable regardless of how many MIMIC patients exist.
    ch = pid <= 13
    mi = ~ch

    n_ch = len(np.unique(pid[ch]))
    n_mi = len(np.unique(pid[mi]))
    assert n_ch > 0, "No CHARIS patients found (pid <= 13)"
    assert n_mi > 0, "No MIMIC patients found (pid > 13)"

    # Verify GroupShuffleSplit will work: need >= 5 unique patients per cohort
    if n_ch < 5 or n_mi < 5:
        raise ValueError(
            f"Too few patients for stratified split: CHARIS={n_ch}, MIMIC={n_mi}. "
            "Need >= 5 per cohort."
        )

    ch_idx = np.where(ch)[0]
    mi_idx = np.where(mi)[0]

    ch_tr, ch_va, ch_te = _split_cohort(X[ch], y[ch], pid[ch])
    mi_tr, mi_va, mi_te = _split_cohort(X[mi], y[mi], pid[mi])

    tr = np.concatenate([ch_idx[ch_tr], mi_idx[mi_tr]])
    va = np.concatenate([ch_idx[ch_va], mi_idx[mi_va]])
    te = np.concatenate([ch_idx[ch_te], mi_idx[mi_te]])

    return (X[tr], y[tr], pid[tr],
            X[va], y[va], pid[va],
            X[te], y[te], pid[te])


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X_tr, y_tr, X_va, y_va):
    n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n0 / n1, 4)
    print(f"  scale_pos_weight = {spw}  (Normal {n0:,} / Abnormal {n1:,})")

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)

    params = {
        "objective":       "binary:logistic",
        "eval_metric":     "auc",
        "eta":             0.1,
        "max_depth":       4,
        "min_child_weight": 5,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "lambda":          1.0,
        "alpha":           0.1,
        "seed":            SEED,
        "tree_method":     "hist",
        "verbosity":       0,
    }

    print("\n[3/5] Training XGBoost binary (max 500 rounds, early_stopping=30) ...")
    bst = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
        evals_result={},
    )
    print(f"  Best iteration: {bst.best_iteration}  |  Best val-AUC: {bst.best_score:.5f}")
    return bst, spw


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — lower is better, 0 is perfect."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - float(y_true[mask].mean()))
    return float(ece)


def calibrate(bst, X_va: np.ndarray, y_va: np.ndarray, pid_va: np.ndarray, n_folds: int = 5):
    """
    Cross-validated isotonic calibration to prevent overfitting to a single val set.

    Splits the validation set into n_folds patient-level folds.
    For each fold: fit IsotonicRegression on the other folds, predict on this fold.
    Aggregates all out-of-fold calibrated probabilities, then fits a final
    IsotonicRegression on all of them — giving a calibrator that generalises
    better than single-fold fitting.

    Returns
    -------
    calibrator   : IsotonicRegression  — fit on all OOF calibrated probs
    threshold    : float               — optimal decision boundary on OOF probs
    ece_before   : float               — ECE of raw XGBoost probs
    ece_after    : float               — ECE after cross-validated calibration
    """
    from sklearn.model_selection import KFold

    prob_raw = bst.predict(xgb.DMatrix(X_va))
    ece_before = _ece(y_va, prob_raw)

    # Patient-level K-fold so same patient never spans train+test within calibration
    unique_pids = np.unique(pid_va)
    n_folds = min(n_folds, len(unique_pids))  # can't have more folds than patients
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(y_va), dtype=np.float64)

    for fold_pids_tr, fold_pids_te in kf.split(unique_pids):
        pids_tr = unique_pids[fold_pids_tr]
        pids_te = unique_pids[fold_pids_te]
        mask_tr = np.isin(pid_va, pids_tr)
        mask_te = np.isin(pid_va, pids_te)

        fold_cal = IsotonicRegression(out_of_bounds="clip")
        fold_cal.fit(prob_raw[mask_tr], y_va[mask_tr])
        oof_probs[mask_te] = fold_cal.predict(prob_raw[mask_te])

    ece_after = _ece(y_va, oof_probs)

    # Final calibrator: fit on all OOF predictions vs true labels
    final_cal = IsotonicRegression(out_of_bounds="clip")
    final_cal.fit(oof_probs, y_va)

    # Optimal threshold on OOF calibrated probs
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f = f1_score(y_va, (oof_probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)

    print(f"  CV calibration: {n_folds} patient-level folds")
    return final_cal, round(best_t, 4), round(ece_before, 4), round(ece_after, 4)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(bst, X_tr, y_tr, X_te, y_te, out_dir: Path,
             calibrator=None, threshold: float = 0.5):
    dtr = xgb.DMatrix(X_tr)
    dte = xgb.DMatrix(X_te)

    prob_tr = bst.predict(dtr)
    prob_te = bst.predict(dte)

    if calibrator is not None:
        prob_tr = calibrator.predict(prob_tr)
        prob_te = calibrator.predict(prob_te)

    ece_te = _ece(y_te, prob_te)

    pred_tr = (prob_tr >= threshold).astype(int)
    pred_te = (prob_te >= threshold).astype(int)

    auc_te  = roc_auc_score(y_te, prob_te)
    f1_te   = f1_score(y_te, pred_te, zero_division=0)
    prec_te = precision_score(y_te, pred_te, zero_division=0)
    rec_te  = recall_score(y_te, pred_te, zero_division=0)
    spec_te = recall_score(1 - y_te, 1 - pred_te, zero_division=0)
    bacc_te = balanced_accuracy_score(y_te, pred_te)

    f1_tr   = f1_score(y_tr, pred_tr, zero_division=0)
    gap     = f1_tr - f1_te

    cm = confusion_matrix(y_te, pred_te)
    tn, fp, fn, tp = cm.ravel()

    cal_tag = f"calibrated (threshold={threshold:.4f})" if calibrator is not None else "uncalibrated (threshold=0.5)"
    SEP = "=" * 60
    lines = [
        SEP,
        "  BINARY CLASSIFICATION RESULTS",
        "  Normal (<15 mmHg)  vs  Abnormal (>=15 mmHg)",
        f"  Probabilities: {cal_tag}",
        SEP,
        "",
        "  Test-set metrics:",
        f"    F1-score        : {f1_te:.4f}   (target >= 0.80)",
        f"    AUC             : {auc_te:.4f}   (target >= 0.90)",
        f"    Precision       : {prec_te:.4f}",
        f"    Recall (Sens.)  : {rec_te:.4f}",
        f"    Specificity     : {spec_te:.4f}",
        f"    Balanced Acc.   : {bacc_te:.4f}",
        f"    ECE (test)      : {ece_te:.4f}   (lower is better, 0=perfect)",
        "",
        "  Overfitting check:",
        f"    Train F1        : {f1_tr:.4f}",
        f"    Test  F1        : {f1_te:.4f}",
        f"    Gap             : {gap:+.4f}  ({'OK' if gap < 0.08 else 'OVERFIT'})",
        "",
        "  Confusion matrix (test set):",
        f"    True Normal  predicted Normal   (TN): {tn:,}",
        f"    True Normal  predicted Abnormal (FP): {fp:,}",
        f"    True Abnormal predicted Normal  (FN): {fn:,}",
        f"    True Abnormal predicted Abnormal(TP): {tp:,}",
        "",
        "  Success criteria:",
        f"    F1 >= 0.80  : {'PASS' if f1_te >= 0.80 else 'FAIL'}  ({f1_te:.4f})",
        f"    AUC >= 0.90 : {'PASS' if auc_te >= 0.90 else 'FAIL'}  ({auc_te:.4f})",
        "",
    ]
    report = classification_report(y_te, pred_te, target_names=CLASS_NAMES, zero_division=0)
    lines += ["  Classification report:", report, SEP]
    output = "\n".join(lines)
    print(output)

    # Plots: ROC | Confusion Matrix | Calibration Curve
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fpr, tpr, color="#2C5282", lw=2, label=f"AUC = {auc_te:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Binary ICP Classifier")
    axes[0].legend(); axes[0].spines[["top", "right"]].set_visible(False)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={"shrink": 0.8}, linewidths=0.5)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix — Test Set")

    frac_pos, mean_pred = calibration_curve(y_te, prob_te, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "s-", color="#2C5282",
                 label=f"Model (ECE={ece_te:.4f})")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].set_title("Reliability Diagram — Test Set")
    axes[2].legend(); axes[2].spines[["top", "right"]].set_visible(False)

    plt.suptitle("XGBoost Binary ICP Classifier  |  CHARIS+MIMIC  |  Test set",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "binary_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save report
    (out_dir / "binary_report.txt").write_text(output, encoding="utf-8")

    return {
        "f1": f1_te, "auc": auc_te, "precision": prec_te,
        "recall": rec_te, "specificity": spec_te, "balanced_acc": bacc_te,
        "ece": round(ece_te, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(processed_dir: Path, out_dir: Path):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  BINARY ICP CLASSIFIER  |  Normal vs Abnormal (>=15 mmHg)")
    print(f"{SEP}")

    print("\n[1/5] Loading and relabelling data ...")
    X, y, pid = load_binary(processed_dir)
    print(f"  Total: {len(X):,} windows | {len(np.unique(pid))} patients | 6 features")
    print_dist(y, "Combined")

    print("\n[2/5] Dataset-stratified split (70 / 10 / 20) ...")
    X_tr, y_tr, pid_tr, X_va, y_va, pid_va, X_te, y_te, pid_te = \
        patient_split(X, y, pid)
    print(f"  Train : {len(X_tr):,} windows | {len(np.unique(pid_tr))} patients")
    print_dist(y_tr, "Train")
    print(f"  Val   : {len(X_va):,} windows | {len(np.unique(pid_va))} patients")
    print_dist(y_va, "Val  ")
    print(f"  Test  : {len(X_te):,} windows | {len(np.unique(pid_te))} patients")
    print_dist(y_te, "Test ")

    # Use CHARIS-only val for early stopping (balanced signal)
    ch_va = pid_va <= 13
    X_va_es = X_va[ch_va] if ch_va.any() else X_va
    y_va_es = y_va[ch_va] if ch_va.any() else y_va
    print(f"  Early-stop val: {len(X_va_es):,} CHARIS windows")

    bst, spw = train(X_tr, y_tr, X_va_es, y_va_es)

    print("\n[3.5/5] Calibrating probabilities (cross-validated isotonic on val set) ...")
    calibrator, threshold, ece_before, ece_after = calibrate(bst, X_va, y_va, pid_va)
    print(f"  ECE before: {ece_before:.4f}  ->  after: {ece_after:.4f}")
    print(f"  Optimal threshold (max val-F1): {threshold:.4f}")

    print("\n[4/5] Evaluating (calibrated) ...")
    metrics = evaluate(bst, X_tr, y_tr, X_te, y_te, out_dir, calibrator, threshold)

    print("[5/5] Saving model + calibrator ...")
    Path("models").mkdir(exist_ok=True)

    pkl_path = Path("models/xgboost_binary.pkl")
    gz_path  = Path("models/xgboost_binary.pkl.gz")

    with open(pkl_path, "wb") as f:
        pickle.dump(bst, f, protocol=5)
    with gzip.open(gz_path, "wb", compresslevel=6) as f:
        pickle.dump(bst, f, protocol=5)

    cal_gz_path = Path("models/xgboost_binary_calibrator.pkl.gz")
    with gzip.open(cal_gz_path, "wb", compresslevel=6) as f:
        pickle.dump(calibrator, f, protocol=5)

    size_kb = gz_path.stat().st_size / 1024
    flag = "OK" if size_kb < 512 else "EXCEEDS LIMIT"
    print(f"  models/xgboost_binary.pkl.gz             ({size_kb:.1f} KB) [{flag}]")
    print(f"  models/xgboost_binary_calibrator.pkl.gz  (saved)")
    print(f"  results saved -> {out_dir}/\n")

    # Store comprehensive metadata for API
    import json
    from datetime import date
    n_mimic   = int(len(np.unique(pid[pid > 13])))
    n_charis  = int(len(np.unique(pid[pid <= 13])))
    meta = {
        "version":               "2.2",
        "training_date":         date.today().isoformat(),
        "scale_pos_weight":      spw,
        "threshold_mmhg":        15.0,
        "prob_threshold":        threshold,
        "ece_before_calibration": ece_before,
        "ece_after_calibration":  ece_after,
        "training_data": {
            "charis_patients":   n_charis,
            "mimic_patients":    n_mimic,
            "total_patients":    n_charis + n_mimic,
            "total_windows":     int(len(X)),
        },
        "metrics": metrics,
    }
    (Path("models") / "binary_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir",       type=Path, default=Path("results/binary"))
    args = parser.parse_args()
    for f in ("features.npy","labels.npy","patient_ids.npy",
              "mimic_features.npy","mimic_labels.npy","mimic_patient_ids.npy"):
        if not (args.processed_dir / f).exists():
            print(f"ERROR: {args.processed_dir/f} not found"); sys.exit(1)
    run(args.processed_dir, args.out_dir)

if __name__ == "__main__":
    main()
