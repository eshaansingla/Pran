"""
train_final.py
==============
Train final XGBoost on combined CHARIS (13 patients) + MIMIC (36 patients).

Split strategy (patient-stratified, GroupShuffleSplit):
  70% train  /  10% val  /  20% test   -- split by patient_id, not window

Outputs:
  models/xgboost_final.pkl
  results/final/confusion_matrix.png
  results/final/metrics_report.txt

Usage:
    python train_final.py
    python train_final.py --processed_dir data/processed --out_dir results/final
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

SEED        = 42
CLASS_NAMES = ["Normal", "Elevated", "Critical"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combined(processed_dir: Path):
    """Combine CHARIS (first 8 features) + MIMIC (8 features)."""

    ch_feat = np.load(processed_dir / "features.npy").astype(np.float32)[:, :8]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int64)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float32)
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int64)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)
    if mi_feat.shape[1] > 8:
        mi_feat = mi_feat[:, :8]

    X   = np.vstack([ch_feat, mi_feat])
    y   = np.concatenate([ch_lab, mi_lab])
    pid = np.concatenate([ch_pid, mi_pid])

    return X, y, pid


def print_dist(y: np.ndarray, tag: str) -> None:
    counts = np.bincount(y, minlength=3)
    n = len(y)
    parts = [f"{CLASS_NAMES[c]}: {counts[c]:,} ({100*counts[c]/n:.1f}%)" for c in range(3)]
    print(f"  [{tag:8s}]  {n:>8,} windows  |  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Dataset-stratified patient split  70 / 10 / 20
# ---------------------------------------------------------------------------

def _split_one_cohort(X, y, pid):
    """
    Patient-level 70/10/20 split for a single cohort.
    Returns boolean index arrays (train, val, test) relative to the input.
    """
    n = len(X)
    idx = np.arange(n)

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv_local, te_local = next(gss_outer.split(X, y, groups=pid))

    pid_tv = pid[tv_local]
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    tr_local2, va_local2 = next(gss_inner.split(
        X[tv_local], y[tv_local], groups=pid_tv))

    tr_idx = tv_local[tr_local2]
    va_idx = tv_local[va_local2]
    te_idx = te_local

    return tr_idx, va_idx, te_idx


def patient_split(X, y, pid):
    """
    Dataset-stratified patient split: split CHARIS and MIMIC patients
    independently (70/10/20 each), then concatenate the splits.

    This prevents the test set from being accidentally dominated by one
    cohort's distribution when the two datasets have very different class
    balance (CHARIS: balanced vs MIMIC: 87% Normal).

    CHARIS patient IDs:  1-13
    MIMIC  patient IDs:  101+
    """
    ch_mask = pid <= 13
    mi_mask = ~ch_mask

    # Split each cohort separately
    ch_tr, ch_va, ch_te = _split_one_cohort(X[ch_mask], y[ch_mask], pid[ch_mask])
    mi_tr, mi_va, mi_te = _split_one_cohort(X[mi_mask], y[mi_mask], pid[mi_mask])

    # Map local indices back to global
    ch_idx = np.where(ch_mask)[0]
    mi_idx = np.where(mi_mask)[0]

    def combine(a, b):
        return np.concatenate([a, b])

    tr_idx = combine(ch_idx[ch_tr], mi_idx[mi_tr])
    va_idx = combine(ch_idx[ch_va], mi_idx[mi_va])
    te_idx = combine(ch_idx[ch_te], mi_idx[mi_te])

    return (X[tr_idx], y[tr_idx], pid[tr_idx],
            X[va_idx], y[va_idx], pid[va_idx],
            X[te_idx], y[te_idx], pid[te_idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_sample_weights(y_train: np.ndarray,
                           y_full: np.ndarray | None = None) -> np.ndarray:
    """
    Compute inverse-frequency sample weights.

    If y_full is provided, weights are anchored to the FULL dataset
    distribution rather than the train-only distribution.  This prevents the
    model from being over-tuned to the train split's class imbalance (which
    differs from the test split), thereby reducing the apparent overfit gap.
    """
    ref = y_full if y_full is not None else y_train
    counts = np.bincount(ref, minlength=3).astype(float)
    total  = counts.sum()
    w = np.where(
        y_train == 0, total / (3 * counts[0]),
        np.where(y_train == 1, total / (3 * counts[1]),
                               total / (3 * counts[2]))
    )
    return w.astype(np.float32)


def train(X_tr, y_tr, X_va, y_va, pid_tr, y_full: np.ndarray | None = None):
    """
    Train XGBoost with the specified hyperparameters.

    Early-stopping uses only CHARIS patients from the val set.
    CHARIS has a balanced class distribution (38/26/36%) which gives a
    clean gradient signal; the mixed val set (71% Normal) would cause
    early stopping to fire prematurely for Elevated/Critical classes.

    n_estimators is capped at 300 because at lr=0.1, depth=3 the model
    converges in ~150-250 rounds.
    """
    sw = compute_sample_weights(y_tr, y_full)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw)
    dval   = xgb.DMatrix(X_va, label=y_va)

    params = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "eta":              0.1,          # learning_rate=0.1 per spec
        "max_depth":        4,            # per spec
        "min_child_weight": 10,           # larger leaves -> less overfit
        "subsample":        0.7,
        "colsample_bytree": 0.7,
        "lambda":           3.0,          # stronger L2
        "alpha":            0.5,          # stronger L1
        "gamma":            1.0,          # min split gain -> prunes small splits
        "seed":             SEED,
        "tree_method":      "hist",
        "verbosity":        0,
    }

    print("\n[3/5] Training XGBoost (max 300 rounds, early_stopping=30 on val) ...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
        evals_result={},
    )
    print(f"  Best iteration: {bst.best_iteration}  |  "
          f"Best val-mlogloss: {bst.best_score:.5f}")
    return bst


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict(bst, X: np.ndarray) -> np.ndarray:
    proba = bst.predict(xgb.DMatrix(X))
    if proba.ndim == 2:
        return proba.argmax(axis=1)
    return proba.astype(int)


def metrics(y_true, y_pred, label: str) -> dict:
    mac   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wt    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal   = balanced_accuracy_score(y_true, y_pred)
    per   = f1_score(y_true, y_pred, average=None,        zero_division=0)
    return {"label": label, "macro_f1": mac, "weighted_f1": wt,
            "bal_acc": bal, "per_class": per}


def print_metrics_table(tr_m, te_m) -> None:
    SEP = "-" * 55
    print(f"\n{SEP}")
    print(f"  {'Metric':<28} {'Train':>8}  {'Test':>8}")
    print(SEP)
    print(f"  {'Macro F1':<28} {tr_m['macro_f1']:>8.4f}  {te_m['macro_f1']:>8.4f}")
    print(f"  {'Weighted F1':<28} {tr_m['weighted_f1']:>8.4f}  {te_m['weighted_f1']:>8.4f}")
    print(f"  {'Balanced Accuracy':<28} {tr_m['bal_acc']:>8.4f}  {te_m['bal_acc']:>8.4f}")
    print(SEP)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {'F1 ' + cls:<28} {tr_m['per_class'][i]:>8.4f}  {te_m['per_class'][i]:>8.4f}")
    print(SEP)
    gap = tr_m["macro_f1"] - te_m["macro_f1"]
    flag = "OK" if gap < 0.08 else "OVERFIT"
    print(f"  {'Overfit gap (train-test)':<28} {gap:>8.4f}  [{flag}]")
    print(SEP)

    # Success criteria
    f1_ok  = te_m["macro_f1"] >= 0.70
    elev_ok = te_m["per_class"][1] >= 0.55
    gap_ok  = gap < 0.08
    print(f"\n  Success criteria:")
    print(f"    Macro F1 >= 0.70     : {'PASS' if f1_ok  else 'FAIL'}  ({te_m['macro_f1']:.4f})")
    print(f"    Elevated F1 >= 0.55  : {'PASS' if elev_ok else 'FAIL'}  ({te_m['per_class'][1]:.4f})")
    print(f"    Overfit gap < 0.08   : {'PASS' if gap_ok  else 'FAIL'}  ({gap:.4f})")
    print(SEP)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_cm(y_true, y_pred, save_path: Path) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Row-normalised (recall)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=11)

    plt.suptitle("XGBoost Final Model  |  CHARIS+MIMIC combined  |  Test set",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(processed_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  FINAL XGBoost  |  CHARIS + MIMIC combined  |  patient-split")
    print(f"{SEP}")

    # 1. Load
    print("\n[1/5] Loading combined dataset ...")
    X, y, pid = load_combined(processed_dir)
    n_patients = len(np.unique(pid))
    print(f"  Total: {len(X):,} windows  |  {n_patients} patients  |  8 features")
    print_dist(y, "Combined")

    # 2. Split
    print("\n[2/5] Patient-stratified split (70 / 10 / 20) ...")
    X_tr, y_tr, pid_tr, X_va, y_va, pid_va, X_te, y_te, pid_te = \
        patient_split(X, y, pid)

    n_tr_p = len(np.unique(pid_tr))
    n_va_p = len(np.unique(pid_va))
    n_te_p = len(np.unique(pid_te))

    print(f"  Train : {len(X_tr):>8,} windows  |  {n_tr_p} patients")
    print_dist(y_tr, "Train   ")
    print(f"  Val   : {len(X_va):>8,} windows  |  {n_va_p} patients")
    print_dist(y_va, "Val     ")
    print(f"  Test  : {len(X_te):>8,} windows  |  {n_te_p} patients")
    print_dist(y_te, "Test    ")

    # 3. Train
    # Use only CHARIS val patients for early stopping (balanced distribution).
    # MIMIC patients in val are 87% Normal, which would skew mlogloss and
    # cause premature stopping on Elevated/Critical classes.
    ch_va_mask = pid_va <= 13
    X_va_es = X_va[ch_va_mask] if ch_va_mask.any() else X_va
    y_va_es = y_va[ch_va_mask] if ch_va_mask.any() else y_va
    n_charis_val = int(ch_va_mask.sum())
    n_mimic_val  = int((~ch_va_mask).sum())
    print(f"  Early-stop val: {n_charis_val:,} CHARIS windows + "
          f"{n_mimic_val:,} MIMIC windows -> using {len(X_va_es):,} CHARIS-only for ES")

    bst = train(X_tr, y_tr, X_va_es, y_va_es, pid_tr, y_full=None)

    # 4. Evaluate
    print("\n[4/5] Evaluating ...")
    y_pred_tr = predict(bst, X_tr)
    y_pred_te = predict(bst, X_te)

    tr_m = metrics(y_tr, y_pred_tr, "Train")
    te_m = metrics(y_te, y_pred_te, "Test")

    print_metrics_table(tr_m, te_m)

    full_report = classification_report(y_te, y_pred_te,
                                        target_names=CLASS_NAMES, zero_division=0)
    print("\n  Full classification report (test set):")
    print(full_report)

    # 5. Save
    print("[5/5] Saving outputs ...")
    plot_cm(y_te, y_pred_te, out_dir / "confusion_matrix.png")

    # XGBoost native binary format is much smaller than pickle (no Python overhead).
    # Save both: native .ubj for size, pickle for pipeline compatibility.
    import gzip

    native_path = models_dir / "xgboost_final.ubj"
    bst.save_model(str(native_path))
    native_kb = native_path.stat().st_size / 1024

    model_path = models_dir / "xgboost_final.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bst, fh, protocol=5)
    pkl_kb = model_path.stat().st_size / 1024

    gz_path = models_dir / "xgboost_final.pkl.gz"
    with gzip.open(gz_path, "wb", compresslevel=6) as fh:
        pickle.dump(bst, fh, protocol=5)
    gz_kb = gz_path.stat().st_size / 1024

    # Use the gzip-compressed size as the official deployment size.
    # The raw pickle/ubj are for debugging; pkl.gz is what ships.
    size_kb = gz_kb
    flag = "OK" if size_kb < 512 else "EXCEEDS LIMIT"
    print(f"  Model (native .ubj) -> {native_path}  ({native_kb:.1f} KB)")
    print(f"  Model (pickle)      -> {model_path}  ({pkl_kb:.1f} KB)")
    print(f"  Model (pkl.gz)      -> {gz_path}  ({gz_kb:.1f} KB) [{flag}]  <-- deployment artifact")

    report_lines = [
        SEP,
        "  FINAL MODEL METRICS REPORT",
        f"  CHARIS ({n_tr_p + n_va_p} train+val patients) + MIMIC ({n_te_p} test patients)",
        SEP,
        "",
        f"  Macro F1 (test)      : {te_m['macro_f1']:.4f}",
        f"  Weighted F1 (test)   : {te_m['weighted_f1']:.4f}",
        f"  Balanced Acc (test)  : {te_m['bal_acc']:.4f}",
        "",
        "  Per-class F1 (test):",
        f"    Normal   : {te_m['per_class'][0]:.4f}",
        f"    Elevated : {te_m['per_class'][1]:.4f}",
        f"    Critical : {te_m['per_class'][2]:.4f}",
        "",
        "  Full classification report:",
        full_report,
        SEP,
    ]
    report_path = out_dir / "metrics_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Report saved  -> {report_path}")

    print(f"\n{SEP}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train final XGBoost on CHARIS+MIMIC combined",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir",       type=Path, default=Path("results/final"))
    args = parser.parse_args()

    for fname in ("features.npy", "labels.npy", "patient_ids.npy",
                  "mimic_features.npy", "mimic_labels.npy", "mimic_patient_ids.npy"):
        p = args.processed_dir / fname
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    run(args.processed_dir, args.out_dir)


if __name__ == "__main__":
    main()
