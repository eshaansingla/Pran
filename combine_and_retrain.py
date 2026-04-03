"""
combine_and_retrain.py
======================
1. Load CHARIS features  (data/processed/features.npy        -- 11 cols, use first 8)
   and MIMIC features    (data/processed/mimic_features.npy  -- 8 cols)
2. Combine them and overwrite data/processed/{features,labels,patient_ids}.npy
3. Retrain XGBoost (--no_phase, 8 features) using patient-level GroupShuffleSplit
4. Print classification report

Usage:
    python combine_and_retrain.py
    python combine_and_retrain.py --out_dir models/xgboost_combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Combine data ───────────────────────────────────────────────────────────────

def combine_datasets(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CHARIS + MIMIC arrays and return combined (X8, y, pids)."""

    # CHARIS
    ch_feat = np.load(processed_dir / "features.npy")        # (N, 11)
    ch_lab  = np.load(processed_dir / "labels.npy")          # (N,)
    ch_pid  = np.load(processed_dir / "patient_ids.npy")     # (N,)  IDs 1-13

    # Use only first 8 features (no phase-lag)
    ch_feat8 = ch_feat[:, :8]

    print(f"  CHARIS : {ch_feat8.shape[0]:>7,} windows | "
          f"{len(set(ch_pid.tolist()))} patients")

    # MIMIC (may not exist yet)
    mimic_feat_path = processed_dir / "mimic_features.npy"
    if mimic_feat_path.exists():
        mi_feat = np.load(mimic_feat_path)                   # (M, 8)
        mi_lab  = np.load(processed_dir / "mimic_labels.npy")
        mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy")

        # Safety: truncate or pad to 8 cols
        if mi_feat.shape[1] > 8:
            mi_feat = mi_feat[:, :8]
        elif mi_feat.shape[1] < 8:
            pad = np.zeros((mi_feat.shape[0], 8 - mi_feat.shape[1]), dtype=np.float32)
            mi_feat = np.hstack([mi_feat, pad])

        print(f"  MIMIC  : {mi_feat.shape[0]:>7,} windows | "
              f"{len(set(mi_pid.tolist()))} patients")

        X  = np.vstack([ch_feat8, mi_feat]).astype(np.float32)
        y  = np.concatenate([ch_lab, mi_lab]).astype(np.int64)
        pid = np.concatenate([ch_pid.astype(np.int32), mi_pid.astype(np.int32)])
    else:
        print("  MIMIC data not found -- training on CHARIS only.")
        X   = ch_feat8.astype(np.float32)
        y   = ch_lab.astype(np.int64)
        pid = ch_pid.astype(np.int32)

    print(f"  Combined: {X.shape[0]:>7,} windows | "
          f"{len(set(pid.tolist()))} patients")
    _print_dist(y, "Combined")

    return X, y, pid


def _print_dist(y: np.ndarray, name: str) -> None:
    for cls, lbl in zip([0, 1, 2], ["Normal", "Elevated", "Critical"]):
        n = int((y == cls).sum())
        print(f"    {lbl:<10}: {n:>7,}  ({100*n/len(y):.1f}%)")


# ── Train + evaluate ───────────────────────────────────────────────────────────

def retrain(
    X: np.ndarray,
    y: np.ndarray,
    pid: np.ndarray,
    out_dir: Path,
    charis_max_pid: int = 13,
) -> None:
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import (
        classification_report, balanced_accuracy_score, f1_score
    )
    import xgboost as xgb
    import pickle

    out_dir.mkdir(parents=True, exist_ok=True)
    SEED = 42

    # ---- Strategy: CHARIS patients -> 70/10/20 split; MIMIC -> train only ----
    # MIMIC ICP has very different distribution (87% Normal) vs CHARIS (38% Normal)
    # Mixing MIMIC into test set would make evaluation unfair and misleading.
    # All MIMIC patients are added to the training set only.

    charis_mask = pid <= charis_max_pid
    mimic_mask  = ~charis_mask

    X_ch, y_ch, pid_ch = X[charis_mask], y[charis_mask], pid[charis_mask]
    X_mi, y_mi         = X[mimic_mask],  y[mimic_mask]

    n_charis_patients = len(set(pid_ch.tolist()))
    n_mimic_windows   = int(mimic_mask.sum())
    print(f"\n[2/3] Split strategy:")
    print(f"  CHARIS ({n_charis_patients} patients): patient-level 70/10/20 split")
    print(f"  MIMIC  ({n_mimic_windows:,} windows):  all added to TRAINING only")

    # CHARIS: Train+Val / Test split
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv_idx, test_idx = next(gss_outer.split(X_ch, y_ch, groups=pid_ch))

    X_ch_tv, y_ch_tv, pid_ch_tv = X_ch[tv_idx], y_ch[tv_idx], pid_ch[tv_idx]
    X_test, y_test               = X_ch[test_idx], y_ch[test_idx]

    # CHARIS: Train / Val split
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    tr_idx, val_idx = next(gss_inner.split(X_ch_tv, y_ch_tv, groups=pid_ch_tv))

    X_ch_train, y_ch_train = X_ch_tv[tr_idx], y_ch_tv[tr_idx]
    X_val,      y_val       = X_ch_tv[val_idx], y_ch_tv[val_idx]

    # Combine CHARIS-train + all MIMIC for final train set
    X_train = np.vstack([X_ch_train, X_mi])
    y_train = np.concatenate([y_ch_train, y_mi])

    n_tr, n_val, n_te = len(X_train), len(X_val), len(X_test)
    print(f"\n  Train: {n_tr:,}  Val: {n_val:,}  Test: {n_te:,}")
    _print_dist(y_train, "Train")
    _print_dist(y_val,   "Val  ")
    _print_dist(y_test,  "Test ")

    # ---- class weights ----
    class_counts = np.bincount(y_train, minlength=3).astype(float)
    total        = class_counts.sum()
    sample_weights = np.where(
        y_train == 0, total / (3 * class_counts[0]),
        np.where(y_train == 1, total / (3 * class_counts[1]),
                               total / (3 * class_counts[2]))
    )

    # ---- build eval set ----
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective":        "multi:softmax",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "eta":              0.05,
        "max_depth":        6,
        "min_child_weight": 5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "lambda":           1.0,
        "alpha":            0.1,
        "random_state":     SEED,
        "tree_method":      "hist",
        "verbosity":        0,
    }

    print(f"\n[3/3] Training XGBoost (early stopping) ...")
    evals_result: dict = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
        evals_result=evals_result,
    )

    print(f"\n  Best iteration: {bst.best_iteration}")

    # ---- evaluate ----
    y_pred = bst.predict(dtest).astype(int)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["Normal", "Elevated", "Critical"])

    gap = macro_f1 - 0.70
    status = "PASS" if macro_f1 >= 0.70 else f"FAIL (need +{abs(gap):.4f})"

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  [{status}]")
    print(f"{'='*60}")
    print(f"  Macro F1-score:    {macro_f1:.4f}   (target >= 0.70)")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print()
    print(report)
    print(f"{'='*60}\n")

    # ---- save ----
    model_pkl = out_dir / "xgboost_combined.pkl"
    with open(model_pkl, "wb") as fh:
        pickle.dump(bst, fh)

    report_txt = out_dir / "classification_report_combined.txt"
    with open(report_txt, "w") as fh:
        fh.write(f"Macro F1-score:    {macro_f1:.4f}\n")
        fh.write(f"Balanced Accuracy: {bal_acc:.4f}\n\n")
        fh.write(report)

    print(f"  Model  saved: {model_pkl}")
    print(f"  Report saved: {report_txt}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine CHARIS+MIMIC datasets and retrain XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir", type=Path, default=Path("data/processed"),
        help="Directory with CHARIS + MIMIC .npy files.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("models/xgboost_combined"),
        help="Where to save the trained model and report.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  COMBINE + RETRAIN PIPELINE")
    print(f"{'='*60}\n")
    print("[1/3] Loading datasets ...")
    X, y, pid = combine_datasets(args.processed_dir)

    retrain(X, y, pid, args.out_dir)


if __name__ == "__main__":
    main()
