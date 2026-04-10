"""
cross_dataset_eval.py
=====================
Proper cross-dataset generalisation test:

  Train : CHARIS only (13 patients, patient-level 80/20 train/val split)
  Val   : held-out CHARIS patients (early-stopping only -- no MIMIC seen)
  Test  : ALL MIMIC patients (true out-of-distribution evaluation)

This avoids the data-leakage in combine_and_retrain.py where MIMIC
windows were injected into the training set while being scored on a
CHARIS-only test set.

Usage:
    python cross_dataset_eval.py
    python cross_dataset_eval.py --processed_dir data/processed --out_dir results/cross_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


SEED        = 42
CLASS_NAMES = ["Normal", "Elevated", "Critical"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _print_dist(y: np.ndarray, tag: str) -> None:
    counts = np.bincount(y, minlength=3)
    total  = len(y)
    parts  = [f"{CLASS_NAMES[c]}: {counts[c]:,} ({100*counts[c]/total:.1f}%)"
              for c in range(3)]
    print(f"  [{tag}]  {total:,} windows  |  " + "  ".join(parts))


def _load(processed_dir: Path) -> tuple:
    """Load CHARIS and MIMIC arrays, select the 6 validated features."""
    KEEP = [0, 1, 2, 3, 4, 5]   # same as train_binary.py

    # CHARIS
    ch_feat = np.load(processed_dir / "features.npy").astype(np.float32)[:, KEEP]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int64)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    # MIMIC
    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float32)[:, KEEP]
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int64)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)

    return ch_feat, ch_lab, ch_pid, mi_feat, mi_lab, mi_pid


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(processed_dir: Path, out_dir: Path) -> None:
    import xgboost as xgb

    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  CROSS-DATASET GENERALISATION EVALUATION")
    print("  Train: CHARIS only  |  Test: MIMIC (OOD, never seen in training)")
    print(f"{SEP}\n")

    # 1. Load
    print("[1/4] Loading datasets ...")
    ch_feat, ch_lab, ch_pid, mi_feat, mi_lab, mi_pid = _load(processed_dir)

    ch_patients = len(np.unique(ch_pid))
    mi_patients = len(np.unique(mi_pid))
    print(f"  CHARIS : {ch_feat.shape[0]:>7,} windows | {ch_patients} patients")
    _print_dist(ch_lab, "CHARIS")
    print(f"  MIMIC  : {mi_feat.shape[0]:>7,} windows | {mi_patients} patients")
    _print_dist(mi_lab, "MIMIC ")

    # 2. Patient-level split WITHIN CHARIS (train / val)
    # Val used only for XGBoost early stopping -- no MIMIC data seen.
    print(f"\n[2/4] Splitting CHARIS patients (80% train / 20% val) ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, val_idx = next(gss.split(ch_feat, ch_lab, groups=ch_pid))

    X_train, y_train = ch_feat[tr_idx],  ch_lab[tr_idx]
    X_val,   y_val   = ch_feat[val_idx], ch_lab[val_idx]
    X_test,  y_test  = mi_feat,          mi_lab

    tr_pats  = len(np.unique(ch_pid[tr_idx]))
    val_pats = len(np.unique(ch_pid[val_idx]))
    print(f"  Train : {len(X_train):>7,} windows | {tr_pats} CHARIS patients")
    _print_dist(y_train, "Train")
    print(f"  Val   : {len(X_val):>7,} windows | {val_pats} CHARIS patients (early-stop only)")
    _print_dist(y_val, "Val  ")
    print(f"  Test  : {len(X_test):>7,} windows | {mi_patients} MIMIC patients (OOD)")
    _print_dist(y_test, "Test ")

    # 3. Class weights (computed on train only)
    counts = np.bincount(y_train, minlength=3).astype(float)
    total  = counts.sum()
    sw = np.where(
        y_train == 0, total / (3 * counts[0]),
        np.where(y_train == 1, total / (3 * counts[1]),
                               total / (3 * counts[2]))
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # 4. Train XGBoost
    print(f"\n[3/4] Training XGBoost on CHARIS (early stopping on CHARIS val) ...")
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
        "seed":             SEED,
        "tree_method":      "hist",
        "verbosity":        0,
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200,
        evals_result={},
    )
    print(f"  Best iteration: {bst.best_iteration}")

    # 5. Evaluate on MIMIC
    print(f"\n[4/4] Evaluating on MIMIC (out-of-distribution) ...")
    y_pred = bst.predict(dtest).astype(int)

    macro_f1  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    wt_f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    bal_acc   = balanced_accuracy_score(y_test, y_pred)
    per_class = f1_score(y_test, y_pred, average=None,        zero_division=0)
    report    = classification_report(y_test, y_pred,
                                      target_names=CLASS_NAMES, zero_division=0)

    # CHARIS internal val F1 for gap comparison
    y_pred_val = bst.predict(dval).astype(int)
    charis_f1  = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
    gap        = charis_f1 - macro_f1

    lines = [
        SEP,
        "  CROSS-DATASET GENERALISATION RESULTS",
        SEP,
        "",
        "  Dataset sizes:",
        f"    CHARIS train : {len(X_train):,} windows ({tr_pats} patients, TBI ICU)",
        f"    CHARIS val   : {len(X_val):,} windows ({val_pats} patients, early-stop only)",
        f"    MIMIC test   : {len(X_test):,} windows ({mi_patients} patients, general ICU)",
        "",
        "  Label thresholds (same for both datasets):",
        "    Normal   = ICP < 15 mmHg",
        "    Elevated = 15-20 mmHg",
        "    Critical = >= 20 mmHg",
        "",
        "  Generalisation metrics (CHARIS -> MIMIC):",
        f"    Macro F1-score     : {macro_f1:.4f}",
        f"    Weighted F1-score  : {wt_f1:.4f}",
        f"    Balanced Accuracy  : {bal_acc:.4f}",
        "",
        "  Per-class F1 on MIMIC:",
        f"    Normal   (ICP<15)  : {per_class[0]:.4f}",
        f"    Elevated (15-20)   : {per_class[1]:.4f}",
        f"    Critical (>=20)    : {per_class[2]:.4f}",
        "",
        "  Domain gap:",
        f"    CHARIS internal F1 : {charis_f1:.4f}  (val set, CHARIS patients)",
        f"    MIMIC OOD F1       : {macro_f1:.4f}  (all MIMIC, never seen)",
        f"    Generalisation gap : {gap:+.4f}  ({'degraded' if gap > 0 else 'improved'} on MIMIC)",
        "",
        "  Full classification report (MIMIC test set):",
        "",
        report,
        SEP,
    ]

    output = "\n".join(lines)
    # Print safely on Windows (replace any stray non-ASCII)
    safe = output.encode("ascii", errors="replace").decode("ascii")
    print(safe)

    # Save (UTF-8 file is fine)
    report_path = out_dir / "cross_dataset_report.txt"
    report_path.write_text(output, encoding="utf-8")
    print(f"\n  Report saved -> {report_path}")

    import pickle
    model_path = out_dir / "xgboost_charis_only.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bst, fh)
    print(f"  Model  saved -> {model_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset eval: train on CHARIS, test on MIMIC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir",       type=Path, default=Path("results/cross_dataset"))
    args = parser.parse_args()

    for fname in ("features.npy", "labels.npy", "patient_ids.npy",
                  "mimic_features.npy", "mimic_labels.npy", "mimic_patient_ids.npy"):
        p = args.processed_dir / fname
        if not p.exists():
            print(f"ERROR: {p} not found. Run the feature-extraction pipelines first.")
            sys.exit(1)

    run(args.processed_dir, args.out_dir)


if __name__ == "__main__":
    main()
