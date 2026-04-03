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
    ch_feat = np.load(processed_dir / "features.npy").astype(np.float32)[:, :8]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int64)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float32)
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int64)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)
    if mi_feat.shape[1] > 8:
        mi_feat = mi_feat[:, :8]

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
    ch = pid <= 13
    mi = ~ch
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
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(bst, X_tr, y_tr, X_te, y_te, out_dir: Path):
    dtr = xgb.DMatrix(X_tr)
    dte = xgb.DMatrix(X_te)

    prob_tr = bst.predict(dtr)
    prob_te = bst.predict(dte)

    pred_tr = (prob_tr >= 0.5).astype(int)
    pred_te = (prob_te >= 0.5).astype(int)

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

    SEP = "=" * 60
    lines = [
        SEP,
        "  BINARY CLASSIFICATION RESULTS",
        "  Normal (<15 mmHg)  vs  Abnormal (>=15 mmHg)",
        SEP,
        "",
        "  Test-set metrics:",
        f"    F1-score        : {f1_te:.4f}   (target >= 0.80)",
        f"    AUC             : {auc_te:.4f}   (target >= 0.90)",
        f"    Precision       : {prec_te:.4f}",
        f"    Recall (Sens.)  : {rec_te:.4f}",
        f"    Specificity     : {spec_te:.4f}",
        f"    Balanced Acc.   : {bacc_te:.4f}",
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

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#2C5282", lw=2,
                 label=f"AUC = {auc_te:.4f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Binary ICP Classifier")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={"shrink":0.8}, linewidths=0.5)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix — Test Set")

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
    print(f"  Total: {len(X):,} windows | {len(np.unique(pid))} patients | 8 features")
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

    print("\n[4/5] Evaluating ...")
    metrics = evaluate(bst, X_tr, y_tr, X_te, y_te, out_dir)

    print("[5/5] Saving model ...")
    Path("models").mkdir(exist_ok=True)

    pkl_path = Path("models/xgboost_binary.pkl")
    gz_path  = Path("models/xgboost_binary.pkl.gz")

    with open(pkl_path, "wb") as f:
        pickle.dump(bst, f, protocol=5)
    with gzip.open(gz_path, "wb", compresslevel=6) as f:
        pickle.dump(bst, f, protocol=5)

    size_kb = gz_path.stat().st_size / 1024
    flag = "OK" if size_kb < 512 else "EXCEEDS LIMIT"
    print(f"  models/xgboost_binary.pkl.gz  ({size_kb:.1f} KB) [{flag}]")
    print(f"  results saved -> {out_dir}/\n")

    # Store scale_pos_weight for API
    meta = {"scale_pos_weight": spw, "threshold": 15.0, "metrics": metrics}
    import json
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
