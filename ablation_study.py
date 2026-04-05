"""
ablation_study.py
=================
Retrain the binary ICP classifier 8 times, each time dropping one feature.
Reports F1 drop vs baseline to validate feature importance rankings.

Usage:
    python ablation_study.py
    python ablation_study.py --processed_dir data/processed
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

SEED = 42
FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
]


# ---------------------------------------------------------------------------
# Data helpers (same as train_binary.py)
# ---------------------------------------------------------------------------

KEEP = [0, 1, 2, 3, 4, 5]   # drop head_angle (6), motion_artifact_flag (7)

def load_binary(processed_dir: Path):
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
    return (X[tr], y[tr], pid[tr],
            X[va], y[va], pid[va],
            X[te], y[te], pid[te])


# ---------------------------------------------------------------------------
# Train + evaluate (lightweight — no calibration for ablation speed)
# ---------------------------------------------------------------------------

def train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te):
    n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(n0 / n1, 4)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    dte    = xgb.DMatrix(X_te)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "eta":              0.1,
        "max_depth":        4,
        "min_child_weight": 5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "lambda":           1.0,
        "alpha":            0.1,
        "seed":             SEED,
        "tree_method":      "hist",
        "verbosity":        0,
    }

    bst = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=0,
    )

    probs = bst.predict(dte)
    preds = (probs >= 0.5).astype(int)
    f1  = f1_score(y_te, preds, zero_division=0)
    auc = roc_auc_score(y_te, probs)
    ba  = balanced_accuracy_score(y_te, preds)
    return round(f1, 4), round(auc, 4), round(ba, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    print("Loading data...")
    X, y, pid = load_binary(processed_dir)
    X_tr, y_tr, pid_tr, X_va, y_va, pid_va, X_te, y_te, pid_te = patient_split(X, y, pid)

    print(f"Train: {len(y_tr):,}  Val: {len(y_va):,}  Test: {len(y_te):,}\n")

    # --- Baseline (all 8 features) ---
    print("Training baseline (all 8 features)...")
    base_f1, base_auc, base_ba = train_eval(X_tr, y_tr, X_va, y_va, X_te, y_te)
    print(f"  Baseline  F1={base_f1:.4f}  AUC={base_auc:.4f}  BalAcc={base_ba:.4f}\n")

    # --- Real gain importances from production model ---
    model_path = Path("models/xgboost_binary.pkl.gz")
    if model_path.exists():
        with gzip.open(model_path, "rb") as fh:
            prod_bst = pickle.load(fh)
        raw = prod_bst.get_score(importance_type="gain")
        fi_map = {f"f{i}": name for i, name in enumerate(FEATURE_NAMES)}
        total_gain = sum(raw.values()) or 1.0
        gain_pct = {fi_map.get(k, k): round(v / total_gain * 100, 2) for k, v in raw.items()}
    else:
        gain_pct = {}

    # --- Ablation: drop one feature at a time ---
    results = []
    for i, feat_name in enumerate(FEATURE_NAMES):
        keep = [j for j in range(6) if j != i]
        print(f"  Dropping {feat_name} ...")
        f1, a, ba = train_eval(
            X_tr[:, keep], y_tr, X_va[:, keep], y_va, X_te[:, keep], y_te
        )
        delta_f1 = round(base_f1 - f1, 4)
        results.append({
            "feature":    feat_name,
            "gain_pct":   gain_pct.get(feat_name, 0.0),
            "ablation_f1":    f1,
            "ablation_auc":   a,
            "ablation_ba":    ba,
            "f1_drop":    delta_f1,
        })
        print(f"    F1={f1:.4f}  AUC={a:.4f}  BalAcc={ba:.4f}  delta_F1={delta_f1:+.4f}")

    # Sort by F1 drop
    results.sort(key=lambda r: -r["f1_drop"])

    # --- Report ---
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS — Binary ICP Classifier v2.1")
    print("=" * 70)
    print(f"{'Feature':<26} {'Gain%':>6}  {'F1':>6}  {'AUC':>6}  {'BalAcc':>7}  {'F1 Drop':>8}")
    print("-" * 70)
    print(f"{'[Baseline — all 8]':<26} {'—':>6}  {base_f1:>6.4f}  {base_auc:>6.4f}  {base_ba:>7.4f}  {'—':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['feature']:<26} {r['gain_pct']:>5.1f}%  "
            f"{r['ablation_f1']:>6.4f}  {r['ablation_auc']:>6.4f}  "
            f"{r['ablation_ba']:>7.4f}  {r['f1_drop']:>+8.4f}"
        )
    print("=" * 70)

    # Save JSON for documentation
    out = {
        "baseline": {"f1": base_f1, "auc": base_auc, "balanced_accuracy": base_ba},
        "gain_importances": gain_pct,
        "ablation": results,
    }
    out_path = Path("results/binary/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
