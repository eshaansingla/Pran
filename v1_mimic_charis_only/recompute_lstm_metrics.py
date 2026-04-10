"""
recompute_lstm_metrics.py
=========================
Recompute LSTM metrics HONESTLY:
  1. Per-horizon AUC (t+1 through t+15)
  2. Per-sequence metrics (peak probability, not flattened)
  3. Flattened metrics (for comparison, labeled as inflated)

This fixes the #1 audit finding: flattened evaluation inflates effective N by 15×.

Usage:
    python recompute_lstm_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
SEQ_LEN = 30
HORIZON_MINS = 15
HORIZONS = np.array([6*i - 1 for i in range(1, HORIZON_MINS + 1)])
MAX_HORIZON = HORIZONS[-1]
KEEP = [0, 1, 2, 3, 4, 5]
BATCH_SIZE = 256


def load_data(processed_dir: Path):
    ch_feat = np.load(processed_dir / "features.npy").astype(np.float64)[:, KEEP]
    ch_lab  = np.load(processed_dir / "labels.npy").astype(np.int32)
    ch_pid  = np.load(processed_dir / "patient_ids.npy").astype(np.int32)

    mi_feat = np.load(processed_dir / "mimic_features.npy").astype(np.float64)[:, KEEP]
    mi_lab  = np.load(processed_dir / "mimic_labels.npy").astype(np.int32)
    mi_pid  = np.load(processed_dir / "mimic_patient_ids.npy").astype(np.int32)

    X   = np.vstack([ch_feat, mi_feat])
    y3  = np.concatenate([ch_lab, mi_lab])
    pid = np.concatenate([ch_pid, mi_pid])

    # Match LSTM training preprocessing
    X[:, 4] = X[:, 4] * 100
    X[:, 3] = (1.0 - X[:, 3]) * 100
    X[:, 3] = np.clip(X[:, 3], 0.05, 5.0)

    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X   = X[~nan_mask]
        y3  = y3[~nan_mask]
        pid = pid[~nan_mask]

    y = (y3 >= 1).astype(np.int32)
    return X, y, pid


def make_sequences(X, y, pid):
    seqs, labs, pids = [], [], []
    for p in np.unique(pid):
        mask  = pid == p
        feats = X[mask]
        lbls  = y[mask]
        n     = len(feats)
        if n < SEQ_LEN + MAX_HORIZON + 1:
            continue
        for i in range(n - SEQ_LEN - MAX_HORIZON):
            seqs.append(feats[i : i + SEQ_LEN])
            labs.append(lbls[i + SEQ_LEN + HORIZONS])
            pids.append(p)
    return (np.array(seqs, dtype=np.float32),
            np.array(labs, dtype=np.float32),
            np.array(pids, dtype=np.int32))


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


def fit_scaler(X_train):
    N, T, F = X_train.shape
    flat = X_train.reshape(-1, F)
    mean_ = flat.mean(axis=0).astype(np.float64)
    std_  = flat.std(axis=0).astype(np.float64)
    std_[std_ < 1e-8] = 1.0
    return mean_, std_


def apply_scaler(X, mean_, std_):
    return ((X - mean_) / std_).astype(np.float32)


def main():
    import tensorflow as tf
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    processed_dir = Path("data/processed")
    model_path = Path("models/lstm_forecast_v1.h5")
    meta_path = Path("models/lstm_meta.json")

    if not model_path.exists():
        print("ERROR: LSTM model not found. Train it first.")
        sys.exit(1)

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  LSTM METRIC RECOMPUTATION — Honest Per-Horizon + Per-Sequence")
    print(SEP)

    # Load and preprocess
    print("\n[1/5] Loading data ...")
    X, y, pid = load_data(processed_dir)
    print(f"  {len(X):,} windows, {len(np.unique(pid))} patients")

    print("\n[2/5] Creating sequences ...")
    X_seq, y_seq, p_seq = make_sequences(X, y, pid)
    print(f"  {len(X_seq):,} sequences")

    print("\n[3/5] Splitting ...")
    (X_tr, y_tr, _, X_va, y_va, _, X_te, y_te, pid_te) = patient_split(X_seq, y_seq, p_seq)
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")
    print(f"  Test patients: {len(np.unique(pid_te))}")

    mean_, std_ = fit_scaler(X_tr)
    X_te_s = apply_scaler(X_te, mean_, std_)

    # Load model
    print("\n[4/5] Loading model ...")
    meta = json.loads(meta_path.read_text())
    threshold = meta.get("threshold", 0.5)

    from tensorflow.keras.models import load_model
    model = load_model(str(model_path), compile=False)
    print(f"  Threshold: {threshold}")

    # Predict
    probs = model.predict(X_te_s, batch_size=BATCH_SIZE, verbose=0)  # (N, 15)
    preds = (probs >= threshold).astype(int)

    # =============================================
    # METHOD 1: Flattened (old, inflated)
    # =============================================
    print(f"\n{SEP}")
    print("  METHOD 1: FLATTENED (old method — inflated N)")
    print(SEP)
    p_flat = probs.ravel()
    d_flat = preds.ravel()
    y_flat = y_te.ravel()
    flat_auc = roc_auc_score(y_flat, p_flat)
    flat_f1  = f1_score(y_flat, d_flat, zero_division=0)
    flat_prec = precision_score(y_flat, d_flat, zero_division=0)
    flat_rec = recall_score(y_flat, d_flat, zero_division=0)
    print(f"  AUC:  {flat_auc:.4f}")
    print(f"  F1:   {flat_f1:.4f}")
    print(f"  Prec: {flat_prec:.4f}")
    print(f"  Rec:  {flat_rec:.4f}")
    print(f"  N (inflated): {len(y_flat):,} = {len(y_te):,} seqs × 15 horizons")

    # =============================================
    # METHOD 2: Per-horizon AUC
    # =============================================
    print(f"\n{SEP}")
    print("  METHOD 2: PER-HORIZON AUC (honest)")
    print(SEP)
    print(f"  {'Horizon':<12} {'AUC':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'-'*48}")
    per_horizon = []
    for h in range(HORIZON_MINS):
        mins = h + 1
        auc_h = roc_auc_score(y_te[:, h], probs[:, h])
        pred_h = (probs[:, h] >= threshold).astype(int)
        f1_h  = f1_score(y_te[:, h], pred_h, zero_division=0)
        prec_h = precision_score(y_te[:, h], pred_h, zero_division=0)
        rec_h = recall_score(y_te[:, h], pred_h, zero_division=0)
        per_horizon.append({
            "horizon_min": mins,
            "auc": round(float(auc_h), 4),
            "f1": round(float(f1_h), 4),
            "precision": round(float(prec_h), 4),
            "recall": round(float(rec_h), 4),
        })
        marker = " ◄" if mins in [1, 5, 10, 15] else ""
        print(f"  t+{mins:>2} min     {auc_h:>7.4f}  {f1_h:>7.4f}  {prec_h:>7.4f}  {rec_h:>7.4f}{marker}")

    # =============================================
    # METHOD 3: Per-sequence (peak horizon)
    # =============================================
    print(f"\n{SEP}")
    print("  METHOD 3: PER-SEQUENCE (honest — 1 decision per sequence)")
    print(SEP)
    # Peak probability across horizons
    seq_prob_peak = probs.max(axis=1)           # (N,)
    seq_prob_mean = probs.mean(axis=1)          # (N,)
    seq_label = y_te.max(axis=1).astype(int)    # 1 if ANY horizon is abnormal

    seq_pred_peak = (seq_prob_peak >= threshold).astype(int)
    seq_auc  = roc_auc_score(seq_label, seq_prob_peak)
    seq_f1   = f1_score(seq_label, seq_pred_peak, zero_division=0)
    seq_prec = precision_score(seq_label, seq_pred_peak, zero_division=0)
    seq_rec  = recall_score(seq_label, seq_pred_peak, zero_division=0)
    seq_bacc = balanced_accuracy_score(seq_label, seq_pred_peak)

    print(f"  Strategy: peak P(Abnormal) across 15 horizons → single prediction")
    print(f"  N (honest): {len(seq_label):,} sequences")
    print(f"  AUC:  {seq_auc:.4f}")
    print(f"  F1:   {seq_f1:.4f}")
    print(f"  Prec: {seq_prec:.4f}")
    print(f"  Rec:  {seq_rec:.4f}")
    print(f"  BAcc: {seq_bacc:.4f}")

    # Also mean-probability approach
    seq_pred_mean = (seq_prob_mean >= threshold).astype(int)
    seq_auc_mean  = roc_auc_score(seq_label, seq_prob_mean)
    seq_f1_mean   = f1_score(seq_label, seq_pred_mean, zero_division=0)
    print(f"\n  (Alt: mean-probability approach: AUC={seq_auc_mean:.4f}, F1={seq_f1_mean:.4f})")

    # =============================================
    # Summary comparison
    # =============================================
    print(f"\n{SEP}")
    print("  SUMMARY COMPARISON")
    print(SEP)
    print(f"  {'Method':<30} {'AUC':>7}  {'F1':>7}  {'N':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Flattened (old, inflated)':<30} {flat_auc:>7.4f}  {flat_f1:>7.4f}  {len(y_flat):>10,}")
    print(f"  {'Per-sequence (peak prob)':<30} {seq_auc:>7.4f}  {seq_f1:>7.4f}  {len(seq_label):>10,}")
    print(f"  {'Per-sequence (mean prob)':<30} {seq_auc_mean:>7.4f}  {seq_f1_mean:>7.4f}  {len(seq_label):>10,}")
    print(f"  {'Per-horizon t+1min':<30} {per_horizon[0]['auc']:>7.4f}  {per_horizon[0]['f1']:>7.4f}  {len(y_te):>10,}")
    print(f"  {'Per-horizon t+15min':<30} {per_horizon[-1]['auc']:>7.4f}  {per_horizon[-1]['f1']:>7.4f}  {len(y_te):>10,}")
    print(SEP)

    # Save results
    out = {
        "evaluation_method": "honest_recomputation",
        "test_sequences": int(len(y_te)),
        "test_patients": int(len(np.unique(pid_te))),
        "threshold": threshold,
        "flattened_metrics": {
            "auc": round(float(flat_auc), 4),
            "f1": round(float(flat_f1), 4),
            "precision": round(float(flat_prec), 4),
            "recall": round(float(flat_rec), 4),
            "note": "INFLATED — treats 15 correlated horizon outputs as independent"
        },
        "per_sequence_metrics": {
            "strategy": "peak_probability_across_horizons",
            "auc": round(float(seq_auc), 4),
            "f1": round(float(seq_f1), 4),
            "precision": round(float(seq_prec), 4),
            "recall": round(float(seq_rec), 4),
            "balanced_accuracy": round(float(seq_bacc), 4),
            "note": "HONEST — 1 decision per sequence"
        },
        "per_horizon_metrics": per_horizon,
    }
    out_path = Path("results/lstm/honest_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {out_path}")

    # Update lstm_meta.json with honest metrics
    meta["honest_metrics"] = {
        "per_sequence_auc": round(float(seq_auc), 4),
        "per_sequence_f1": round(float(seq_f1), 4),
        "per_horizon_auc_t1": per_horizon[0]["auc"],
        "per_horizon_auc_t15": per_horizon[-1]["auc"],
        "evaluation_note": "Per-sequence: peak prob across 15 horizons → 1 decision. Flattened metrics are inflated."
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Updated → {meta_path}")


if __name__ == "__main__":
    main()
