"""Diagnose the recall problem — check raw vs calibrated at various thresholds."""
import numpy as np
import pickle, gzip
import xgboost as xgb
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, balanced_accuracy_score
from pathlib import Path

# Load model and data
KEEP = [0, 1, 2, 3, 4, 5]
processed = Path("data/processed")

ch_feat = np.load(processed / "features.npy").astype(np.float32)[:, KEEP]
ch_lab  = (np.load(processed / "labels.npy") >= 1).astype(np.int64)
ch_pid  = np.load(processed / "patient_ids.npy").astype(np.int32)

mi_feat = np.load(processed / "mimic_features.npy").astype(np.float32)[:, KEEP]
mi_lab  = (np.load(processed / "mimic_labels.npy") >= 1).astype(np.int64)
mi_pid  = np.load(processed / "mimic_patient_ids.npy").astype(np.int32)

X = np.vstack([ch_feat, mi_feat])
y = np.concatenate([ch_lab, mi_lab])
pid = np.concatenate([ch_pid, mi_pid])

# Load model
with gzip.open("models/xgboost_binary.pkl.gz", "rb") as f:
    bst = pickle.load(f)

# Load calibrator
with gzip.open("models/xgboost_binary_calibrator.pkl.gz", "rb") as f:
    cal = pickle.load(f)

# Do same split as training
from sklearn.model_selection import GroupShuffleSplit
SEED = 42
ch = pid <= 13; mi = ~ch
ch_idx = np.where(ch)[0]; mi_idx = np.where(mi)[0]

def _split(X,y,pid):
    g1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(g1.split(X, y, groups=pid))
    g2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr, va = next(g2.split(X[tv], y[tv], groups=pid[tv]))
    return tv[tr], tv[va], te

ch_tr, ch_va, ch_te = _split(X[ch], y[ch], pid[ch])
mi_tr, mi_va, mi_te = _split(X[mi], y[mi], pid[mi])
te_idx = np.concatenate([ch_idx[ch_te], mi_idx[mi_te]])

X_te, y_te, pid_te = X[te_idx], y[te_idx], pid[te_idx]
print(f"Test set: {len(y_te):,} windows, {len(np.unique(pid_te))} patients")
print(f"Test labels: Normal={int((y_te==0).sum())}, Abnormal={int((y_te==1).sum())} ({100*y_te.mean():.1f}%)")

dte = xgb.DMatrix(X_te)
raw_probs = bst.predict(dte)
cal_probs = cal.predict(raw_probs)

print(f"\nRaw probs range: [{raw_probs.min():.4f}, {raw_probs.max():.4f}], mean={raw_probs.mean():.4f}")
print(f"Cal probs range: [{cal_probs.min():.4f}, {cal_probs.max():.4f}], mean={cal_probs.mean():.4f}")

print(f"\n{'Threshold':>10}  {'Source':>10}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'BAcc':>6}")
print("-" * 60)
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    for name, probs in [("raw", raw_probs), ("calib", cal_probs)]:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_te, preds, zero_division=0)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        bacc = balanced_accuracy_score(y_te, preds)
        print(f"{t:>10.2f}  {name:>10}  {f1:>6.4f}  {prec:>6.4f}  {rec:>6.4f}  {bacc:>6.4f}")
    print()
