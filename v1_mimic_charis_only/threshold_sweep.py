"""
threshold_sweep.py — show F1/Precision/Recall at different decision thresholds
using the same data split as train_binary.py (SEED=42, dataset-stratified).
"""
import gzip, pickle, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

SEED = 42
base = Path(__file__).parent
proc = base.parent / "data" / "processed"

# ---- load same data as train_binary.py ----
KEEP = [0, 1, 2, 3, 4, 5]
ch_feat = np.load(proc / "features.npy").astype(np.float32)[:, KEEP]
ch_lab  = np.load(proc / "labels.npy").astype(np.int64)
ch_pid  = np.load(proc / "patient_ids.npy").astype(np.int32)
mi_feat = np.load(proc / "mimic_features.npy").astype(np.float32)[:, KEEP]
mi_lab  = np.load(proc / "mimic_labels.npy").astype(np.int64)
mi_pid  = np.load(proc / "mimic_patient_ids.npy").astype(np.int32)

X   = np.vstack([ch_feat, mi_feat])
y3  = np.concatenate([ch_lab, mi_lab])
pid = np.concatenate([ch_pid, mi_pid])
y   = (y3 >= 1).astype(np.int64)

# ---- replicate patient_split ----
def _split_cohort(X, y, pid):
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(X, y, groups=pid))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr, va = next(gss2.split(X[tv], y[tv], groups=pid[tv]))
    return tv[tr], tv[va], te

ch = pid <= 13
mi = ~ch
ch_idx = np.where(ch)[0]
mi_idx = np.where(mi)[0]

ch_tr, ch_va, ch_te = _split_cohort(X[ch], y[ch], pid[ch])
mi_tr, mi_va, mi_te = _split_cohort(X[mi], y[mi], pid[mi])

te = np.concatenate([ch_idx[ch_te], mi_idx[mi_te]])
X_te = X[te]
y_te = y[te]

print(f"Test set: {len(y_te):,} windows, {(y_te==1).sum():,} abnormal ({100*(y_te==1).mean():.1f}%)")

# ---- load model ----
with gzip.open(base / "models/xgboost_binary.pkl.gz", "rb") as f:
    model = pickle.load(f)

import xgboost as xgb
dtest = xgb.DMatrix(X_te)
probs = model.predict(dtest)

auc = roc_auc_score(y_te, probs)
print(f"Test AUC: {auc:.4f}\n")

print(f"{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Specificity':>12} {'BalAcc':>8}")
print("-" * 66)

for t in [0.25, 0.30, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_te, preds, zero_division=0)
    pr = precision_score(y_te, preds, zero_division=0)
    rc = recall_score(y_te, preds, zero_division=0)
    cm = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bacc = (rc + spec) / 2
    note = ""
    if abs(t - 0.33) < 0.005:
        note = "  <-- trained (Youden+recall>=0.75)"
    elif abs(t - 0.50) < 0.005:
        note = "  <-- default (0.5)"
    print(f"{t:>10.2f} {f1:>8.4f} {pr:>10.4f} {rc:>8.4f} {spec:>12.4f} {bacc:>8.4f}{note}")

print()
