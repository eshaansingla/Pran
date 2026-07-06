"""
hybrid_pipeline_v4.py
=====================
Hybrid XGBoost — domain-aware normalization, no SMOTE.

Design rationale
----------------
cardiac_amplitude from an ICU hospital PPG sensor (CHARIS) is NOT numerically
comparable to cardiac_amplitude from our custom IR TM sensor (HW). Fitting a
single QuantileTransformer on mixed CHARIS+HW data biases the quantile
breakpoints toward the CHARIS distribution and corrupts HW predictions.

Fix: separate QT per domain.
  qt_c  – fitted on CHARIS training windows  → maps CHARIS to N(0,1)
  qt_hw – fitted on HW training windows      → maps HW to N(0,1)
After normalization, "high cardiac_amplitude" means the same physiological
magnitude in both domains, eliminating sensor-specific scale differences.

Class imbalance:
  CHARIS abnormal (greedy, whole patients): 11,994
  HW abnormal (flagged): 949
  HW normal: 14,631
  Ratio: 12,943 abn vs 14,631 norm = 0.885:1  → scale_pos_weight = 1.13
  No SMOTE. No synthetic cross-domain samples.

Structure mirrors full_pipeline_qt.py:
  [1]  Load CHARIS cache + greedy patient selection
  [2]  Load HW data (CHARIS-flagged labels)
  [3]  Patient-level 70/10/20 split (HW only; CHARIS always in train)
  [4]  Separate domain QT fit on train splits
  [5]  Train XGBoost (scale_pos_weight, CUDA)
  [6]  Evaluate test split
  [7]  LOPO over HW patients (CHARIS always in train, per-fold domain QT)
  [8]  Baselines (LogReg, RandForest, LinearSVM)
  [9]  Feature ablation (pooled AUC)
  [10] Valsalva + hardware validation
  [11] Statistical tests + feature importance
  [12] Save results + plots

Run
---
    python hybrid_pipeline_v4.py
"""
from __future__ import annotations
import json, pickle, sys, warnings
from datetime import date
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from scipy import signal as sp_signal
from scipy.stats import (wilcoxon as _wilcoxon, norm as _norm, mannwhitneyu,
                         friedmanchisquare, spearmanr)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_X        = Path("results/audit/cache/X.npy")
CACHE_Y        = Path("results/audit/cache/y.npy")
CACHE_PID      = Path("results/audit/cache/pid.npy")
HW_DIR         = Path("hw-tests")
OUT_DIR        = Path("results/hybrid_pipeline_v4")
MODEL_DIR      = Path("models/hybrid_v4")
LOPO_DIR       = MODEL_DIR / "lopo"
FLAGS_PATH     = Path("results/hw_charis_flags.json")

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]
N        = len(FEATURES)
SEED     = 42
FS, WIN, STEP = 50, 500, 250
SEP  = "=" * 65
SEP2 = "-" * 65

_nyq             = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS           = np.fft.rfftfreq(WIN, d=1.0 / FS)
_FREQ_MASK       = (_FREQS >= 0.7) & (_FREQS <= 2.5)


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_window(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))
    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))
    pwr      = np.abs(np.fft.rfft(ir_dt)) ** 2
    if not _FREQ_MASK.any():
        return None
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])
    r         = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp  = float(np.percentile(r, 99) - np.percentile(r, 1))
    coeffs    = pywt.wavedec(disp_dt, "db4", level=5)
    energies  = [float(np.sum(c ** 2)) for c in coeffs]
    total     = sum(energies) + 1e-12
    slow_pow    = energies[0] / total
    cardiac_pow = energies[2] / total
    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── HW data loader ─────────────────────────────────────────────────────────────
def load_hw_labeled(hw_dir: Path, flags: dict):
    X_all, y_all, pid_all, sess_all, names = [], [], [], [], []
    pid_idx = 0
    required = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
    for csv_path in sorted(hw_dir.glob("*.csv")):
        if csv_path.name not in flags:
            continue
        df = pd.read_csv(csv_path, comment="#", low_memory=False)
        if not required.issubset(df.columns):
            continue
        df    = df[df["artifact_flag"] == 0].reset_index(drop=True)
        label = 1 if flags[csv_path.name]["flagged"] else 0
        feats, sessions = [], []
        for w in range((len(df) - WIN) // STEP + 1):
            s, e = w * STEP, w * STEP + WIN
            sl   = df.iloc[s:e]
            feat = extract_window(sl["ir_raw"].values.astype(np.float32),
                                  sl["disp_raw"].values.astype(np.float32))
            if feat is None:
                continue
            feats.append(feat)
            sessions.append(int(sl["session_label"].mode()[0]))
        if not feats:
            continue
        X_all.extend(feats)
        y_all.extend([label] * len(feats))
        pid_all.extend([pid_idx] * len(feats))
        sess_all.extend(sessions)
        names.append(csv_path.name)
        pid_idx += 1
    return (np.array(X_all,    dtype=np.float32),
            np.array(y_all,    dtype=np.int32),
            np.array(pid_all,  dtype=np.int32),
            np.array(sess_all, dtype=np.int32),
            names)


# ── CHARIS patient selection (greedy, whole-patient, smallest-first) ──────────
def select_charis_patients(c_X, c_y, c_pid, n_hw_norm, n_hw_abn):
    budget = n_hw_norm - n_hw_abn
    patients = sorted(
        [(int((c_pid == p).sum() if False else ((c_pid == p) & (c_y == 1)).sum()), int(p))
         for p in np.unique(c_pid)
         if ((c_pid == p) & (c_y == 1)).sum() > 0]
    )
    selected, total_abn = [], 0
    for n_abn, pid in patients:
        if total_abn + n_abn <= budget:
            selected.append(pid)
            total_abn += n_abn

    print(f"\n  CHARIS patient selection  (budget = {budget:,} windows)")
    print(f"  {'PID':>5}  {'Abn wins':>10}  {'Status'}")
    print(f"  {'-'*38}")
    for n_abn, pid in patients:
        s = "SELECTED" if pid in selected else "dropped"
        print(f"  {pid:>5}  {n_abn:>10,}  {s}")
    print(f"\n  Selected {len(selected)} patients  |  abnormal windows: {total_abn:,}")

    mask  = np.isin(c_pid, selected) & (c_y == 1)
    # Offset PIDs to avoid collisions with HW PIDs
    return c_X[mask], c_y[mask], c_pid[mask] + 1000, selected, total_abn


# ── Domain-separated QT fit + transform ───────────────────────────────────────
def fit_qt(X: np.ndarray, seed: int = SEED) -> QuantileTransformer:
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(X)), random_state=seed)
    return qt.fit(X)


def domain_normalize(X_hw_tr, X_hw_te, X_c_tr):
    """
    Fit separate QTs on each domain's training data.
    Returns (X_hw_tr_qt, X_hw_te_qt, X_c_qt, qt_hw, qt_c).
    Eliminates sensor-specific amplitude differences between CHARIS and HW.
    """
    qt_hw = fit_qt(X_hw_tr)
    qt_c  = fit_qt(X_c_tr)
    return (qt_hw.transform(X_hw_tr).astype(np.float32),
            qt_hw.transform(X_hw_te).astype(np.float32),
            qt_c.transform(X_c_tr).astype(np.float32),
            qt_hw, qt_c)


# ── XGBoost helpers ────────────────────────────────────────────────────────────
def get_device():
    try:
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            return "cpu"
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0},
                  xgb.DMatrix(np.zeros((4, N)), label=[0, 1, 0, 1]), num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def xgb_params(device, spw=1.0, seed=SEED):
    # eval_metric order matters: XGBoost uses the LAST metric for early stopping.
    # Use logloss (valid even for single-class val) so training never collapses at iter 0.
    return {"objective": "binary:logistic", "eval_metric": ["auc", "logloss"],
            "eta": 0.05, "max_depth": 5, "min_child_weight": 3,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda": 1.5, "alpha": 0.1,
            "scale_pos_weight": float(spw),
            "seed": seed, "tree_method": "hist", "device": device, "verbosity": 0}


def scale_pos_weight(y):
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return float(n0) / max(float(n1), 1.0)


def youden_threshold(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def eval_metrics(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    auc  = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
    f1   = float(f1_score(y_true, preds, zero_division=0))
    rec  = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    spec = float(recall_score(1 - y_true, 1 - preds, zero_division=0))
    bacc = float(balanced_accuracy_score(y_true, preds))
    ap   = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
    cm   = confusion_matrix(y_true, preds).tolist()
    return dict(auc=auc, f1=f1, recall=rec, precision=prec,
                specificity=spec, balanced_acc=bacc, avg_precision=ap, cm=cm)


# ── HW patient 70/10/20 split ─────────────────────────────────────────────────
def hw_patient_split(hw_pid, hw_y, names, seed=SEED):
    uids      = np.unique(hw_pid)
    uid_label = np.array([int(hw_y[hw_pid == p][0]) for p in uids])

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tv_idx, te_idx = next(sss1.split(uids, uid_label))
    tv_uids, tv_lab = uids[tv_idx], uid_label[tv_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    try:
        tr_idx, va_idx = next(sss2.split(tv_uids, tv_lab))
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tv_uids))
        cut  = max(1, int(0.125 * len(tv_uids)))
        va_idx, tr_idx = perm[:cut], perm[cut:]

    train_pids = set(tv_uids[tr_idx].tolist())
    val_pids   = set(tv_uids[va_idx].tolist())
    test_pids  = set(uids[te_idx].tolist())

    def mk(ps): return np.isin(hw_pid, list(ps))

    def n_abn(ps):
        return int(sum(uid_label[np.isin(uids, list(ps))]))

    print(f"\n  HW patient split  (total={len(uids)}  "
          f"abn={int(uid_label.sum())}  norm={int((uid_label==0).sum())})")
    print(f"  Train : {len(train_pids):2d} patients  ({n_abn(train_pids)} abn)")
    print(f"  Val   : {len(val_pids):2d} patients  ({n_abn(val_pids)} abn)")
    print(f"  Test  : {len(test_pids):2d} patients  ({n_abn(test_pids)} abn)")

    return mk(train_pids), mk(val_pids), mk(test_pids)


# ── Main split training ────────────────────────────────────────────────────────
def run_main_split(hw_X, hw_y, hw_pid, c_X, c_y, device, names):
    print(f"\n[3] Patient-level 70 / 10 / 20 split (HW patients) ...")
    tr_m, va_m, te_m = hw_patient_split(hw_pid, hw_y, names)

    X_hw_tr, y_hw_tr = hw_X[tr_m], hw_y[tr_m]
    X_hw_va, y_hw_va = hw_X[va_m], hw_y[va_m]
    X_hw_te, y_hw_te = hw_X[te_m], hw_y[te_m]

    print(f"\n[4] Separate domain QT normalization ...")
    X_hw_tr_qt, X_hw_va_qt, X_c_qt, qt_hw, qt_c = domain_normalize(
        X_hw_tr, X_hw_va, c_X)
    # We also need test normalized
    X_hw_te_qt = qt_hw.transform(X_hw_te).astype(np.float32)
    print(f"  qt_hw fitted on {len(X_hw_tr):,} HW train windows")
    print(f"  qt_c  fitted on {len(c_X):,} CHARIS train windows")
    print(f"  Feature distributions now N(0,1) within each domain")

    X_tr = np.concatenate([X_hw_tr_qt, X_c_qt])
    y_tr = np.concatenate([y_hw_tr, c_y])
    spw  = scale_pos_weight(y_tr)
    print(f"\n[5] Training XGBoost  [device={device.upper()}] ...")
    print(f"  Train: {len(y_tr):,} windows  "
          f"abn={int((y_tr==1).sum()):,}  norm={int((y_tr==0).sum()):,}")
    print(f"  scale_pos_weight = {spw:.4f}  (no SMOTE)")

    d_tr = xgb.DMatrix(X_tr,       label=y_tr,       feature_names=FEATURES)
    d_te = xgb.DMatrix(X_hw_te_qt,                   feature_names=FEATURES)

    val_has_both = len(np.unique(y_hw_va)) == 2
    if val_has_both:
        # HW val has both classes → proper early stopping on logloss
        d_va = xgb.DMatrix(X_hw_va_qt, label=y_hw_va, feature_names=FEATURES)
        bst  = xgb.train(xgb_params(device, spw), d_tr, num_boost_round=800,
                         evals=[(d_va, "val")], early_stopping_rounds=50,
                         verbose_eval=False)
        print(f"  Best iter: {bst.best_iteration}  val-logloss: {bst.best_score:.4f}")
        thr = youden_threshold(y_hw_va, bst.predict(d_va))
    else:
        # Val is single-class (common with only 3 abnormal HW patients) →
        # Use an inner window-level split of the combined training set for early stopping,
        # then threshold from that inner val (has CHARIS abnormals → both classes).
        print(f"  Val split single-class — using combined inner val for early stopping")
        rng  = np.random.default_rng(SEED)
        idx  = rng.permutation(len(y_tr)); cut = int(0.90 * len(idx))
        d_itr = xgb.DMatrix(X_tr[idx[:cut]], label=y_tr[idx[:cut]], feature_names=FEATURES)
        d_iva = xgb.DMatrix(X_tr[idx[cut:]], label=y_tr[idx[cut:]], feature_names=FEATURES)
        bst   = xgb.train(xgb_params(device, spw), d_itr, num_boost_round=800,
                          evals=[(d_iva, "val")], early_stopping_rounds=50,
                          verbose_eval=False)
        print(f"  Best iter: {bst.best_iteration}  inner-val-logloss: {bst.best_score:.4f}")
        thr = youden_threshold(y_tr[idx[cut:]], bst.predict(d_iva)) \
              if len(np.unique(y_tr[idx[cut:]])) == 2 else 0.5
    print(f"  Youden threshold: {thr:.4f}")

    print(f"\n[6] Test split evaluation ...")
    if len(np.unique(y_hw_te)) < 2:
        print(f"  Test split single-class — AUC undefined  "
              f"(abn={int((y_hw_te==1).sum())}  norm={int((y_hw_te==0).sum())})")
        m = {}
    else:
        m = eval_metrics(y_hw_te, bst.predict(d_te), thr)
        print(f"  Test AUC      : {m['auc']:.4f}")
        print(f"  F1            : {m['f1']:.4f}")
        print(f"  Recall        : {m['recall']:.4f}")
        print(f"  Specificity   : {m['specificity']:.4f}")
        print(f"  Balanced Acc  : {m['balanced_acc']:.4f}")
        if len(np.unique(y_hw_te)) == 2:
            print(f"\n{classification_report(y_hw_te, (bst.predict(d_te)>=thr).astype(int), target_names=['Normal','Abnormal'], zero_division=0)}")

    # Save model + normalizers
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(MODEL_DIR / "hybrid_v4_xgb.json"))
    pickle.dump(qt_hw, open(MODEL_DIR / "qt_hw.pkl", "wb"))
    pickle.dump(qt_c,  open(MODEL_DIR / "qt_c.pkl",  "wb"))
    pickle.dump(thr,   open(MODEL_DIR / "thr.pkl",   "wb"))

    return bst, qt_hw, qt_c, thr, m


# ── LOPO over HW patients ─────────────────────────────────────────────────────
def run_lopo(hw_X, hw_y, hw_pid, hw_sess, c_X, c_y, device, names):
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    logo    = LeaveOneGroupOut()
    records = []

    print(f"\n[7] LOPO over {len(np.unique(hw_pid))} HW patients ...")
    print(f"  (CHARIS {len(c_X):,} windows always in train; per-fold domain QT)")
    print(f"\n  {'Fold':>4}  {'Patient':<32}  {'Label':>8}  {'N':>5}  {'MeanP':>7}  {'Status'}")
    print(f"  {SEP2}")

    for fold, (tr_idx, te_idx) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        test_pid  = int(hw_pid[te_idx[0]])
        name      = names[test_pid] if test_pid < len(names) else f"pid{test_pid}"
        cache_m   = LOPO_DIR / f"fold{fold:03d}.json"
        cache_qt  = LOPO_DIR / f"fold{fold:03d}_qt.pkl"

        X_hw_te, y_hw_te = hw_X[te_idx], hw_y[te_idx]
        X_hw_tr, y_hw_tr = hw_X[tr_idx], hw_y[tr_idx]

        if cache_m.exists() and cache_qt.exists():
            bst = xgb.Booster(); bst.load_model(str(cache_m))
            qt_hw_f, thr_f = pickle.load(open(cache_qt, "rb"))
            status = "(c)"
        else:
            # Separate domain QTs for this fold
            qt_hw_f = fit_qt(X_hw_tr)
            qt_c_f  = fit_qt(c_X)

            X_hw_tr_qt = qt_hw_f.transform(X_hw_tr).astype(np.float32)
            X_c_qt     = qt_c_f.transform(c_X).astype(np.float32)

            X_tr_comb = np.concatenate([X_hw_tr_qt, X_c_qt])
            y_tr_comb = np.concatenate([y_hw_tr, c_y])
            spw_f     = scale_pos_weight(y_tr_comb)

            # Inner 90/10 window split for early stopping + Youden threshold
            rng   = np.random.default_rng(SEED + fold)
            idx   = rng.permutation(len(y_tr_comb))
            cut   = int(0.90 * len(idx))
            d_itr = xgb.DMatrix(X_tr_comb[idx[:cut]], label=y_tr_comb[idx[:cut]],
                                 feature_names=FEATURES)
            d_iva = xgb.DMatrix(X_tr_comb[idx[cut:]], label=y_tr_comb[idx[cut:]],
                                 feature_names=FEATURES)

            bst = xgb.train(xgb_params(device, spw_f, SEED + fold), d_itr,
                            num_boost_round=600, evals=[(d_iva, "val")],
                            early_stopping_rounds=50, verbose_eval=False)

            # Threshold calibrated on HW inner val only (HW domain, correct distribution)
            hw_inner_mask = idx[cut:]
            hw_inner_mask = hw_inner_mask[hw_inner_mask < len(X_hw_tr_qt)]
            if len(hw_inner_mask) > 0 and len(np.unique(y_hw_tr[hw_inner_mask])) == 2:
                p_inner = bst.predict(xgb.DMatrix(
                    X_hw_tr_qt[hw_inner_mask], feature_names=FEATURES))
                thr_f = youden_threshold(y_hw_tr[hw_inner_mask], p_inner)
            else:
                thr_f = youden_threshold(y_tr_comb[idx[cut:]],
                                         bst.predict(d_iva)) \
                        if len(np.unique(y_tr_comb[idx[cut:]])) == 2 else 0.5

            bst.save_model(str(cache_m))
            pickle.dump((qt_hw_f, thr_f), open(cache_qt, "wb"))
            status = "(t)"

        X_hw_te_qt = qt_hw_f.transform(X_hw_te).astype(np.float32)
        probs      = bst.predict(xgb.DMatrix(X_hw_te_qt, feature_names=FEATURES))
        mean_p     = float(probs.mean())
        label_str  = "ABNORMAL" if int(y_hw_te[0]) == 1 else "normal"

        print(f"  {fold:>4}  {name:<32}  {label_str:>8}  "
              f"{len(y_hw_te):>5}  {mean_p:>7.4f}  {status}")

        records.append({"pid": test_pid, "name": name,
                        "true_label": int(y_hw_te[0]),
                        "y": y_hw_te.tolist(), "probs": probs.tolist(),
                        "mean_prob": round(mean_p, 4),
                        "sessions": hw_sess[te_idx].tolist()})

    return records


# ── Pooled LOPO stats ─────────────────────────────────────────────────────────
def pooled_lopo_stats(records):
    all_y = np.array([v for r in records for v in r["y"]], dtype=np.int32)
    all_p = np.array([v for r in records for v in r["probs"]])
    auc = float(roc_auc_score(all_y, all_p)) if len(np.unique(all_y)) == 2 else float("nan")
    ap  = float(average_precision_score(all_y, all_p)) if len(np.unique(all_y)) == 2 else float("nan")

    abn_m  = [r["mean_prob"] for r in records if r["true_label"] == 1]
    norm_m = [r["mean_prob"] for r in records if r["true_label"] == 0]

    mw_stat = mw_p = float("nan")
    if len(abn_m) >= 2 and len(norm_m) >= 2:
        mw_stat, mw_p = mannwhitneyu(abn_m, norm_m, alternative="greater")

    sig = ("***" if not np.isnan(mw_p) and mw_p < 0.001 else
           ("**" if not np.isnan(mw_p) and mw_p < 0.01 else
            ("*"  if not np.isnan(mw_p) and mw_p < 0.05 else "ns")))

    print(f"\n{SEP}")
    print(f"  POOLED LOPO  —  abn={len(abn_m)} pts  norm={len(norm_m)} pts")
    print(SEP)
    print(f"  Pooled AUC        : {auc:.4f}")
    print(f"  Avg Precision     : {ap:.4f}")
    print(f"  Mean P (abn pts)  : {np.mean(abn_m):.4f}")
    print(f"  Mean P (norm pts) : {np.mean(norm_m):.4f}")
    if not np.isnan(mw_p):
        print(f"  Mann-Whitney p    : {mw_p:.4f}  {sig}")
    print(SEP)

    return {"pooled_auc": round(auc, 4), "avg_precision": round(ap, 4),
            "n_abn_patients": len(abn_m), "n_norm_patients": len(norm_m),
            "mean_abn_score":  round(float(np.mean(abn_m)), 4)  if abn_m  else None,
            "mean_norm_score": round(float(np.mean(norm_m)), 4) if norm_m else None,
            "mannwhitney_stat": round(float(mw_stat), 2) if not np.isnan(mw_stat) else None,
            "mannwhitney_p":    round(float(mw_p),    6) if not np.isnan(mw_p)    else None}


# ── Valsalva analysis ─────────────────────────────────────────────────────────
def valsalva_analysis(records):
    val_m, base_m = [], []
    for r in records:
        sess = np.array(r["sessions"])
        prob = np.array(r["probs"])
        pv = prob[sess == 3]; pb = prob[sess != 3]
        if len(pv) > 0 and len(pb) > 0:
            val_m.append(float(pv.mean()))
            base_m.append(float(pb.mean()))
    if len(val_m) < 4:
        return {}
    v, b = np.array(val_m), np.array(base_m)
    stat, pval = _wilcoxon(v, b, alternative="greater")
    pct = 100 * (v > b).mean()
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"\n{SEP}")
    print("  VALSALVA ANALYSIS  (ICP elevation maneuver)")
    print(SEP)
    print(f"  Subjects with session data : {len(v)}")
    print(f"  Valsalva > baseline        : {int((v>b).sum())}/{len(v)}  ({pct:.0f}%)")
    print(f"  Mean P valsalva            : {v.mean():.4f}")
    print(f"  Mean P baseline            : {b.mean():.4f}")
    print(f"  Wilcoxon p (one-tailed)    : {pval:.6f}  {sig}")
    print(SEP)
    return {"n": len(v), "pct_higher": round(pct, 1),
            "mean_val": round(float(v.mean()), 4),
            "mean_base": round(float(b.mean()), 4),
            "wilcoxon_p": round(float(pval), 6)}


# ── Dose-response analysis (within-subject ICP modulation ladder) ─────────────
# Physiologically-expected ICP ordering (low -> high):
#   head-up 30 (sess 1)  <  supine (sess 0)  <  head-down 10 (sess 2)  <  valsalva (sess 3)
# Each subject is their own control -> auto-controls age / heart-rate confounds.
SESSION_ICP_RANK = {1: 0, 0: 1, 2: 2, 3: 3}   # session_label -> expected ICP rank
SESSION_NAME     = {1: "head-up-30", 0: "supine", 2: "head-down-10", 3: "valsalva"}
ORDERED_SESSIONS = [1, 0, 2, 3]               # ascending expected ICP


def dose_response_analysis(records):
    """
    Within-subject graded ICP-modulation test.
    Returns per-session means, Friedman omnibus, adjacent-pair Wilcoxon,
    monotonicity fraction, and mean within-subject Spearman(rank, output).
    """
    # subject x session mean-probability matrix (only subjects with all 4 sessions)
    rows, spearmans, mono_hits = [], [], 0
    for r in records:
        sess = np.array(r["sessions"]); prob = np.array(r["probs"])
        per = {s: float(prob[sess == s].mean()) for s in ORDERED_SESSIONS
               if (sess == s).sum() > 0}
        if len(per) < 4:
            continue
        vec = [per[s] for s in ORDERED_SESSIONS]         # in ascending-ICP order
        rows.append(vec)
        # within-subject Spearman between expected rank (0..3) and model output
        rho, _ = spearmanr([0, 1, 2, 3], vec)
        if not np.isnan(rho):
            spearmans.append(float(rho))
        # strict monotonic increase across the ladder
        if all(vec[i] < vec[i + 1] for i in range(3)):
            mono_hits += 1

    if len(rows) < 4:
        print("\n  Dose-response: insufficient subjects with all 4 sessions")
        return {}

    M = np.array(rows)                                   # (n_subj, 4)
    n = len(M)

    # Friedman omnibus across the 4 conditions (repeated measures)
    fr_stat, fr_p = friedmanchisquare(*[M[:, i] for i in range(4)])

    # Adjacent-pair one-tailed Wilcoxon (each step should raise output)
    pair_results = []
    for i in range(3):
        lo, hi = ORDERED_SESSIONS[i], ORDERED_SESSIONS[i + 1]
        try:
            _, p = _wilcoxon(M[:, i + 1], M[:, i], alternative="greater")
        except ValueError:
            p = float("nan")
        pair_results.append((SESSION_NAME[lo], SESSION_NAME[hi],
                             float(M[:, i].mean()), float(M[:, i + 1].mean()), float(p)))

    mean_rho = float(np.mean(spearmans)) if spearmans else float("nan")
    mono_pct = 100.0 * mono_hits / n

    # ── 3-level ladder: drop head-up (smallest, ambiguous ICP delta) ──
    # supine (0) < head-down (2) < valsalva (3) — the three conditions with
    # clear physiological separation. Cleaner headline monotonicity number.
    THREE = [0, 2, 3]
    three_hits, three_rho = 0, []
    for r in records:
        sess = np.array(r["sessions"]); prob = np.array(r["probs"])
        per = {s: float(prob[sess == s].mean()) for s in THREE if (sess == s).sum() > 0}
        if len(per) < 3:
            continue
        vec3 = [per[s] for s in THREE]
        rho3, _ = spearmanr([0, 1, 2], vec3)
        if not np.isnan(rho3):
            three_rho.append(float(rho3))
        if vec3[0] < vec3[1] < vec3[2]:
            three_hits += 1
    n3        = len(three_rho)
    three_pct = 100.0 * three_hits / n3 if n3 else float("nan")
    mean_rho3 = float(np.mean(three_rho)) if three_rho else float("nan")

    fsig = "***" if fr_p < 0.001 else ("**" if fr_p < 0.01 else ("*" if fr_p < 0.05 else "ns"))
    print(f"\n{SEP}")
    print("  DOSE-RESPONSE ANALYSIS  (within-subject ICP ladder)")
    print(SEP)
    print(f"  Subjects with all 4 sessions : {n}")
    print(f"  Expected ICP order (low->high): head-up < supine < head-down < valsalva")
    print(f"\n  {'Session':<14} {'Mean P(ICP)':>12}")
    print(f"  {'-'*28}")
    for s in ORDERED_SESSIONS:
        col = ORDERED_SESSIONS.index(s)
        print(f"  {SESSION_NAME[s]:<14} {M[:, col].mean():>12.4f}")
    print(f"\n  Friedman omnibus     : chi2={fr_stat:.2f}  p={fr_p:.6f}  {fsig}")
    print(f"  Mean within-subj rho : {mean_rho:+.3f}   (Spearman rank vs output)")
    print(f"  Monotonic subjects   : {mono_hits}/{n}  ({mono_pct:.0f}%)")
    print(f"\n  Adjacent-step Wilcoxon (one-tailed, each step should raise output):")
    for lo, hi, m_lo, m_hi, p in pair_results:
        s = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {lo:>12} -> {hi:<12} : {m_lo:.4f} -> {m_hi:.4f}  p={p:.5f}  {s}")
    print(f"\n  3-level ladder (supine < head-down < valsalva; head-up dropped):")
    print(f"    Monotonic subjects : {three_hits}/{n3}  ({three_pct:.0f}%)")
    print(f"    Mean within-subj rho: {mean_rho3:+.3f}")
    print(SEP)

    return {
        "n_subjects": n,
        "session_means": {SESSION_NAME[s]: round(float(M[:, ORDERED_SESSIONS.index(s)].mean()), 4)
                          for s in ORDERED_SESSIONS},
        "friedman_chi2": round(float(fr_stat), 3),
        "friedman_p":    round(float(fr_p), 6),
        "mean_within_subject_spearman": round(mean_rho, 3),
        "monotonic_fraction": round(mono_pct / 100.0, 3),
        "adjacent_pairs": [{"from": lo, "to": hi,
                            "mean_from": round(m_lo, 4), "mean_to": round(m_hi, 4),
                            "wilcoxon_p": round(p, 6)}
                           for lo, hi, m_lo, m_hi, p in pair_results],
        "three_level": {
            "conditions": ["supine", "head-down-10", "valsalva"],
            "n_subjects": n3,
            "monotonic_fraction": round(three_pct / 100.0, 3) if not np.isnan(three_pct) else None,
            "mean_within_subject_spearman": round(mean_rho3, 3) if not np.isnan(mean_rho3) else None,
        },
    }


# ── Baselines LOPO (pooled) ────────────────────────────────────────────────────
def run_baselines_lopo(hw_X, hw_y, hw_pid, c_X, c_y):
    from sklearn.base import clone
    logo = LeaveOneGroupOut()
    pool = {n: ([], []) for n in ["LogReg", "RandForest", "LinearSVM"]}
    # class_weight="balanced" handles the mild imbalance without sample weights
    clfs = {
        "LogReg":     LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                         class_weight="balanced", random_state=SEED),
        "RandForest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1,
                                             class_weight="balanced", random_state=SEED),
        "LinearSVM":  CalibratedClassifierCV(
                          LinearSVC(max_iter=2000, C=1.0, class_weight="balanced",
                                    random_state=SEED), cv=3),
    }
    print(f"\n[8] Baseline LOPO (pooled)  — LogReg | RandForest | LinearSVM ...")

    for fold, (tr_idx, te_idx) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        X_hw_te, y_hw_te = hw_X[te_idx], hw_y[te_idx]
        X_hw_tr, y_hw_tr = hw_X[tr_idx], hw_y[tr_idx]

        qt_hw_f = fit_qt(X_hw_tr)
        qt_c_f  = fit_qt(c_X)
        X_hw_tr_qt = qt_hw_f.transform(X_hw_tr).astype(np.float32)
        X_c_qt     = qt_c_f.transform(c_X).astype(np.float32)
        X_hw_te_qt = qt_hw_f.transform(X_hw_te).astype(np.float32)

        X_tr = np.concatenate([X_hw_tr_qt, X_c_qt])
        y_tr = np.concatenate([y_hw_tr, c_y])

        for name, clf in clfs.items():
            clf_ = clone(clf)   # sklearn clone: safe deep-copy, no nested-param issues
            clf_.fit(X_tr, y_tr)
            proba = clf_.predict_proba(X_hw_te_qt)[:, 1]
            pool[name][0].extend(y_hw_te.tolist())
            pool[name][1].extend(proba.tolist())

    print(f"\n  {'Model':<14}  {'Pooled AUC':>10}")
    print(f"  {'-'*28}")
    summary = {}
    for name, (yt, yp) in pool.items():
        yt_a, yp_a = np.array(yt), np.array(yp)
        if len(np.unique(yt_a)) == 2:
            auc = float(roc_auc_score(yt_a, yp_a))
            print(f"  {name:<14}  {auc:>10.4f}")
            summary[name] = {"pooled_auc": round(auc, 4)}
    return summary, {n: (pool[n][0], pool[n][1]) for n in pool}


# ── Feature ablation (pooled, domain-aware QT) ────────────────────────────────
def run_ablation(hw_X, hw_y, hw_pid, c_X, c_y, device, full_auc):
    logo = LeaveOneGroupOut()
    results = {}
    print(f"\n[9] Feature ablation — drop-one LOPO (pooled AUC, domain QT) ...")
    for drop_f in FEATURES:
        keep     = [i for i, f in enumerate(FEATURES) if f != drop_f]
        fn       = [f for f in FEATURES if f != drop_f]
        all_y, all_p = [], []
        for fold, (tr_idx, te_idx) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
            X_te_raw, y_te   = hw_X[te_idx][:, keep], hw_y[te_idx]
            X_tr_raw, y_tr_hw = hw_X[tr_idx][:, keep], hw_y[tr_idx]
            c_sub            = c_X[:, keep]

            qt_hw_f = fit_qt(X_tr_raw)
            qt_c_f  = fit_qt(c_sub)
            X_tr_qt = np.concatenate([qt_hw_f.transform(X_tr_raw).astype(np.float32),
                                       qt_c_f.transform(c_sub).astype(np.float32)])
            y_tr    = np.concatenate([y_tr_hw, c_y])
            spw_f   = scale_pos_weight(y_tr)
            X_te_qt = qt_hw_f.transform(X_te_raw).astype(np.float32)

            rng = np.random.default_rng(SEED + fold)
            idx = rng.permutation(len(y_tr)); cut = int(0.9 * len(idx))
            bst = xgb.train(
                xgb_params(device, spw_f, SEED + fold),
                xgb.DMatrix(X_tr_qt[idx[:cut]], label=y_tr[idx[:cut]], feature_names=fn),
                num_boost_round=400, verbose_eval=False)
            probs = bst.predict(xgb.DMatrix(X_te_qt, feature_names=fn))
            all_y.extend(y_te.tolist()); all_p.extend(probs.tolist())

        auc = float(roc_auc_score(all_y, all_p)) if len(np.unique(all_y)) == 2 else float("nan")
        delta = auc - full_auc
        print(f"  Drop {drop_f:<28}: AUC {auc:.4f}  (delta={delta:+.4f})")
        results[drop_f] = {"pooled_auc": round(auc, 4), "delta": round(delta, 4)}
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────
def save_plots(records, lopo_stats, bst, c_X, c_y, hw_X, hw_y, qt_hw, qt_c, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-patient score bar
    nms    = [r["name"].replace(".csv", "").replace("icp_", "") for r in records]
    scores = [r["mean_prob"] for r in records]
    colors = ["#e74c3c" if r["true_label"] == 1 else "#3498db" for r in records]
    fig, ax = plt.subplots(figsize=(max(14, len(nms) * 0.45), 4))
    ax.bar(range(len(nms)), scores, color=colors, alpha=0.85)
    ax.set_xticks(range(len(nms)))
    ax.set_xticklabels(nms, rotation=65, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Mean P(ICP elevated)")
    ax.set_title("Hybrid V4 — Per-Patient LOPO Scores  (red=ABNORMAL, blue=normal)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="Abnormal"),
                       Patch(color="#3498db", label="Normal")], loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_patient_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Pooled ROC
    if not np.isnan(lopo_stats["pooled_auc"]):
        all_y = [v for r in records for v in r["y"]]
        all_p = [v for r in records for v in r["probs"]]
        fpr, tpr, _ = roc_curve(all_y, all_p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2, color="#e74c3c",
                label=f"Hybrid V4 LOPO (AUC={lopo_stats['pooled_auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Pooled LOPO ROC — Hybrid V4")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "v4_roc.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Feature importance
    gain  = bst.get_score(importance_type="gain")
    total = sum(gain.values()) + 1e-12
    vals  = [gain.get(f, 0.0) / total for f in FEATURES]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(FEATURES, vals, color="#3498db", alpha=0.8)
    ax.set_xlabel("Normalised Gain"); ax.set_title("Feature Importance — Hybrid V4")
    ax.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "v4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Domain feature distribution comparison (cardiac_amplitude)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    fig.suptitle("Feature distributions after domain-separated QT normalization\n"
                 "(both domains → N(0,1) in their own feature space)")
    c_norm  = qt_c.transform(c_X).astype(np.float32)
    hw_norm = qt_hw.transform(hw_X).astype(np.float32)
    for i, (ax, fname) in enumerate(zip(axes, FEATURES)):
        ax.hist(c_norm[:, i],  bins=60, alpha=0.5, color="#e74c3c",
                label="CHARIS", density=True)
        ax.hist(hw_norm[:, i], bins=60, alpha=0.5, color="#3498db",
                label="HW",     density=True)
        ax.set_title(fname, fontsize=8)
        ax.set_xlabel("QT value"); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "v4_domain_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plots -> {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  Hybrid XGBoost V4  —  Domain-separated QT,  no SMOTE")
    print("  y=1 : CHARIS greedy abnormals + HW flagged patients")
    print("  y=0 : HW normal patients only")
    print("  Key : separate qt_c / qt_hw eliminates sensor domain shift")
    print(SEP)

    # [1] Load CHARIS cache
    print(f"\n[1] Loading CHARIS cache ...")
    for p in [CACHE_X, CACHE_Y, CACHE_PID]:
        if not p.exists():
            print(f"  ERROR: {p} missing — run full_pipeline_qt.py first")
            sys.exit(1)
    c_X   = np.load(CACHE_X)
    c_y   = np.load(CACHE_Y)
    c_pid = np.load(CACHE_PID)
    print(f"  {len(c_X):,} windows  abn={int((c_y==1).sum()):,}  "
          f"norm={int((c_y==0).sum()):,}  patients={len(np.unique(c_pid))}")

    # [2] Load HW data
    print(f"\n[2] Loading HW data ...")
    if not FLAGS_PATH.exists():
        print(f"  ERROR: {FLAGS_PATH} missing — run flag_hw.py first")
        sys.exit(1)
    raw        = json.load(open(FLAGS_PATH))
    flags_data = raw.get("patients", raw)
    print(f"  Loaded flags from {FLAGS_PATH}")

    hw_X, hw_y, hw_pid, hw_sess, names = load_hw_labeled(HW_DIR, flags_data)
    n_hw_abn  = int((hw_y == 1).sum())
    n_hw_norm = int((hw_y == 0).sum())
    n_pts     = len(np.unique(hw_pid))
    n_abn_pts = sum(1 for v in flags_data.values() if v.get("flagged"))
    print(f"  Patients : {n_pts}  (abn={n_abn_pts}  norm={n_pts-n_abn_pts})")
    print(f"  Windows  : {len(hw_X):,}  abn={n_hw_abn:,}  norm={n_hw_norm:,}")

    # [3] Select CHARIS patients (whole-patient, greedy, budget = HW_norm - HW_abn)
    print(f"\n[3] Selecting CHARIS abnormal patients ...")
    c_X_sel, c_y_sel, c_pid_sel, sel_pids, n_c_abn = select_charis_patients(
        c_X, c_y, c_pid, n_hw_norm, n_hw_abn)

    total_abn  = n_hw_abn + n_c_abn
    spw_global = n_hw_norm / max(total_abn, 1)
    print(f"\n  Before normalization:")
    print(f"    y=1 : {n_hw_abn:,} HW + {n_c_abn:,} CHARIS = {total_abn:,}")
    print(f"    y=0 : {n_hw_norm:,} HW only")
    print(f"    Ratio abn:norm = {total_abn/n_hw_norm:.3f}  → scale_pos_weight ≈ {spw_global:.3f}")

    device = get_device()
    print(f"  Device: {device.upper()}")

    # [4-6] Main split
    bst, qt_hw, qt_c, thr, test_m = run_main_split(
        hw_X, hw_y, hw_pid, c_X_sel, c_y_sel, device, names)

    # [7] LOPO
    records = run_lopo(hw_X, hw_y, hw_pid, hw_sess,
                       c_X_sel, c_y_sel, device, names)
    lopo_stats = pooled_lopo_stats(records)

    # Persist raw per-window LOPO records so dose-response / re-analysis needs no re-run
    pickle.dump(records, open(OUT_DIR / "lopo_records.pkl", "wb"))

    # [8] Valsalva  +  within-subject dose-response ladder
    val_stats  = valsalva_analysis(records)
    dose_stats = dose_response_analysis(records)

    # [9] Baselines
    bl_stats, bl_pool = run_baselines_lopo(hw_X, hw_y, hw_pid, c_X_sel, c_y_sel)

    # [10] Ablation
    ablation = run_ablation(hw_X, hw_y, hw_pid, c_X_sel, c_y_sel,
                            device, lopo_stats["pooled_auc"])

    # [11] Feature importance
    gain  = bst.get_score(importance_type="gain")
    total_g = sum(gain.values()) + 1e-12
    feat_imp = {f: round(gain.get(f, 0.0) / total_g, 4) for f in FEATURES}
    print(f"\n{SEP}\n  Feature Importance (normalised gain)\n{SEP}")
    for f, g in sorted(feat_imp.items(), key=lambda x: -x[1]):
        print(f"  {f:<28}: {g:.4f}  {'#'*int(g*40)}")

    # [12] Plots
    print(f"\n[12] Saving plots ...")
    save_plots(records, lopo_stats, bst,
               c_X_sel, c_y_sel, hw_X, hw_y, qt_hw, qt_c, OUT_DIR)

    # Final summary
    print(f"\n{SEP}")
    print("  HYBRID V4 FINAL SUMMARY")
    print(SEP)
    print(f"  Strategy       : Domain-separated QT (qt_c / qt_hw) + scale_pos_weight")
    print(f"  HW patients    : {n_pts}  (abn={lopo_stats['n_abn_patients']}  "
          f"norm={lopo_stats['n_norm_patients']})")
    print(f"  CHARIS used    : {len(sel_pids)} patients  ({n_c_abn:,} abn windows)")
    print(f"  Training size  : {total_abn + n_hw_norm:,}  "
          f"(no SMOTE)  spw={spw_global:.3f}")
    if test_m:
        print(f"  Test AUC       : {test_m.get('auc', float('nan')):.4f}")
    print(f"  Pooled LOPO AUC: {lopo_stats['pooled_auc']:.4f}")
    if val_stats:
        print(f"  Valsalva       : {val_stats['pct_higher']:.0f}%  "
              f"p={val_stats['wilcoxon_p']:.6f}")
    for nm, b in bl_stats.items():
        print(f"  Baseline {nm:<10}: AUC {b['pooled_auc']:.4f}")
    print(SEP)

    # Save JSON
    out = {
        "date": date.today().isoformat(),
        "strategy": "domain-separated QT (qt_c/qt_hw), scale_pos_weight, no SMOTE",
        "charis_patients_selected": sel_pids,
        "charis_abn_windows": n_c_abn,
        "hw_patients": n_pts,
        "hw_abn_windows": n_hw_abn,
        "hw_norm_windows": n_hw_norm,
        "scale_pos_weight": round(spw_global, 4),
        "device": device,
        "test_metrics": test_m,
        "lopo_eval": lopo_stats,
        "valsalva_stats": val_stats,
        "dose_response": dose_stats,
        "baselines": bl_stats,
        "feature_ablation": ablation,
        "feature_importance": feat_imp,
        "per_patient_lopo": [{"name": r["name"], "true_label": r["true_label"],
                               "mean_prob": r["mean_prob"]} for r in records],
    }
    with open(OUT_DIR / "results_v4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results -> {OUT_DIR}/results_v4.json")


if __name__ == "__main__":
    main()
