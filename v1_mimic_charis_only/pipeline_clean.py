"""
pipeline_clean.py — Clean CHARIS XGBoost ICP Binary Classifier
==============================================================
Single self-contained script. Run once, get full results.

Design decisions
----------------
Threshold  : 20 mmHg                   [Ye et al. 2022, BTF 2016]
Label rule : >60% samples >= 20 mmHg  [Ye et al. 2022]
Preprocess : clip [-5, 50] mmHg        [Ye et al. 2022, Step 1]
Features   : 5 ICP-signal features     [hardware-compatible: no MAP/ABP needed]
Window     : 10 s @ 50 Hz = 500 samp  [Ye et al. 2022 / PMC11986046]
Overlap    : 50%                        [PMC11986046]
Split      : GroupShuffleSplit 70/10/20 [PMC9744906]
SMOTE      : within-patient, exact 50/50[PMC9744906]
Model      : XGBoost + GPU              [PMC11986046, PMC9744906]

Hardware: ESP32-C3, TM laser disp (SFH9206), PPG (MAX30102), IMU (MPU6050)
All 5 features extractable from TM displacement waveform on ESP32-C3.

Usage:
    cd C:/Users/asus/Documents/GitHub/Pran/v1_mimic_charis_only
    python pipeline_clean.py
"""
from __future__ import annotations
import json, sys, warnings
from datetime import date
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
import seaborn as sns
import wfdb
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ─────────────────────────────────────────────────────────────────
FS          = 50           # CHARIS native Hz
WIN         = 500          # 10 s × 50 Hz
STEP        = 250          # 50% overlap
THRESH      = 20.0         # mmHg  [Ye et al. 2022 / BTF 2016]
LABEL_FRAC  = 0.60         # >60% samples >= 20 → abnormal  [Ye et al. 2022]
SEED        = 42

ICP_CH = {"ICP", "ICP1", "ICP2", "ICPC"}

FEATURES = [
    "cardiac_amplitude",    # P99-P1 of 1.0–2.5 Hz bandpass (ICP pulse size)
    "cardiac_frequency",    # dominant freq in 0.7–2.5 Hz
    "respiratory_amplitude",# P99-P1 of 0.1–0.5 Hz bandpass
    "slow_wave_power",      # db4 cA5 energy fraction (<0.78 Hz)
    "cardiac_power",        # db4 cD4 energy fraction (1.56–3.12 Hz)
]
N = len(FEATURES)   # 5

CHARIS_DIR = Path("C:/Users/asus/Documents/GitHub/Pran/data/raw/charis")
OUT_DIR    = Path("C:/Users/asus/Documents/GitHub/Pran/results/clean")
MODEL_DIR  = Path("C:/Users/asus/Documents/GitHub/Pran/models")


# ── Feature extraction ────────────────────────────────────────────────────────

def _bp(x, lo, hi):
    nyq = FS / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, x)

def extract(win: np.ndarray) -> np.ndarray | None:
    if win.std() < 0.02:
        return None
    x = win.astype(np.float64)

    # cardiac amplitude: P99-P1 of 1.0–2.5 Hz band
    c = _bp(x, 1.0, 2.5)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    # cardiac frequency: dominant freq in 0.7–2.5 Hz
    freqs = np.fft.rfftfreq(len(x), d=1.0 / FS)
    pwr   = np.abs(np.fft.rfft(x)) ** 2
    mask  = (freqs >= 0.7) & (freqs <= 2.5)
    if not mask.any():
        return None
    card_freq = float(freqs[mask][np.argmax(pwr[mask])])

    # respiratory amplitude: P99-P1 of 0.1–0.5 Hz band
    r = _bp(x, 0.1, 0.5)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    # wavelet powers: db4, level 5 @ 50 Hz
    # bands: cA5=0–0.78 Hz (slow), cD5=0.78–1.56, cD4=1.56–3.12 Hz (cardiac)
    coeffs   = pywt.wavedec(x, "db4", level=5)
    energies = [float(np.sum(c**2)) for c in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total   # cA5
    cardiac_pow = energies[2] / total   # cD4

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Load CHARIS ───────────────────────────────────────────────────────────────

def load_charis():
    records = sorted([f.stem for f in CHARIS_DIR.glob("*.hea")])
    print(f"  {len(records)} CHARIS records found")
    Xall, yall, pids = [], [], []

    for rec_name in records:
        pid = int("".join(filter(str.isdigit, rec_name)) or 0)
        if pid == 0:
            continue
        try:
            rec = wfdb.rdrecord(str(CHARIS_DIR / rec_name))
        except Exception as e:
            print(f"  SKIP {rec_name}: {e}"); continue

        sig = [s.upper() for s in rec.sig_name]
        icp_i = next((i for i, s in enumerate(sig) if s in ICP_CH), None)
        if icp_i is None:
            print(f"  SKIP {rec_name}: no ICP channel {sig}"); continue

        icp = rec.p_signal[:, icp_i].astype(np.float32)
        orig_fs = int(rec.fs)

        # Resample to 50 Hz if needed
        if orig_fs != FS:
            n = int(len(icp) * FS / orig_fs)
            icp = np.interp(np.linspace(0, 1, n),
                            np.linspace(0, 1, len(icp)), icp).astype(np.float32)

        # Ye et al. Step 1: clip to [-5, 50] mmHg, forward-fill
        bad = (icp < -5.0) | (icp > 50.0)
        icp[bad] = np.nan
        mask = np.isnan(icp)
        if mask.any():
            idx = np.where(~mask, np.arange(len(icp)), 0)
            np.maximum.accumulate(idx, out=idx)
            icp = icp[idx]

        n_ok = n_skip = n0 = n1 = 0
        n_win = (len(icp) - WIN) // STEP + 1
        for w in range(n_win):
            s, e = w * STEP, w * STEP + WIN
            win = icp[s:e]
            label = 1 if (win >= THRESH).mean() > LABEL_FRAC else 0
            feat  = extract(win)
            if feat is None:
                n_skip += 1; continue
            Xall.append(feat); yall.append(label); pids.append(pid)
            n_ok += 1
            if label == 0: n0 += 1
            else: n1 += 1

        print(f"  {rec_name}: {n_ok:,} windows [norm={n0} abn={n1}] skip={n_skip}")

    X   = np.array(Xall, dtype=np.float32)
    y   = np.array(yall, dtype=np.int64)
    pid = np.array(pids, dtype=np.int32)
    return X, y, pid


# ── SMOTE exact 50/50 ─────────────────────────────────────────────────────────

def smote(X, y, pid):
    Xs, ys = [], []
    for p in np.unique(pid):
        m = pid == p
        Xp, yp = X[m], y[m]
        n0, n1 = (yp==0).sum(), (yp==1).sum()
        if n0 == 0 or n1 == 0:
            Xs.append(Xp); ys.append(yp); continue
        k = max(1, min(5, min(n0, n1) - 1))
        try:
            Xp, yp = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xp, yp)
        except Exception:
            Xp, yp = RandomOverSampler(random_state=SEED).fit_resample(Xp, yp)
        Xs.append(Xp); ys.append(yp)
    X_out = np.vstack(Xs).astype(np.float32)
    y_out = np.concatenate(ys)
    if (y_out==0).sum() != (y_out==1).sum():
        X_out, y_out = RandomOverSampler(random_state=SEED).fit_resample(X_out, y_out)
        X_out = X_out.astype(np.float32)
    assert (y_out==0).sum() == (y_out==1).sum()
    print(f"  SMOTE: {int((y_out==0).sum()):,} normal | {int((y_out==1).sum()):,} abnormal (50/50)")
    return X_out, y_out


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    try:
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            return "cpu"
        xgb.train({"device":"cuda","tree_method":"hist","verbosity":0},
                  xgb.DMatrix(np.zeros((4,N)), label=[0,1,0,1]), num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, X_va, y_va, device):
    dt = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
    dv = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURES)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.05 if device == "cuda" else 0.1,
        "max_depth": 6 if device == "cuda" else 4,
        "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0,   # 50/50 after SMOTE
        "lambda": 1.0, "alpha": 0.1, "seed": SEED,
        "tree_method": "hist", "device": device, "verbosity": 0,
    }
    evals = {}
    bst = xgb.train(params, dt,
                    num_boost_round=1000 if device=="cuda" else 500,
                    evals=[(dt,"train"),(dv,"val")],
                    early_stopping_rounds=50,
                    verbose_eval=50, evals_result=evals)
    print(f"  Best iter: {bst.best_iteration}  val-AUC: {bst.best_score:.4f}")
    return bst, evals


# ── Evaluate + plots ──────────────────────────────────────────────────────────

def evaluate(bst, X_va, y_va, X_te, y_te, pid_te, evals, out_dir):
    dv = xgb.DMatrix(X_va, feature_names=FEATURES)
    dt = xgb.DMatrix(X_te, feature_names=FEATURES)
    pv = bst.predict(dv)
    pt = bst.predict(dt)

    # Youden's J threshold on val
    fpr_v, tpr_v, thr_v = roc_curve(y_va, pv)
    thr = float(thr_v[np.argmax(tpr_v - fpr_v)])

    pred = (pt >= thr).astype(int)
    auc  = roc_auc_score(y_te, pt)
    f1   = f1_score(y_te, pred, zero_division=0)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    spec = recall_score(1-y_te, 1-pred, zero_division=0)
    bacc = balanced_accuracy_score(y_te, pred)
    ap   = average_precision_score(y_te, pt)
    cm   = confusion_matrix(y_te, pred)

    SEP = "="*60
    print(f"\n{SEP}")
    print("  RESULTS  (CHARIS-only, 20 mmHg, 5 features)")
    print(SEP)
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  F1-score     : {f1:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  Specificity  : {spec:.4f}")
    print(f"  Balanced Acc : {bacc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  Threshold    : {thr:.4f}")
    print(f"\n{classification_report(y_te, pred, target_names=['Normal','Abnormal'], zero_division=0)}")
    print(SEP)

    # Save report
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(
        f"AUC={auc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
        f"Spec={spec:.4f} BalAcc={bacc:.4f} AP={ap:.4f} thr={thr:.4f}\n"
        + classification_report(y_te, pred,
                                 target_names=["Normal","Abnormal"], zero_division=0),
        encoding="utf-8")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("CHARIS XGBoost — 20 mmHg threshold — 5 features", fontsize=12)

    # 1. AUC convergence
    ax = axes[0,0]
    ax.plot(evals["train"]["auc"], lw=1.5, label="Train")
    ax.plot(evals["val"]["auc"],   lw=1.5, label="Val")
    ax.axvline(bst.best_iteration, color="orange", ls="--", lw=1,
               label=f"Stop@{bst.best_iteration}")
    ax.set(xlabel="Round", ylabel="AUC", title="Convergence (AUC)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Logloss
    ax = axes[0,1]
    ax.plot(evals["train"]["logloss"], lw=1.5, label="Train")
    ax.plot(evals["val"]["logloss"],   lw=1.5, label="Val")
    ax.axvline(bst.best_iteration, color="orange", ls="--", lw=1)
    ax.set(xlabel="Round", ylabel="Logloss", title="Convergence (Logloss)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. ROC
    ax = axes[0,2]
    fpr, tpr, _ = roc_curve(y_te, pt)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 4. PR
    ax = axes[1,0]
    pc, rc, _ = precision_recall_curve(y_te, pt)
    ax.plot(rc, pc, lw=2, label=f"AP={ap:.4f}")
    ax.axhline(y_te.mean(), color="gray", ls="--", lw=1, label="Baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Confusion matrix
    ax = axes[1,1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"],
                ax=ax, cbar=False, annot_kws={"size":14})
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix (thr={thr:.3f})")

    # 6. Feature importance
    ax = axes[1,2]
    gain = bst.get_score(importance_type="gain")
    vals  = [gain.get(f, 0.0) for f in FEATURES]
    ax.barh(FEATURES[::-1], vals[::-1], color="#2C5282")
    ax.set(xlabel="Gain", title="Feature Importance")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {out_dir / 'results.png'}")

    return {"auc":round(float(auc),4), "f1":round(float(f1),4),
            "precision":round(float(prec),4), "recall":round(float(rec),4),
            "specificity":round(float(spec),4), "balanced_acc":round(float(bacc),4),
            "avg_precision":round(float(ap),4), "threshold":round(float(thr),4),
            "cm": cm.tolist()}, thr


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def cv(X, y, pid, device, thr):
    print(f"\n  5-fold GroupKFold CV ...")
    f1s, aucs = [], []
    for fold, (tr_i, te_i) in enumerate(GroupKFold(5).split(X, y, groups=pid)):
        X_b, y_b = smote(X[tr_i], y[tr_i], pid[tr_i])
        rng = np.random.RandomState(SEED+fold)
        inner = rng.permutation(len(y_b)); cut = int(0.9*len(inner))
        d_tr = xgb.DMatrix(X_b[inner[:cut]], label=y_b[inner[:cut]], feature_names=FEATURES)
        d_va = xgb.DMatrix(X_b[inner[cut:]], label=y_b[inner[cut:]], feature_names=FEATURES)
        d_te = xgb.DMatrix(X[te_i], feature_names=FEATURES)
        params = {"objective":"binary:logistic","eval_metric":["logloss","auc"],
                  "eta":0.05 if device=="cuda" else 0.1,
                  "max_depth":6 if device=="cuda" else 4,
                  "min_child_weight":3,"subsample":0.8,"colsample_bytree":0.8,
                  "lambda":1.0,"alpha":0.1,"seed":SEED+fold,
                  "tree_method":"hist","device":device,"verbosity":0}
        b = xgb.train(params, d_tr, num_boost_round=500,
                      evals=[(d_va,"val")], early_stopping_rounds=50, verbose_eval=0)
        pr = b.predict(d_te)
        f1s.append(f1_score(y[te_i],(pr>=thr).astype(int), zero_division=0))
        aucs.append(roc_auc_score(y[te_i], pr))
        print(f"    Fold {fold+1}: F1={f1s[-1]:.4f}  AUC={aucs[-1]:.4f}"
              f"  ({len(np.unique(pid[te_i]))} test patients)")
    print(f"  CV F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return {"f1_mean":round(float(np.mean(f1s)),4),"f1_std":round(float(np.std(f1s)),4),
            "auc_mean":round(float(np.mean(aucs)),4),"auc_std":round(float(np.std(aucs)),4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  CHARIS XGBoost Pipeline  |  Clean Version")
    print(f"  Threshold: {THRESH} mmHg  |  Label: >60% samples >= {THRESH}")
    print(f"  Features : {N} (no MAP — hardware compatible)")
    print("="*60)

    # 1. Extract
    print("\n[1/5] Extracting features from CHARIS ...")
    X, y, pid = load_charis()
    print(f"\n  Total: {len(X):,} windows | {len(np.unique(pid))} patients")
    print(f"  Normal: {(y==0).sum():,} ({100*(y==0).mean():.1f}%)")
    print(f"  Abnormal: {(y==1).sum():,} ({100*(y==1).mean():.1f}%)")

    # 2. Split
    print("\n[2/5] Patient-level split 70/10/20 ...")
    gss = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss.split(X, y, groups=pid))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr_s, va_s = next(gss2.split(X[tv], y[tv], groups=pid[tv]))
    tr, va = tv[tr_s], tv[va_s]
    print(f"  Train: {len(np.unique(pid[tr]))} patients, {len(tr):,} windows")
    print(f"  Val  : {len(np.unique(pid[va]))} patients, {len(va):,} windows")
    print(f"  Test : {len(np.unique(pid[te]))} patients, {len(te):,} windows")

    # 3. SMOTE
    print("\n[3/5] SMOTE (exact 50/50, within-patient) ...")
    X_tr, y_tr = smote(X[tr], y[tr], pid[tr])

    # 4. Train
    device = get_device()
    print(f"\n[4/5] Training XGBoost [{device.upper()}] ...")
    bst, evals = train_model(X_tr, y_tr, X[va], y[va], device)

    # 5. Evaluate
    print("\n[5/5] Evaluating ...")
    metrics, thr = evaluate(bst, X[va], y[va], X[te], y[te], pid[te], evals, OUT_DIR)
    cv_res = cv(X, y, pid, device, thr)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(MODEL_DIR / "xgb_clean.json"))
    meta = {
        "date": date.today().isoformat(),
        "threshold_mmhg": THRESH,
        "label_rule": ">60% samples >= 20 mmHg [Ye et al. 2022]",
        "features": FEATURES,
        "smote": "within-patient exact 50/50 [PMC9744906]",
        "device": device,
        "metrics": metrics,
        "cv": cv_res,
        "papers": ["Ye et al. 2022 PMC9252333","PMC11986046","PMC9744906","BTF 2016"],
    }
    (MODEL_DIR / "xgb_clean_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Model -> {MODEL_DIR / 'xgb_clean.json'}")
    print(f"  Plot  -> {OUT_DIR / 'results.png'}")
    print(f"\n{'='*60}")
    print(f"  AUC : {metrics['auc']:.4f}   F1 : {metrics['f1']:.4f}")
    print(f"  Prec: {metrics['precision']:.4f}   Rec: {metrics['recall']:.4f}")
    print(f"  CV AUC: {cv_res['auc_mean']:.4f} ± {cv_res['auc_std']:.4f}")
    print(f"  CV F1 : {cv_res['f1_mean']:.4f} ± {cv_res['f1_std']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
