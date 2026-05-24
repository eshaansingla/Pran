"""
full_pipeline_qt.py
===================
Complete ICP pipeline: QT baked into training so hardware inference is genuine.

Steps
-----
1  Load CHARIS cache
2  Patient 70/10/20 split
3  Fit QuantileTransformer on train split only  -> transform all splits
4  SMOTE within-patient (on QT features)
5  Train XGBoost  ->  evaluate test set
6  LOPO CV  (per-fold QT fit, leakage-free)
7  Extract hardware features (detrend fix)  ->  apply same QT  ->  predict
8  Full metrics + honest verdict

Run
---
    cd "C:\\Users\\asus\\Documents\\GitHub\\Pran"; python full_pipeline_qt.py
"""
from __future__ import annotations
import json, sys, warnings
from datetime import date
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import xgboost as xgb
from scipy import signal as sp_signal
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_X   = Path("results/audit/cache/X.npy")
CACHE_Y   = Path("results/audit/cache/y.npy")
CACHE_PID = Path("results/audit/cache/pid.npy")
MODEL_DIR = Path("models")
OUT_DIR   = Path("results/qt_pipeline")
HW_CSVS   = [
    Path("C:/Users/asus/Downloads/icp_1_27min.csv"),
    Path("C:/Users/asus/Downloads/icp_2.csv"),
]

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]
N    = len(FEATURES)
SEED = 42
SESSION_NAMES = {0:"supine", 1:"head-up-30deg", 2:"head-down-10deg", 3:"valsalva+recovery"}

# ── Hardware signal constants ─────────────────────────────────────────────────
FS, WIN, STEP = 50, 500, 250
_nyq            = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS      = np.fft.rfftfreq(WIN, d=1.0/FS)
_FREQ_MASK  = (_FREQS >= 0.7) & (_FREQS <= 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware feature extraction (detrend fix applied)
# ─────────────────────────────────────────────────────────────────────────────
def extract_hw_window(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))

    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    pwr = np.abs(np.fft.rfft(ir_dt)) ** 2
    if not _FREQ_MASK.any(): return None
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])

    r        = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    coeffs   = pywt.wavedec(disp_dt, "db4", level=5)
    energies = [float(np.sum(cc**2)) for cc in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total
    cardiac_pow = energies[2] / total

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def load_hw_csv(path: Path):
    df = pd.read_csv(path, comment="#")
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)
    has_sess = "session_label" in df.columns
    feats, sessions = [], []
    n_win = (len(df) - WIN) // STEP + 1
    for w in range(n_win):
        s, e = w*STEP, w*STEP+WIN
        sl   = df.iloc[s:e]
        feat = extract_hw_window(sl["ir_raw"].values.astype(np.float32),
                                 sl["disp_raw"].values.astype(np.float32))
        if feat is None: continue
        feats.append(feat)
        sessions.append(int(sl["session_label"].mode()[0]) if has_sess else 0)
    return np.array(feats, dtype=np.float32), np.array(sessions, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def smote_balance(X, y, pid):
    Xs, ys = [], []
    for p in np.unique(pid):
        m = pid == p
        Xp, yp = X[m], y[m]
        n0, n1 = (yp==0).sum(), (yp==1).sum()
        if n0==0 or n1==0: Xs.append(Xp); ys.append(yp); continue
        k = max(1, min(5, min(n0,n1)-1))
        try:    Xp, yp = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xp, yp)
        except: Xp, yp = RandomOverSampler(random_state=SEED).fit_resample(Xp, yp)
        Xs.append(Xp); ys.append(yp)
    Xo = np.vstack(Xs).astype(np.float32); yo = np.concatenate(ys)
    if (yo==0).sum() != (yo==1).sum():
        Xo, yo = RandomOverSampler(random_state=SEED).fit_resample(Xo, yo)
        Xo = Xo.astype(np.float32)
    return Xo, yo


def get_device():
    try:
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            return "cpu"
        xgb.train({"device":"cuda","tree_method":"hist","verbosity":0},
                  xgb.DMatrix(np.zeros((4,N)), label=[0,1,0,1]), num_boost_round=1)
        return "cuda"
    except: return "cpu"


def xgb_params(device, seed=SEED):
    return {"objective":"binary:logistic","eval_metric":["logloss","auc"],
            "eta":0.05,"max_depth":5,"min_child_weight":3,
            "subsample":0.8,"colsample_bytree":0.8,
            "lambda":1.0,"alpha":0.1,"scale_pos_weight":1.0,
            "seed":seed,"tree_method":"hist","device":device,"verbosity":0}


def fit_qt(X_train: np.ndarray) -> QuantileTransformer:
    qt = QuantileTransformer(output_distribution="normal",
                             random_state=SEED, n_quantiles=min(1000, len(X_train)))
    qt.fit(X_train)
    return qt


def youden_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def kl_div(p, q, n=200):
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    if lo == hi: return 0.0
    grid = np.linspace(lo, hi, n)
    try:
        pk = gaussian_kde(p)(grid) + 1e-10
        qk = gaussian_kde(q)(grid) + 1e-10
    except: return float("nan")
    pk /= pk.sum(); qk /= qk.sum()
    return float(np.sum(pk * np.log(pk / qk)))


# ─────────────────────────────────────────────────────────────────────────────
# Main split evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_main_eval(X, y, pid, device):
    print("\n[STEP 2] Patient-level 70 / 10 / 20 split ...")
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(X, y, groups=pid))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr_s, va_s = next(gss2.split(X[tv], y[tv], groups=pid[tv]))
    tr, va = tv[tr_s], tv[va_s]
    print(f"  Train: {len(np.unique(pid[tr]))} patients  {len(tr):,} windows")
    print(f"  Val  : {len(np.unique(pid[va]))} patients  {len(va):,} windows")
    print(f"  Test : {len(np.unique(pid[te]))} patients  {len(te):,} windows")

    print("\n[STEP 3] Fitting QuantileTransformer on train split only ...")
    qt = fit_qt(X[tr])
    X_tr_qt = qt.transform(X[tr]).astype(np.float32)
    X_va_qt = qt.transform(X[va]).astype(np.float32)
    X_te_qt = qt.transform(X[te]).astype(np.float32)
    print("  QT fitted. All splits transformed to N(0,1) space.")

    print("\n[STEP 4] SMOTE within-patient on QT-transformed train ...")
    X_tr_sm, y_tr_sm = smote_balance(X_tr_qt, y[tr], pid[tr])
    print(f"  After SMOTE: {(y_tr_sm==0).sum():,} normal | {(y_tr_sm==1).sum():,} abnormal")

    print(f"\n[STEP 5] Training XGBoost [{device.upper()}] on QT features ...")
    d_tr = xgb.DMatrix(X_tr_sm, label=y_tr_sm, feature_names=FEATURES)
    d_va = xgb.DMatrix(X_va_qt, label=y[va],   feature_names=FEATURES)
    d_te = xgb.DMatrix(X_te_qt,                 feature_names=FEATURES)
    evals_res = {}
    bst = xgb.train(xgb_params(device), d_tr, num_boost_round=1000,
                    evals=[(d_tr,"train"),(d_va,"val")],
                    early_stopping_rounds=50, verbose_eval=100, evals_result=evals_res)
    print(f"  Best iteration: {bst.best_iteration}  val-AUC: {bst.best_score:.4f}")

    # Threshold on val, apply to test
    pv  = bst.predict(d_va)
    pt  = bst.predict(d_te)
    ptr = bst.predict(d_tr)
    thr = youden_threshold(y[va], pv)
    pred = (pt >= thr).astype(int)

    auc   = roc_auc_score(y[te], pt)
    auc_tr= roc_auc_score(y_tr_sm, ptr)
    f1    = f1_score(y[te], pred, zero_division=0)
    prec  = precision_score(y[te], pred, zero_division=0)
    rec   = recall_score(y[te], pred, zero_division=0)
    spec  = recall_score(1-y[te], 1-pred, zero_division=0)
    bacc  = balanced_accuracy_score(y[te], pred)
    ap    = average_precision_score(y[te], pt)
    cm    = confusion_matrix(y[te], pred)

    SEP = "="*65
    print(f"\n{SEP}")
    print("  MAIN SPLIT RESULTS  (QT-aligned features)")
    print(SEP)
    print(f"  Train AUC    : {auc_tr:.4f}  |  Test AUC   : {auc:.4f}  |  Gap: {auc_tr-auc:+.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  Specificity  : {spec:.4f}")
    print(f"  Balanced Acc : {bacc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  Threshold    : {thr:.4f}  (Youden's J on val)")
    print(f"\n{classification_report(y[te], pred, target_names=['Normal','Abnormal'], zero_division=0)}")

    return bst, qt, thr, evals_res, {
        "auc_train": round(float(auc_tr),4), "auc_test": round(float(auc),4),
        "f1": round(float(f1),4), "precision": round(float(prec),4),
        "recall": round(float(rec),4), "specificity": round(float(spec),4),
        "balanced_acc": round(float(bacc),4), "avg_precision": round(float(ap),4),
        "threshold": round(float(thr),4), "cm": cm.tolist(),
        "test_patients": sorted(np.unique(pid[te]).tolist()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOPO CV  (per-fold QT — no leakage)
# ─────────────────────────────────────────────────────────────────────────────
def run_lopo(X, y, pid, device):
    print("\n[STEP 6] LOPO CV (13 folds, per-fold QT fit, leakage-free) ...")
    logo = LeaveOneGroupOut()
    results = []

    for fold, (tr_i, te_i) in enumerate(logo.split(X, y, groups=pid)):
        test_pid = int(np.unique(pid[te_i])[0])
        n0, n1 = int((y[te_i]==0).sum()), int((y[te_i]==1).sum())
        if n0==0 or n1==0:
            print(f"  Patient {test_pid:2d}: SKIP (single class)"); continue

        # Fit QT on THIS fold's training data only
        qt_fold = fit_qt(X[tr_i])
        X_tr_qt = qt_fold.transform(X[tr_i]).astype(np.float32)
        X_te_qt = qt_fold.transform(X[te_i]).astype(np.float32)

        # SMOTE on fold training (QT-transformed)
        X_tr_sm, y_tr_sm = smote_balance(X_tr_qt, y[tr_i], pid[tr_i])

        # Inner 90/10 split for early stopping
        rng   = np.random.RandomState(SEED+fold)
        inner = rng.permutation(len(y_tr_sm)); cut = int(0.9*len(inner))
        d_tr  = xgb.DMatrix(X_tr_sm[inner[:cut]], label=y_tr_sm[inner[:cut]], feature_names=FEATURES)
        d_va  = xgb.DMatrix(X_tr_sm[inner[cut:]], label=y_tr_sm[inner[cut:]], feature_names=FEATURES)
        d_te  = xgb.DMatrix(X_te_qt, feature_names=FEATURES)

        bst_f = xgb.train(xgb_params(device, SEED+fold), d_tr, num_boost_round=500,
                          evals=[(d_va,"val")], early_stopping_rounds=50, verbose_eval=0)

        # Youden threshold on inner val (no leakage)
        pv   = bst_f.predict(d_va)
        thr  = youden_threshold(y_tr_sm[inner[cut:]], pv)
        probs= bst_f.predict(d_te)
        pred = (probs >= thr).astype(int)

        a   = roc_auc_score(y[te_i], probs)
        f1  = f1_score(y[te_i], pred, zero_division=0)
        rec = recall_score(y[te_i], pred, zero_division=0)
        spe = recall_score(1-y[te_i], 1-pred, zero_division=0)
        results.append({"patient":test_pid,"auc":a,"f1":f1,"recall":rec,"specificity":spe,
                        "n_windows":len(te_i),"n_abnormal":n1})
        print(f"  Patient {test_pid:2d}: AUC={a:.4f}  F1={f1:.4f}  "
              f"Rec={rec:.4f}  Spec={spe:.4f}  (n={len(te_i):,} abn={n1})")

    aucs = [r["auc"] for r in results]; f1s = [r["f1"] for r in results]
    rng2 = np.random.RandomState(SEED)
    boots_auc = [np.mean(rng2.choice(aucs, len(aucs), replace=True)) for _ in range(2000)]
    boots_f1  = [np.mean(rng2.choice(f1s,  len(f1s),  replace=True)) for _ in range(2000)]

    print(f"\n  LOPO Summary")
    print(f"  AUC : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
          f"95%CI [{np.percentile(boots_auc,2.5):.4f}, {np.percentile(boots_auc,97.5):.4f}]")
    print(f"  F1  : {np.mean(f1s):.4f}  +/- {np.std(f1s):.4f}  "
          f"95%CI [{np.percentile(boots_f1,2.5):.4f}, {np.percentile(boots_f1,97.5):.4f}]")

    return results, {
        "auc_mean": round(float(np.mean(aucs)),4),
        "auc_std":  round(float(np.std(aucs)),4),
        "auc_ci":   [round(float(np.percentile(boots_auc,2.5)),4),
                     round(float(np.percentile(boots_auc,97.5)),4)],
        "f1_mean":  round(float(np.mean(f1s)),4),
        "f1_std":   round(float(np.std(f1s)),4),
        "f1_ci":    [round(float(np.percentile(boots_f1,2.5)),4),
                     round(float(np.percentile(boots_f1,97.5)),4)],
        "per_patient": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hardware test
# ─────────────────────────────────────────────────────────────────────────────
def run_hw_test(bst, qt, thr, X_charis):
    print("\n[STEP 7] Hardware CSV test ...")
    SEP = "="*65
    charis_qt = qt.transform(X_charis)
    hw_results = {}

    for csv_path in HW_CSVS:
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found"); continue

        print(f"\n  {csv_path.name}")
        X_hw, sessions = load_hw_csv(csv_path)
        print(f"  {len(X_hw):,} windows extracted")

        # Feature stats
        print(f"\n  Feature stats (hardware, after detrend):")
        print(f"  {'Feature':<28} {'HW Mean':>10} {'HW Std':>10} {'CHARIS Mean':>12} {'CHARIS Std':>11}")
        print(f"  {'-'*75}")
        for i, f in enumerate(FEATURES):
            print(f"  {f:<28} {X_hw[:,i].mean():>10.4f} {X_hw[:,i].std():>10.4f} "
                  f"{X_charis[:,i].mean():>12.4f} {X_charis[:,i].std():>11.4f}")

        # Apply QT (same QT fitted on CHARIS training split)
        X_hw_qt = qt.transform(X_hw).astype(np.float32)

        # KL divergence in QT space
        print(f"\n  KL divergence (CHARIS vs hardware, AFTER QT alignment):")
        kl_vals = {}
        for i, f in enumerate(FEATURES):
            kl = kl_div(charis_qt[:,i], X_hw_qt[:,i])
            kl_vals[f] = round(float(kl), 4)
            flag = "OK  " if kl < 0.1 else "WARN" if kl < 0.5 else "HIGH"
            bar  = "#" * min(int(kl * 5), 25)
            print(f"  [{flag}] {f:<28} KL={kl:.3f}  {bar}")

        # Predict
        dm    = xgb.DMatrix(X_hw_qt, feature_names=FEATURES)
        probs = bst.predict(dm)
        preds = (probs >= thr).astype(int)
        pct   = 100 * preds.mean()

        print(f"\n{SEP}")
        print(f"  HARDWARE RESULTS  ->  {csv_path.name}")
        print(SEP)
        print(f"  Windows          : {len(probs):,}")
        print(f"  Threshold        : {thr:.4f}")
        print(f"  Mean P(abnormal) : {probs.mean():.4f}")
        print(f"  Min  P(abnormal) : {probs.min():.4f}")
        print(f"  Max  P(abnormal) : {probs.max():.4f}")
        print(f"  Flagged abnormal : {preds.sum():,} / {len(preds):,}  ({pct:.1f}%)")

        unique_sess = np.unique(sessions)
        sess_results = {}
        if len(unique_sess) > 1:
            print(f"\n  Per-session breakdown:")
            print(f"  {'Session':<22} {'Win':>6} {'Flag':>6} {'Flag%':>7} "
                  f"{'MeanP':>8} {'MinP':>7} {'MaxP':>7}")
            print(f"  {'-'*68}")
            for sess in sorted(unique_sess):
                m      = sessions == sess
                n_s    = m.sum()
                n_flag = preds[m].sum()
                p_pct  = 100 * preds[m].mean()
                mean_p = probs[m].mean()
                min_p  = probs[m].min()
                max_p  = probs[m].max()
                sname  = SESSION_NAMES.get(int(sess), f"sess_{sess}")
                tag    = " << Valsalva" if sess == 3 else ""
                print(f"  {sname:<22} {n_s:>6,} {n_flag:>6,} {p_pct:>6.1f}%  "
                      f"{mean_p:>8.4f} {min_p:>7.4f} {max_p:>7.4f}{tag}")
                sess_results[sname] = {
                    "n_windows": int(n_s), "flagged": int(n_flag),
                    "pct_flagged": round(float(p_pct),2),
                    "mean_prob": round(float(mean_p),4),
                    "min_prob":  round(float(min_p),4),
                    "max_prob":  round(float(max_p),4),
                }

        # Verdict
        print(f"\n  Specificity (windows correctly called normal): {100-pct:.1f}%")
        print(f"\n  VERDICT: ", end="")
        if pct < 5:
            v = "EXCELLENT -- model correctly identifies healthy subject as normal"
        elif pct < 15:
            v = "GOOD -- small false alarm rate, within acceptable domain-gap range"
        elif pct < 30:
            v = "BORDERLINE -- notable false alarms, hardware/signal quality check needed"
        else:
            v = "HIGH FALSE POSITIVE -- check sensor contact, cable noise, artifact filter"
        print(v)

        hw_results[csv_path.name] = {
            "n_windows": len(probs), "mean_prob": round(float(probs.mean()),4),
            "pct_flagged": round(float(pct),2), "kl_divergence": kl_vals,
            "specificity": round(float(100-pct),2), "verdict": v,
            "per_session": sess_results,
        }

    return hw_results


# ─────────────────────────────────────────────────────────────────────────────
# Save plots
# ─────────────────────────────────────────────────────────────────────────────
def save_plots(bst, evals_res, lopo_results, main_metrics, X_charis, qt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = plt.figure(figsize=(18, 10))
    figs.suptitle("XGBoost QT Pipeline -- CHARIS Results", fontsize=13, fontweight="bold")

    # Convergence
    ax = figs.add_subplot(2,3,1)
    ax.plot(evals_res["train"]["auc"], lw=1.5, label="Train")
    ax.plot(evals_res["val"]["auc"],   lw=1.5, label="Val")
    ax.axvline(bst.best_iteration, color="orange", ls="--", lw=1,
               label=f"Stop@{bst.best_iteration}")
    ax.set(title="Convergence (AUC)", xlabel="Round", ylabel="AUC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = figs.add_subplot(2,3,2)
    ax2.plot(evals_res["train"]["logloss"], lw=1.5, label="Train")
    ax2.plot(evals_res["val"]["logloss"],   lw=1.5, label="Val")
    ax2.axvline(bst.best_iteration, color="orange", ls="--", lw=1)
    ax2.set(title="Convergence (Logloss)", xlabel="Round", ylabel="Logloss"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # LOPO per-patient
    ax3 = figs.add_subplot(2,3,3)
    pats = [r["patient"] for r in lopo_results]
    aucs = [r["auc"]     for r in lopo_results]
    colors = ["#1565C0" if a >= 0.90 else "#C62828" for a in aucs]
    bars = ax3.bar(range(len(pats)), aucs, color=colors, alpha=0.85)
    ax3.axhline(np.mean(aucs), color="#1565C0", ls="-", lw=2, label=f"Mean={np.mean(aucs):.3f}")
    ax3.axhline(0.80, color="gray", ls="--", lw=1, label="Target=0.80")
    for bar, v in zip(bars, aucs):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax3.set(title="LOPO AUC per Patient", xlabel="Patient",
            ylabel="AUC", xticks=range(len(pats)),
            xticklabels=[f"P{p}" for p in pats], ylim=[0.5, 1.02])
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, axis="y")

    # Feature importance
    ax4 = figs.add_subplot(2,3,4)
    gain = bst.get_score(importance_type="gain")
    total_g = sum(gain.values()) + 1e-12
    vals = [gain.get(f,0)/total_g*100 for f in FEATURES]
    ax4.barh(FEATURES[::-1], vals[::-1], color="#2C5282", alpha=0.85)
    ax4.set(title="Feature Importance (Gain %)", xlabel="Gain %"); ax4.grid(alpha=0.3, axis="x")
    for i, v in enumerate(vals[::-1]):
        ax4.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # QT feature distributions (CHARIS before/after)
    ax5 = figs.add_subplot(2,3,5)
    charis_qt = qt.transform(X_charis)
    for i, f in enumerate(FEATURES):
        ax5.hist(charis_qt[:,i], bins=50, alpha=0.4, density=True, label=f)
    ax5.set(title="CHARIS Features After QT (N(0,1))", xlabel="QT Value"); ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    # LOPO F1 per patient
    ax6 = figs.add_subplot(2,3,6)
    f1s = [r["f1"] for r in lopo_results]
    colors2 = ["#2E7D32" if f >= 0.75 else "#C62828" for f in f1s]
    bars2 = ax6.bar(range(len(pats)), f1s, color=colors2, alpha=0.85)
    ax6.axhline(np.mean(f1s), color="#2E7D32", ls="-", lw=2, label=f"Mean={np.mean(f1s):.3f}")
    ax6.axhline(0.75, color="gray", ls="--", lw=1, label="Target=0.75")
    for bar, v in zip(bars2, f1s):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax6.set(title="LOPO F1 per Patient", xlabel="Patient",
            ylabel="F1", xticks=range(len(pats)),
            xticklabels=[f"P{p}" for p in pats], ylim=[0.0, 1.08])
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "qt_pipeline_results.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Plot saved -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP = "="*65
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  XGBoost QT Pipeline  |  CHARIS Train -> Hardware Test")
    print("  QT baked into training: genuine hardware inference")
    print(SEP)

    # Step 1: Load CHARIS cache
    print("\n[STEP 1] Loading CHARIS cache ...")
    for p in [CACHE_X, CACHE_Y, CACHE_PID]:
        if not p.exists():
            print(f"  ERROR: {p} not found. Run audit_plots.py first.")
            sys.exit(1)
    X   = np.load(CACHE_X)
    y   = np.load(CACHE_Y)
    pid = np.load(CACHE_PID)
    print(f"  {len(X):,} windows | {len(np.unique(pid))} patients")
    print(f"  Normal: {(y==0).sum():,}  Abnormal: {(y==1).sum():,}")

    device = get_device()
    print(f"  Device: {device.upper()}")

    # Steps 2-5: main split
    bst, qt, thr, evals_res, main_m = run_main_eval(X, y, pid, device)

    # Step 6: LOPO
    lopo_results, lopo_m = run_lopo(X, y, pid, device)

    # Step 7: hardware
    hw_results = run_hw_test(bst, qt, thr, X)

    # Save model + QT
    bst.save_model(str(MODEL_DIR / "xgb_qt.json"))
    import pickle
    with open(MODEL_DIR / "qt_scaler.pkl", "wb") as f:
        pickle.dump(qt, f)
    print(f"\n  Model saved -> {MODEL_DIR / 'xgb_qt.json'}")
    print(f"  QT scaler  -> {MODEL_DIR / 'qt_scaler.pkl'}")

    # Plots
    save_plots(bst, evals_res, lopo_results, main_m, X, qt, OUT_DIR)

    # Save JSON
    meta = {
        "date": date.today().isoformat(),
        "alignment": "QuantileTransformer(N(0,1)) fitted on train split only",
        "main_split": main_m,
        "lopo": lopo_m,
        "hardware": hw_results,
    }
    (OUT_DIR / "qt_results.json").write_text(json.dumps(meta, indent=2))

    # Final summary
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(SEP)
    print(f"  CHARIS Test AUC   : {main_m['auc_test']:.4f}  (train-test gap: {main_m['auc_train']-main_m['auc_test']:+.4f})")
    print(f"  CHARIS Test F1    : {main_m['f1']:.4f}")
    print(f"  CHARIS Recall     : {main_m['recall']:.4f}")
    print(f"  CHARIS Specificity: {main_m['specificity']:.4f}")
    print(f"  LOPO AUC          : {lopo_m['auc_mean']:.4f} +/- {lopo_m['auc_std']:.4f}  "
          f"95%CI [{lopo_m['auc_ci'][0]:.4f}, {lopo_m['auc_ci'][1]:.4f}]")
    print(f"  LOPO F1           : {lopo_m['f1_mean']:.4f} +/- {lopo_m['f1_std']:.4f}  "
          f"95%CI [{lopo_m['f1_ci'][0]:.4f}, {lopo_m['f1_ci'][1]:.4f}]")
    print(f"\n  Hardware Results:")
    for fname, r in hw_results.items():
        print(f"  {fname:<30} flagged={r['pct_flagged']:.1f}%  "
              f"specificity={r['specificity']:.1f}%  mean_P={r['mean_prob']:.4f}")
        print(f"    -> {r['verdict']}")
    print(f"\n  Results -> {OUT_DIR}/qt_results.json")
    print(f"  Plot    -> {OUT_DIR}/qt_pipeline_results.png")
    print(SEP)


if __name__ == "__main__":
    main()
