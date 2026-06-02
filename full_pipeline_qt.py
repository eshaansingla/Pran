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
import json, re, sys, warnings
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC
from scipy.stats import wilcoxon as _wilcoxon, norm as _norm
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_X   = Path("results/audit/cache/X.npy")
CACHE_Y   = Path("results/audit/cache/y.npy")
CACHE_PID = Path("results/audit/cache/pid.npy")
MODEL_DIR      = Path("models")
BASELINE_CACHE = Path("models/baselines")
OUT_DIR   = Path("results/qt_pipeline")
HW_DIR  = Path("hw-tests")   # all *.csv files here are auto-tested
HW_META = {                  # override metadata for specific files (comorbidities, notes, etc.)
    "icp_1_27min.csv": {"label": "Subject 1",  "age": "19-21", "gender": "?", "profile": "Healthy young adult"},
    "icp_2.csv":       {"label": "Subject 2",  "age": "19-21", "gender": "?", "profile": "Healthy young adult"},
    "icp_3.csv":       {"label": "Subject 3",  "age": "65-75", "gender": "?", "profile": "Elderly adult"},
    "icp_4.csv":       {"label": "Subject 4",  "age": "65-75", "gender": "?", "profile": "Elderly adult, prior haemorrhage"},
    "icp_5.csv":       {"label": "Subject 5",  "age": "?",     "gender": "?", "profile": "Unknown"},
    "icp_11_72_F.csv": {"label": "Subject 11", "age": "72",    "gender": "F", "profile": "Female | Hypertension/Diabetes (unconfirmed which)"},
    "icp_12_75_M.csv": {"label": "Subject 12", "age": "75",    "gender": "M", "profile": "Male   | Hypertension/Diabetes (unconfirmed which)"},
}


def parse_hw_meta(path: Path) -> dict:
    """HW_META takes priority; otherwise parse icp_{num}_{age}_{gender}.csv automatically."""
    if path.name in HW_META:
        return HW_META[path.name]
    m = re.match(r"icp_(\d+)_(\d+)_([MF])$", path.stem, re.IGNORECASE)
    if m:
        num, age, gender = m.group(1), m.group(2), m.group(3).upper()
        return {"label": f"Subject {num}", "age": age, "gender": gender,
                "profile": "Male" if gender == "M" else "Female"}
    return {"label": path.stem, "age": "?", "gender": "?", "profile": "Unknown"}

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


def brier_ece(y_true, probs, n_bins=10):
    """Returns (Brier score, ECE). Lower = better for both."""
    y = np.asarray(y_true); p = np.asarray(probs)
    brier = float(np.mean((p - y) ** 2))
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if m.sum() == 0: continue
        ece += m.sum() * abs(y[m].mean() - p[m].mean())
    ece /= len(y)
    return round(brier, 4), round(ece, 4)


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
# DeLong test  (subsampled for speed — 10k per class, O(n²) on sample)
# ─────────────────────────────────────────────────────────────────────────────
def delong_test(y_true, pred_a, pred_b, max_per_class=10_000):
    """DeLong 1988 — returns (auc_a, auc_b, z, p)."""
    y  = np.asarray(y_true, dtype=np.int32)
    pa = np.asarray(pred_a,  dtype=np.float64)
    pb = np.asarray(pred_b,  dtype=np.float64)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(SEED)
    if len(pos_idx) > max_per_class:
        pos_idx = rng.choice(pos_idx, max_per_class, replace=False)
    if len(neg_idx) > max_per_class:
        neg_idx = rng.choice(neg_idx, max_per_class, replace=False)
    n1, n0 = len(pos_idx), len(neg_idx)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def _place(scores_pos, scores_neg):
        # V10[i] = fraction of negatives beaten by positive i
        V10 = ((scores_pos[:, None] > scores_neg[None, :]).mean(1) +
               0.5 * (scores_pos[:, None] == scores_neg[None, :]).mean(1))
        V01 = ((scores_neg[:, None] < scores_pos[None, :]).mean(1) +
               0.5 * (scores_neg[:, None] == scores_pos[None, :]).mean(1))
        return V10, V01

    V10_a, V01_a = _place(pa[pos_idx], pa[neg_idx])
    V10_b, V01_b = _place(pb[pos_idx], pb[neg_idx])

    auc_a, auc_b = float(V10_a.mean()), float(V10_b.mean())
    S10 = np.cov(np.stack([V10_a, V10_b]), ddof=1)
    S01 = np.cov(np.stack([V01_a, V01_b]), ddof=1)

    var_diff = ((S10[0,0] + S10[1,1] - 2*S10[0,1]) / n1 +
                (S01[0,0] + S01[1,1] - 2*S01[0,1]) / n0)
    if var_diff <= 0:
        return auc_a, auc_b, float("nan"), float("nan")
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - _norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p


def run_statistical_tests(lopo_y, lopo_probs, baseline_pooled,
                          lopo_per_fold, baseline_m):
    """DeLong test on pooled LOPO predictions + Wilcoxon on per-fold AUCs."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  STATISTICAL TESTS  (XGBoost vs each baseline)")
    print(SEP)
    lopo_y  = np.array(lopo_y);  lopo_probs = np.array(lopo_probs)
    xgb_fold_aucs = [r["auc"] for r in lopo_per_fold]
    stat_results  = {}

    for name, (bl_y, bl_probs) in baseline_pooled.items():
        bl_y    = np.array(bl_y);    bl_probs = np.array(bl_probs)
        bl_fold = [r["auc"] for r in baseline_m[name]["per_patient"]]

        # DeLong on pooled cross-validated predictions
        auc_x, auc_b, z, p_dl = delong_test(lopo_y, lopo_probs, bl_probs)

        # Wilcoxon signed-rank on per-fold AUC pairs
        diffs = np.array(xgb_fold_aucs) - np.array(bl_fold)
        try:
            _, p_wx = _wilcoxon(diffs, alternative="greater")
        except Exception:
            p_wx = float("nan")

        sig_dl = "***" if p_dl < 0.001 else ("**" if p_dl < 0.01 else ("*" if p_dl < 0.05 else "ns"))
        sig_wx = "***" if p_wx < 0.001 else ("**" if p_wx < 0.01 else ("*" if p_wx < 0.05 else "ns"))

        print(f"\n  XGBoost vs {name}:")
        print(f"    DeLong test  : z={z:+.3f}  p={p_dl:.4f}  {sig_dl}")
        print(f"    Wilcoxon     : p={p_wx:.4f}  {sig_wx}  (per-fold AUCs, one-tailed)")
        print(f"    ΔAUC (pooled): {auc_x - auc_b:+.4f}  ({auc_x:.4f} vs {auc_b:.4f})")

        stat_results[f"XGBoost_vs_{name}"] = {
            "delong_z": round(float(z), 4),     "delong_p": round(float(p_dl), 6),
            "wilcoxon_p": round(float(p_wx), 6),"delta_auc": round(float(auc_x-auc_b), 4),
        }

    return stat_results


# ─────────────────────────────────────────────────────────────────────────────
# Feature ablation  (drop-one LOPO — cached per feature)
# ─────────────────────────────────────────────────────────────────────────────
def run_feature_ablation(X, y, pid, device, full_auc):
    import pickle as _pkl
    ABLATION_BASE = Path("models/ablation")
    print("\n[STEP 8] Feature Ablation (drop-one LOPO, XGBoost) ...")
    logo    = LeaveOneGroupOut()
    ablation_results = {}

    for drop_i, drop_feat in enumerate(FEATURES):
        feat_idx  = [i for i in range(len(FEATURES)) if i != drop_i]
        feat_sub  = [FEATURES[i] for i in feat_idx]
        X_sub     = X[:, feat_idx]
        cache_dir = ABLATION_BASE / f"drop_{drop_feat}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fold_aucs = []

        for fold, (tr_i, te_i) in enumerate(logo.split(X_sub, y, groups=pid)):
            test_pid = int(np.unique(pid[te_i])[0])
            n0, n1   = int((y[te_i]==0).sum()), int((y[te_i]==1).sum())
            if n0 == 0 or n1 == 0: continue

            cache_p = cache_dir / f"fold{fold:02d}_pid{test_pid:02d}.json"
            thr_p   = cache_dir / f"fold{fold:02d}_pid{test_pid:02d}_thr.pkl"
            qt_fold = fit_qt(X_sub[tr_i])
            X_te_qt = qt_fold.transform(X_sub[te_i]).astype(np.float32)
            d_te    = xgb.DMatrix(X_te_qt, feature_names=feat_sub)

            if cache_p.exists() and thr_p.exists():
                bst_f = xgb.Booster(); bst_f.load_model(str(cache_p))
                thr   = _pkl.load(open(thr_p, "rb"))
            else:
                X_tr_qt = qt_fold.transform(X_sub[tr_i]).astype(np.float32)
                X_tr_sm, y_tr_sm = smote_balance(X_tr_qt, y[tr_i], pid[tr_i])
                rng   = np.random.RandomState(SEED + fold)
                inner = rng.permutation(len(y_tr_sm)); cut = int(0.9*len(inner))
                d_tr  = xgb.DMatrix(X_tr_sm[inner[:cut]], label=y_tr_sm[inner[:cut]], feature_names=feat_sub)
                d_va  = xgb.DMatrix(X_tr_sm[inner[cut:]], label=y_tr_sm[inner[cut:]], feature_names=feat_sub)
                bst_f = xgb.train(xgb_params(device, SEED+fold), d_tr, num_boost_round=500,
                                  evals=[(d_va,"val")], early_stopping_rounds=50, verbose_eval=0)
                thr   = youden_threshold(y_tr_sm[inner[cut:]], bst_f.predict(d_va))
                bst_f.save_model(str(cache_p)); _pkl.dump(thr, open(thr_p,"wb"))

            fold_aucs.append(roc_auc_score(y[te_i], bst_f.predict(d_te)))

        mean_auc = float(np.mean(fold_aucs))
        delta    = mean_auc - full_auc
        ablation_results[drop_feat] = {
            "auc_mean": round(mean_auc, 4),
            "auc_std":  round(float(np.std(fold_aucs)), 4),
            "delta_vs_full": round(delta, 4),
        }
        print(f"  Drop {drop_feat:<26}: AUC {mean_auc:.4f} ± {np.std(fold_aucs):.4f}  "
              f"(Δ={delta:+.4f})")

    print(f"\n  Full model AUC: {full_auc:.4f}")
    return ablation_results


# ─────────────────────────────────────────────────────────────────────────────
# Valsalva statistical analysis  (paired Wilcoxon across all subjects)
# ─────────────────────────────────────────────────────────────────────────────
def run_hw_valsalva_stats(hw_results):
    valsalva, baseline = [], []
    for fname, r in hw_results.items():
        sess = r.get("per_session", {})
        if "valsalva+recovery" not in sess: continue
        v_pct = sess["valsalva+recovery"]["pct_flagged"]
        others = [s["pct_flagged"] for k,s in sess.items() if k != "valsalva+recovery"]
        if not others: continue
        valsalva.append(v_pct)
        baseline.append(float(np.mean(others)))

    if len(valsalva) < 4:
        print("  Valsalva stats: insufficient data"); return {}

    valsalva = np.array(valsalva); baseline = np.array(baseline)
    diffs    = valsalva - baseline
    try:
        _, p_wx = _wilcoxon(diffs, alternative="greater")
    except Exception:
        p_wx = float("nan")

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  VALSALVA ANALYSIS  (ICP elevation maneuver validation)")
    print(SEP)
    print(f"  Subjects with session data : {len(valsalva)}")
    print(f"  Mean valsalva flag%        : {valsalva.mean():.1f}%")
    print(f"  Mean baseline flag%        : {baseline.mean():.1f}%")
    print(f"  Mean difference            : {diffs.mean():+.1f}%")
    print(f"  Subjects where val > base  : {(diffs > 0).sum()}/{len(diffs)}")
    sig = "***" if p_wx < 0.001 else ("**" if p_wx < 0.01 else ("*" if p_wx < 0.05 else "ns"))
    print(f"  Wilcoxon (one-tailed)      : p={p_wx:.4f}  {sig}")
    print(f"  Interpretation: {'Valsalva maneuver significantly elevates ICP proxy signal (p<0.05)' if p_wx < 0.05 else 'No significant elevation detected'}")

    return {
        "n_subjects": len(valsalva),
        "mean_valsalva_pct": round(float(valsalva.mean()), 2),
        "mean_baseline_pct": round(float(baseline.mean()), 2),
        "mean_diff": round(float(diffs.mean()), 2),
        "subjects_elevated": int((diffs > 0).sum()),
        "wilcoxon_p": round(float(p_wx), 6),
    }


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

    import pickle as _pkl
    _xgb_path = MODEL_DIR / "xgb_qt.json"
    _thr_path  = MODEL_DIR / "xgb_qt_thr.pkl"
    d_tr = xgb.DMatrix(X_tr_sm, label=y_tr_sm, feature_names=FEATURES)
    d_va = xgb.DMatrix(X_va_qt, label=y[va],   feature_names=FEATURES)
    d_te = xgb.DMatrix(X_te_qt,                 feature_names=FEATURES)

    if _xgb_path.exists() and _thr_path.exists():
        print(f"\n[STEP 5] Loading cached XGBoost model [{_xgb_path}] ...")
        bst = xgb.Booster(); bst.load_model(str(_xgb_path))
        thr, evals_res = _pkl.load(open(_thr_path, "rb"))
    else:
        print(f"\n[STEP 5] Training XGBoost [{device.upper()}] on QT features ...")
        evals_res = {}
        bst = xgb.train(xgb_params(device), d_tr, num_boost_round=1000,
                        evals=[(d_tr,"train"),(d_va,"val")],
                        early_stopping_rounds=50, verbose_eval=100, evals_result=evals_res)
        print(f"  Best iteration: {bst.best_iteration}  val-AUC: {bst.best_score:.4f}")
        thr = youden_threshold(y[va], bst.predict(d_va))

    # Threshold on val, apply to test
    pv  = bst.predict(d_va)
    pt  = bst.predict(d_te)
    ptr = bst.predict(d_tr)
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

    return bst, qt, thr, evals_res, pt, y[te], {
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
    import pickle as _pkl
    XGB_LOPO_CACHE = MODEL_DIR / "xgb_lopo"
    XGB_LOPO_CACHE.mkdir(parents=True, exist_ok=True)
    print("\n[STEP 6] LOPO CV (13 folds, per-fold QT fit, leakage-free) ...")
    logo = LeaveOneGroupOut()
    results = []
    pooled_y, pooled_probs = [], []

    for fold, (tr_i, te_i) in enumerate(logo.split(X, y, groups=pid)):
        test_pid = int(np.unique(pid[te_i])[0])
        n0, n1 = int((y[te_i]==0).sum()), int((y[te_i]==1).sum())
        if n0==0 or n1==0:
            print(f"  Patient {test_pid:2d}: SKIP (single class)"); continue

        cache_path = XGB_LOPO_CACHE / f"fold{fold:02d}_pid{test_pid:02d}.json"
        thr_path   = XGB_LOPO_CACHE / f"fold{fold:02d}_pid{test_pid:02d}_thr.pkl"

        # Fit QT on THIS fold's training data only
        qt_fold = fit_qt(X[tr_i])
        X_te_qt = qt_fold.transform(X[te_i]).astype(np.float32)
        d_te    = xgb.DMatrix(X_te_qt, feature_names=FEATURES)

        if cache_path.exists() and thr_path.exists():
            bst_f = xgb.Booster(); bst_f.load_model(str(cache_path))
            thr   = _pkl.load(open(thr_path, "rb"))
            status = "cached"
        else:
            X_tr_qt = qt_fold.transform(X[tr_i]).astype(np.float32)
            X_tr_sm, y_tr_sm = smote_balance(X_tr_qt, y[tr_i], pid[tr_i])
            rng   = np.random.RandomState(SEED+fold)
            inner = rng.permutation(len(y_tr_sm)); cut = int(0.9*len(inner))
            d_tr  = xgb.DMatrix(X_tr_sm[inner[:cut]], label=y_tr_sm[inner[:cut]], feature_names=FEATURES)
            d_va  = xgb.DMatrix(X_tr_sm[inner[cut:]], label=y_tr_sm[inner[cut:]], feature_names=FEATURES)
            bst_f = xgb.train(xgb_params(device, SEED+fold), d_tr, num_boost_round=500,
                              evals=[(d_va,"val")], early_stopping_rounds=50, verbose_eval=0)
            thr   = youden_threshold(y_tr_sm[inner[cut:]], bst_f.predict(d_va))
            bst_f.save_model(str(cache_path))
            _pkl.dump(thr, open(thr_path, "wb"))
            status = "trained"

        probs= bst_f.predict(d_te)
        pred = (probs >= thr).astype(int)
        a   = roc_auc_score(y[te_i], probs)
        f1  = f1_score(y[te_i], pred, zero_division=0)
        rec = recall_score(y[te_i], pred, zero_division=0)
        spe = recall_score(1-y[te_i], 1-pred, zero_division=0)
        results.append({"patient":test_pid,"auc":a,"f1":f1,"recall":rec,"specificity":spe,
                        "n_windows":len(te_i),"n_abnormal":n1})
        pooled_y.extend(y[te_i].tolist()); pooled_probs.extend(probs.tolist())
        print(f"  Patient {test_pid:2d}: AUC={a:.4f}  F1={f1:.4f}  "
              f"Rec={rec:.4f}  Spec={spe:.4f}  (n={len(te_i):,} abn={n1})  [{status}]")

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
    }, pooled_y, pooled_probs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline LOPO CV  (LogReg | RandomForest | LinearSVM — same protocol as XGB)
# ─────────────────────────────────────────────────────────────────────────────
_BASELINES = {
    "LogReg":       LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED),
    "RandForest":   RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
    "LinearSVM":    CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED), cv=3),
}


def run_baselines_lopo(X, y, pid):
    import pickle as _pkl
    BASELINE_CACHE.mkdir(parents=True, exist_ok=True)
    print("\n[STEP 6b] Baseline LOPO CV  (LogReg | RandForest | LinearSVM) ...")
    logo        = LeaveOneGroupOut()
    results     = {name: [] for name in _BASELINES}
    pooled      = {name: ([], []) for name in _BASELINES}  # (y_list, prob_list)
    n_folds     = len(np.unique(pid))

    for fold, (tr_i, te_i) in enumerate(logo.split(X, y, groups=pid)):
        test_pid = int(np.unique(pid[te_i])[0])
        n0, n1   = int((y[te_i]==0).sum()), int((y[te_i]==1).sum())
        if n0 == 0 or n1 == 0:
            print(f"  Patient {test_pid:2d}: SKIP (single class)"); continue

        qt_fold  = fit_qt(X[tr_i])
        X_tr_qt  = qt_fold.transform(X[tr_i]).astype(np.float32)
        X_te_qt  = qt_fold.transform(X[te_i]).astype(np.float32)

        # Inner 90/10 split for Youden threshold (mirrors XGB protocol)
        # Built from SMOTE data — only computed if at least one model needs training
        X_tr_sm = y_tr_sm = X_itr = y_itr = X_iva = y_iva = None

        row = []
        for name, clf in _BASELINES.items():
            cache_path = BASELINE_CACHE / f"fold{fold:02d}_pid{test_pid:02d}_{name}.pkl"

            if cache_path.exists():
                with open(cache_path, "rb") as fh:
                    clf_f, thr_cached = _pkl.load(fh)
                pt   = clf_f.predict_proba(X_te_qt)[:, 1]
                thr  = thr_cached
                status = "cached"
            else:
                # Lazy SMOTE — only run once per fold, shared across models
                if X_tr_sm is None:
                    X_tr_sm, y_tr_sm = smote_balance(X_tr_qt, y[tr_i], pid[tr_i])
                    rng = np.random.RandomState(SEED + fold)
                    idx = rng.permutation(len(y_tr_sm)); cut = int(0.9 * len(idx))
                    X_itr, y_itr = X_tr_sm[idx[:cut]], y_tr_sm[idx[:cut]]
                    X_iva, y_iva = X_tr_sm[idx[cut:]], y_tr_sm[idx[cut:]]

                clf_f = clone(clf)
                clf_f.fit(X_itr, y_itr)
                pv   = clf_f.predict_proba(X_iva)[:, 1]
                thr  = youden_threshold(y_iva, pv)
                pt   = clf_f.predict_proba(X_te_qt)[:, 1]
                with open(cache_path, "wb") as fh:
                    _pkl.dump((clf_f, thr), fh)
                status = "trained"

            pred = (pt >= thr).astype(int)
            a   = roc_auc_score(y[te_i], pt)
            f1  = f1_score(y[te_i], pred, zero_division=0)
            rec = recall_score(y[te_i], pred, zero_division=0)
            spe = recall_score(1 - y[te_i], 1 - pred, zero_division=0)
            results[name].append({"patient": test_pid, "auc": a, "f1": f1,
                                   "recall": rec, "specificity": spe})
            pooled[name][0].extend(y[te_i].tolist())
            pooled[name][1].extend(pt.tolist())
            row.append(f"{name}={a:.4f}({status[0]})")
        print(f"  P{test_pid:02d} ({fold+1:2d}/{n_folds}): " + "  ".join(row))

    print("\n  Baseline LOPO Summary (AUC mean ± std):")
    summary = {}
    for name, res in results.items():
        aucs = [r["auc"] for r in res]; f1s = [r["f1"] for r in res]
        rng2 = np.random.RandomState(SEED)
        b_auc = [np.mean(rng2.choice(aucs, len(aucs), replace=True)) for _ in range(2000)]
        summary[name] = {
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std":  round(float(np.std(aucs)), 4),
            "auc_ci":   [round(float(np.percentile(b_auc, 2.5)), 4),
                         round(float(np.percentile(b_auc, 97.5)), 4)],
            "f1_mean":  round(float(np.mean(f1s)), 4),
            "f1_std":   round(float(np.std(f1s)), 4),
            "per_patient": res,
        }
        print(f"  {name:<14}: AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
              f"F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    return summary, pooled


# ─────────────────────────────────────────────────────────────────────────────
# Hardware test
# ─────────────────────────────────────────────────────────────────────────────
def run_hw_test(bst, qt, thr, X_charis):
    print("\n[STEP 7] Hardware Validation ...")
    SEP  = "="*65
    SEP2 = "-"*65
    charis_qt = qt.transform(X_charis)
    hw_results = {}

    hw_csvs = sorted(HW_DIR.glob("*.csv")) if HW_DIR.exists() else []
    if not hw_csvs:
        print(f"  No CSV files found in {HW_DIR}/"); return {}

    for csv_path in hw_csvs:
        meta  = parse_hw_meta(csv_path)
        X_hw, sessions = load_hw_csv(csv_path)

        # Apply QT (same QT fitted on CHARIS training split)
        X_hw_qt = qt.transform(X_hw).astype(np.float32)

        # KL divergence (stored in results, not printed to keep output clean)
        kl_vals = {}
        for i, f in enumerate(FEATURES):
            kl = kl_div(charis_qt[:,i], X_hw_qt[:,i])
            kl_vals[f] = round(float(kl), 4) if np.isfinite(kl) else None

        # Predict
        dm    = xgb.DMatrix(X_hw_qt, feature_names=FEATURES)
        probs = bst.predict(dm)
        preds = (probs >= thr).astype(int)
        pct   = 100 * preds.mean()

        gender_str = f"  |  {meta['gender']}" if meta.get("gender", "?") != "?" else ""
        print(f"\n{SEP}")
        print(f"  {meta['label']}  |  Age: {meta['age']}{gender_str}  |  {meta['profile']}")
        print(SEP)
        print(f"  Windows analysed  : {len(probs):,}")
        print(f"  Mean P(ICP anomaly): {probs.mean():.4f}  (threshold: {thr:.4f})")
        print(f"  Windows flagged    : {preds.sum():,} / {len(preds):,}  ({pct:.1f}%)")

        unique_sess = np.unique(sessions)
        sess_results = {}
        if len(unique_sess) > 1:
            print(f"\n  Per-session breakdown:")
            print(f"  {'Session':<22} {'Win':>6} {'Flagged':>8} {'Flag%':>7} {'Mean P':>8}")
            print(f"  {'-'*55}")
            for sess in sorted(unique_sess):
                m      = sessions == sess
                n_s    = m.sum()
                n_flag = preds[m].sum()
                p_pct  = 100 * preds[m].mean()
                mean_p = probs[m].mean()
                sname  = SESSION_NAMES.get(int(sess), f"sess_{sess}")
                tag    = "  [ICP elevation expected]" if sess == 3 else ""
                print(f"  {sname:<22} {n_s:>6,} {n_flag:>8,} {p_pct:>6.1f}%  {mean_p:>8.4f}{tag}")
                sess_results[sname] = {
                    "n_windows": int(n_s), "flagged": int(n_flag),
                    "pct_flagged": round(float(p_pct),2),
                    "mean_prob": round(float(mean_p),4),
                }

        # Clinical interpretation — profile-aware
        profile_low = meta["profile"].lower()
        has_haemorrhage = "haemorrhage" in profile_low or "hemorrhage" in profile_low
        has_comorbidity = any(x in profile_low for x in ["hypertension", "diabetes", "htn", "dm"])

        print(f"\n  Clinical Interpretation:")
        if pct < 10:
            v = "Normal ICP profile -- consistent with healthy subject"
            interp = "Model correctly identifies subject as neurologically normal."
        elif pct < 20:
            v = "Mildly elevated ICP signals -- within expected physiological range"
            interp = "Minor elevation consistent with age-related vascular changes."
        elif pct < 40:
            v = "Moderately elevated ICP signals -- age-related or subclinical changes detected"
            interp = "Elevation consistent with reduced cerebrovascular compliance in elderly subjects."
        else:
            v = "Significantly elevated ICP signals -- altered cerebrovascular compliance detected"
            if has_haemorrhage:
                interp = "Strong ICP anomaly signal consistent with known haemorrhagic history and altered TM-ICP coupling."
            elif has_comorbidity:
                interp = "Elevated signal consistent with hypertension/diabetes-related cerebrovascular remodelling and reduced compliance."
            else:
                interp = "Elevated signal consistent with age-related reduction in cerebrovascular compliance and ICP dynamics."
        print(f"  {v}")
        print(f"  {interp}")

        hw_results[csv_path.name] = {
            "subject": meta["label"], "age": meta["age"],
            "gender": meta.get("gender", "?"), "profile": meta["profile"],
            "n_windows": len(probs), "mean_prob": round(float(probs.mean()),4),
            "pct_flagged": round(float(pct),2),
            "kl_divergence": kl_vals, "verdict": v,
            "per_session": sess_results,
        }

    return hw_results


# ─────────────────────────────────────────────────────────────────────────────
# Save plots
# ─────────────────────────────────────────────────────────────────────────────
def save_calibration_plot(test_probs, test_labels, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    frac_pos, mean_pred = calibration_curve(test_labels, test_probs, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "s-", color="#1565C0", lw=2, label="XGBoost")
    ax.plot([0,1], [0,1], "k--", lw=1, label="Perfect calibration")
    ax.set(title="Reliability Diagram (Calibration Curve)",
           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           xlim=[0,1], ylim=[0,1])
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    p = out_dir / "fig_calibration.png"
    plt.tight_layout(); plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Calibration plot -> {p}")


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
# Baseline comparison plot
# ─────────────────────────────────────────────────────────────────────────────
def save_comparison_plot(lopo_m, baseline_m, out_dir):
    models     = ["LogReg", "RandForest", "LinearSVM", "XGBoost"]
    auc_means  = [baseline_m[m]["auc_mean"] for m in models[:3]] + [lopo_m["auc_mean"]]
    auc_stds   = [baseline_m[m]["auc_std"]  for m in models[:3]] + [lopo_m["auc_std"]]
    f1_means   = [baseline_m[m]["f1_mean"]  for m in models[:3]] + [lopo_m["f1_mean"]]
    f1_stds    = [baseline_m[m]["f1_std"]   for m in models[:3]] + [lopo_m["f1_std"]]
    colors     = ["#78909C", "#78909C", "#78909C", "#1565C0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LOPO CV Model Comparison — XGBoost vs Baselines", fontsize=13, fontweight="bold")

    x = np.arange(len(models))
    bars1 = ax1.bar(x, auc_means, yerr=auc_stds, color=colors, alpha=0.85,
                    capsize=5, error_kw={"elinewidth": 1.5})
    ax1.axhline(0.80, color="gray", ls="--", lw=1, label="Target 0.80")
    ax1.set(title="LOPO AUC (mean ± std)", ylabel="AUC", ylim=[0.5, 1.05],
            xticks=x, xticklabels=models)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")
    for bar, v, s in zip(bars1, auc_means, auc_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, v + s + 0.012,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, f1_means, yerr=f1_stds, color=colors, alpha=0.85,
                    capsize=5, error_kw={"elinewidth": 1.5})
    ax2.axhline(0.75, color="gray", ls="--", lw=1, label="Target 0.75")
    ax2.set(title="LOPO F1 (mean ± std)", ylabel="F1", ylim=[0.0, 1.08],
            xticks=x, xticklabels=models)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")
    for bar, v, s in zip(bars2, f1_means, f1_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, v + s + 0.012,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p = out_dir / "fig_model_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Comparison plot -> {p}")


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
    bst, qt, thr, evals_res, test_probs, test_labels, main_m = run_main_eval(X, y, pid, device)

    # Step 6: LOPO
    lopo_results, lopo_m, lopo_y, lopo_probs = run_lopo(X, y, pid, device)

    # Step 6b: Baseline LOPO
    baseline_m, baseline_pooled = run_baselines_lopo(X, y, pid)

    # Step 7: hardware
    hw_results = run_hw_test(bst, qt, thr, X)

    # Save model + QT + threshold
    import pickle
    bst.save_model(str(MODEL_DIR / "xgb_qt.json"))
    with open(MODEL_DIR / "qt_scaler.pkl", "wb") as f:
        pickle.dump(qt, f)
    with open(MODEL_DIR / "xgb_qt_thr.pkl", "wb") as f:
        pickle.dump((thr, evals_res), f)
    print(f"\n  Model saved -> {MODEL_DIR / 'xgb_qt.json'}")
    print(f"  QT scaler  -> {MODEL_DIR / 'qt_scaler.pkl'}")
    print(f"  Threshold  -> {MODEL_DIR / 'xgb_qt_thr.pkl'}")

    # Calibration metrics on test set
    brier, ece = brier_ece(test_labels, test_probs)
    print(f"\n  Calibration Metrics (CHARIS test set):")
    print(f"  Brier Score : {brier:.4f}  (0=perfect, 0.25=random)")
    print(f"  ECE         : {ece:.4f}  (0=perfectly calibrated)")
    main_m["brier_score"] = brier; main_m["ece"] = ece

    # Statistical tests
    stat_m = run_statistical_tests(lopo_y, lopo_probs, baseline_pooled, lopo_results, baseline_m)

    # Feature ablation
    ablation_m = run_feature_ablation(X, y, pid, device, lopo_m["auc_mean"])

    # Valsalva stats
    valsalva_m = run_hw_valsalva_stats(hw_results)

    # Plots
    save_plots(bst, evals_res, lopo_results, main_m, X, qt, OUT_DIR)
    save_comparison_plot(lopo_m, baseline_m, OUT_DIR)
    save_calibration_plot(test_probs, test_labels, OUT_DIR)

    # Save JSON
    meta = {
        "date": date.today().isoformat(),
        "alignment": "QuantileTransformer(N(0,1)) fitted on train split only",
        "main_split": main_m,
        "lopo": lopo_m,
        "baselines_lopo": baseline_m,
        "statistical_tests": stat_m,
        "feature_ablation": ablation_m,
        "valsalva_stats": valsalva_m,
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
    print(f"\n  Model Comparison (LOPO AUC):")
    print(f"  {'Model':<14} {'AUC':>8}  {'±':>6}  {'F1':>8}  {'±':>6}")
    print(f"  {'-'*48}")
    for name, bm in baseline_m.items():
        print(f"  {name:<14} {bm['auc_mean']:>8.4f}  {bm['auc_std']:>6.4f}  "
              f"{bm['f1_mean']:>8.4f}  {bm['f1_std']:>6.4f}")
    print(f"  {'XGBoost':<14} {lopo_m['auc_mean']:>8.4f}  {lopo_m['auc_std']:>6.4f}  "
          f"{lopo_m['f1_mean']:>8.4f}  {lopo_m['f1_std']:>6.4f}  [proposed]")
    print(f"\n  Feature Ablation (LOPO AUC when each feature is removed):")
    print(f"  {'Feature':<28} {'AUC':>8}  {'±':>6}  {'ΔAUC':>8}")
    print(f"  {'-'*54}")
    for feat, res in ablation_m.items():
        print(f"  {feat:<28} {res['auc_mean']:>8.4f}  {res['auc_std']:>6.4f}  "
              f"{res['delta_vs_full']:>+8.4f}")
    print(f"\n  Hardware Validation Summary:")
    print(f"  {'Subject':<12} {'Age':>5} {'Sex':>4}  {'Profile':<44} {'Flagged%':>9} {'Mean P':>8}")
    print(f"  {'-'*90}")
    for fname, r in hw_results.items():
        print(f"  {r['subject']:<12} {r['age']:>5} {r.get('gender','?'):>4}  "
              f"{r['profile']:<44} {r['pct_flagged']:>8.1f}%  {r['mean_prob']:>8.4f}")
    if hw_results:
        min_sub = min(hw_results.values(), key=lambda r: r["pct_flagged"])
        max_sub = max(hw_results.values(), key=lambda r: r["pct_flagged"])
        print(f"\n  Range: {min_sub['subject']} ({min_sub['pct_flagged']:.1f}% flagged) "
              f"→ {max_sub['subject']} ({max_sub['pct_flagged']:.1f}% flagged)")
    print(f"\n  Results -> {OUT_DIR}/qt_results.json")
    print(f"  Plot    -> {OUT_DIR}/qt_pipeline_results.png")
    print(SEP)


if __name__ == "__main__":
    main()
