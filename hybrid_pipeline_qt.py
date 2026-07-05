"""
hybrid_pipeline_qt.py
=====================
Hybrid XGBoost ICP classification:
  CHARIS abnormals (subsampled to 5:1) + HW normals + HW valsalva abnormals

Key design
----------
- HW labels from session_label: session 3 (valsalva) = y=1, sessions 0-2 = y=0
- CHARIS abnormals subsampled so total_pos/total_neg <= MAX_RATIO (5.0)
- LOPO over HW patients only; CHARIS always in train
- QT fitted per-fold on train data (no leakage)
- Threshold via inner 90/10 val split of SMOTE'd train (never touches test fold)
- Valsalva stats computed from LOPO fold predictions (unbiased)
- Final model threshold = mean of per-fold thresholds

Run
---
    cd "C:\\Users\\asus\\Documents\\GitHub\\Pran"
    python hybrid_pipeline_qt.py
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
from scipy.stats import wilcoxon as _wilcoxon, norm as _norm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix, average_precision_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_X   = Path("results/audit/cache/X.npy")
CACHE_Y   = Path("results/audit/cache/y.npy")
HW_DIR    = Path("hw-tests")
OUT_DIR   = Path("results/hybrid_pipeline")
MODEL_DIR = Path("models/hybrid")
LOPO_DIR  = MODEL_DIR / "lopo"

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES  = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
             "slow_wave_power", "cardiac_power"]
N         = len(FEATURES)
SEED      = 42
FS, WIN, STEP = 50, 500, 250
MAX_RATIO = 5.0   # pos:neg cap before SMOTE

_nyq             = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS           = np.fft.rfftfreq(WIN, d=1.0 / FS)
_FREQ_MASK       = (_FREQS >= 0.7) & (_FREQS <= 2.5)


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_hw_window(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))

    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    pwr = np.abs(np.fft.rfft(ir_dt)) ** 2
    if not _FREQ_MASK.any():
        return None
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])

    r        = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    coeffs   = pywt.wavedec(disp_dt, "db4", level=5)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total
    cardiac_pow = energies[2] / total

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Hardware loader ────────────────────────────────────────────────────────────
def load_hw_labeled(hw_dir: Path):
    """
    Load all HW CSVs.  session 3 → y=1, sessions 0-2 → y=0.
    Skips CSVs without required columns.
    Returns (X, y, pid, sessions, names).
    """
    X_all, y_all, pid_all, sess_all = [], [], [], []
    names: list[str] = []
    pid_idx = 0

    for csv_path in sorted(hw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, comment="#")
        required = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
        if not required.issubset(df.columns):
            print(f"  [skip] {csv_path.name} — missing required columns")
            continue
        df = df[df["artifact_flag"] == 0].reset_index(drop=True)

        n_win   = (len(df) - WIN) // STEP + 1
        n_added = 0
        for w in range(n_win):
            s, e = w * STEP, w * STEP + WIN
            sl   = df.iloc[s:e]
            feat = extract_hw_window(
                sl["ir_raw"].values.astype(np.float32),
                sl["disp_raw"].values.astype(np.float32),
            )
            if feat is None:
                continue
            sess  = int(sl["session_label"].mode()[0])
            label = 1 if sess == 3 else 0
            X_all.append(feat); y_all.append(label)
            pid_all.append(pid_idx); sess_all.append(sess)
            n_added += 1

        n0 = sum(1 for y, p in zip(y_all, pid_all) if p == pid_idx and y == 0)
        n1 = sum(1 for y, p in zip(y_all, pid_all) if p == pid_idx and y == 1)
        print(f"  {csv_path.name:<32} windows={n_added:4d}  normal={n0:3d}  valsalva={n1:3d}")
        names.append(csv_path.name)
        pid_idx += 1

    if not X_all:
        print("  ERROR: no usable hardware CSVs.")
        sys.exit(1)

    return (np.array(X_all,    dtype=np.float32),
            np.array(y_all,    dtype=np.int32),
            np.array(pid_all,  dtype=np.int32),
            np.array(sess_all, dtype=np.int32),
            names)


# ── Utilities ──────────────────────────────────────────────────────────────────
def fit_qt(X_train: np.ndarray) -> QuantileTransformer:
    qt = QuantileTransformer(output_distribution="normal",
                             random_state=SEED, n_quantiles=min(1000, len(X_train)))
    return qt.fit(X_train)


def smote_global(X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """SMOTE to 1:1; adapts k to minority class size."""
    n0, n1   = int((y == 0).sum()), int((y == 1).sum())
    minority = min(n0, n1)
    k = max(1, min(5, minority - 1))
    try:
        X, y = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    except Exception:
        X, y = RandomOverSampler(random_state=seed).fit_resample(X, y)
    return X.astype(np.float32), np.asarray(y, dtype=np.int32)


def youden_threshold(y_true, probs) -> float:
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def xgb_params(device: str, seed: int = SEED) -> dict:
    return {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
            "eta": 0.05, "max_depth": 5, "min_child_weight": 3,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda": 1.0, "alpha": 0.1, "scale_pos_weight": 1.0,
            "seed": seed, "tree_method": "hist", "device": device, "verbosity": 0}


def get_device() -> str:
    try:
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            return "cpu"
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0},
                  xgb.DMatrix(np.zeros((4, N)), label=[0, 1, 0, 1]), num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def subsample_charis_abnormals(charis_X, charis_y, n_hw_normals: int, n_hw_abnormals: int,
                                max_ratio: float = MAX_RATIO, seed: int = SEED):
    """Return CHARIS abnormal windows capped at (cap + n_hw_abn) / n_hw_norm <= max_ratio."""
    abn_idx = np.where(charis_y == 1)[0]
    cap     = max(int(max_ratio * n_hw_normals) - n_hw_abnormals, 50)
    if len(abn_idx) > cap:
        rng     = np.random.default_rng(seed)
        abn_idx = rng.choice(abn_idx, size=cap, replace=False)
    return charis_X[abn_idx], np.ones(len(abn_idx), dtype=np.int32)


# ── DeLong test (subsampled) ───────────────────────────────────────────────────
def delong_test(y_true, pred_a, pred_b, max_per_class: int = 10_000):
    """DeLong 1988 — returns (auc_a, auc_b, z, p)."""
    y  = np.asarray(y_true, dtype=np.int32)
    pa = np.asarray(pred_a,  dtype=np.float64)
    pb = np.asarray(pred_b,  dtype=np.float64)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    if len(pos) > max_per_class: pos = rng.choice(pos, max_per_class, replace=False)
    if len(neg) > max_per_class: neg = rng.choice(neg, max_per_class, replace=False)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def _place(sp, sn):
        V10 = ((sp[:, None] > sn[None, :]).mean(1) +
               0.5 * (sp[:, None] == sn[None, :]).mean(1))
        V01 = ((sn[:, None] < sp[None, :]).mean(1) +
               0.5 * (sn[:, None] == sp[None, :]).mean(1))
        return V10, V01

    V10a, V01a = _place(pa[pos], pa[neg])
    V10b, V01b = _place(pb[pos], pb[neg])
    auc_a, auc_b = float(V10a.mean()), float(V10b.mean())
    S10 = np.cov(np.stack([V10a, V10b]), ddof=1)
    S01 = np.cov(np.stack([V01a, V01b]), ddof=1)
    var  = (S10[0,0]+S10[1,1]-2*S10[0,1])/n1 + (S01[0,0]+S01[1,1]-2*S01[0,1])/n0
    if var <= 0:
        return auc_a, auc_b, float("nan"), float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    p = float(2 * (1 - _norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p


# ── Hybrid LOPO ───────────────────────────────────────────────────────────────
def run_hybrid_lopo(hw_X, hw_y, hw_pid, charis_X, charis_y,
                    device, names, use_cache=True):
    """
    LOPO over HW patients.  CHARIS abnormals (subsampled) always in train.
    Threshold from inner 90/10 val split of SMOTE train — never touches test fold.
    Returns (results, summary, lopo_records, mean_fold_thr).
    lopo_records: list of {pid, y, probs} for each fold — used for valsalva stats.
    """
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    logo         = LeaveOneGroupOut()
    results      = []
    lopo_records = []   # unbiased per-patient predictions
    fold_thrs    = []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        test_pid  = int(hw_pid[te_idx[0]])
        fold_name = names[test_pid] if test_pid < len(names) else f"pid{test_pid}"
        cache_m   = LOPO_DIR / f"fold{fold:03d}_pid{test_pid:03d}.json"
        cache_t   = LOPO_DIR / f"fold{fold:03d}_pid{test_pid:03d}_thr.pkl"

        X_te, y_te   = hw_X[te_idx], hw_y[te_idx]
        X_hw_tr, y_hw_tr = hw_X[tr_idx], hw_y[tr_idx]

        n_hw_norm = int((y_hw_tr == 0).sum())
        n_hw_abn  = int((y_hw_tr == 1).sum())
        X_c, y_c  = subsample_charis_abnormals(charis_X, charis_y,
                                                n_hw_norm, n_hw_abn, seed=SEED + fold)
        X_tr = np.concatenate([X_hw_tr, X_c])
        y_tr = np.concatenate([y_hw_tr, y_c])

        # QT fit on train (deterministic → safe to refit when loading cache)
        qt = fit_qt(X_tr)

        if use_cache and cache_m.exists() and cache_t.exists():
            bst = xgb.Booster(); bst.load_model(str(cache_m))
            with open(cache_t, "rb") as f:
                thr = pickle.load(f)
            status = "(c)"
        else:
            X_tr_qt = qt.transform(X_tr).astype(np.float32)
            X_sm, y_sm = smote_global(X_tr_qt, y_tr, seed=SEED + fold)

            # Inner 90/10 split: threshold from val, never from test fold
            rng = np.random.default_rng(SEED + fold)
            idx = rng.permutation(len(y_sm)); cut = int(0.9 * len(idx))
            d_itr = xgb.DMatrix(X_sm[idx[:cut]],  label=y_sm[idx[:cut]],  feature_names=FEATURES)
            d_iva = xgb.DMatrix(X_sm[idx[cut:]], label=y_sm[idx[cut:]], feature_names=FEATURES)

            bst = xgb.train(xgb_params(device, SEED + fold), d_itr, num_boost_round=600,
                            evals=[(d_iva, "val")], early_stopping_rounds=40,
                            verbose_eval=False)
            p_iva = bst.predict(d_iva)
            thr   = youden_threshold(y_sm[idx[cut:]], p_iva) if len(np.unique(y_sm[idx[cut:]])) == 2 else 0.5

            bst.save_model(str(cache_m))
            with open(cache_t, "wb") as f:
                pickle.dump(thr, f)
            status = "(t)"

        X_te_qt = qt.transform(X_te).astype(np.float32)
        probs   = bst.predict(xgb.DMatrix(X_te_qt, feature_names=FEATURES))
        preds   = (probs >= thr).astype(int)

        auc  = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) == 2 else float("nan")
        f1   = float(f1_score(y_te, preds, zero_division=0))
        rec  = float(recall_score(y_te, preds, zero_division=0))
        prec = float(precision_score(y_te, preds, zero_division=0))
        spec = float(recall_score(1 - y_te, 1 - preds, zero_division=0))
        cm   = confusion_matrix(y_te, preds, labels=[0, 1]).tolist()

        print(f"  F{fold:3d} {status}  {fold_name:<32}  "
              f"AUC={auc:.4f}  F1={f1:.3f}  rec={rec:.3f}  spec={spec:.3f}  "
              f"te: {int((y_te==0).sum())}neg/{int((y_te==1).sum())}pos")

        results.append({"fold": fold, "patient": fold_name,
                        "auc": round(auc, 4), "f1": round(f1, 4),
                        "recall": round(rec, 4), "precision": round(prec, 4),
                        "specificity": round(spec, 4),
                        "n_test_neg": int((y_te==0).sum()), "n_test_pos": int((y_te==1).sum()),
                        "threshold": round(thr, 4), "confusion_matrix": cm})
        lopo_records.append({"pid": test_pid, "y": y_te.tolist(), "probs": probs.tolist()})
        fold_thrs.append(thr)

    valid_aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]
    valid_f1s  = [r["f1"]  for r in results]
    auc_mean   = float(np.mean(valid_aucs))
    auc_std    = float(np.std(valid_aucs))
    ci_lo      = float(np.percentile(valid_aucs, 2.5))
    ci_hi      = float(np.percentile(valid_aucs, 97.5))

    print(f"\n  LOPO AUC  : {auc_mean:.4f} ± {auc_std:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  LOPO F1   : {float(np.mean(valid_f1s)):.4f}")
    print(f"  Valid folds: {len(valid_aucs)}/{len(results)}")

    summary = {"auc_mean": round(auc_mean, 4), "auc_std": round(auc_std, 4),
               "auc_ci": [round(ci_lo, 4), round(ci_hi, 4)],
               "f1_mean": round(float(np.mean(valid_f1s)), 4),
               "n_folds": len(results), "n_valid": len(valid_aucs)}
    return results, summary, lopo_records, float(np.mean(fold_thrs))


# ── Valsalva statistics (from unbiased LOPO predictions) ──────────────────────
def run_valsalva_stats(lopo_records: list) -> dict:
    """
    Per-patient: mean P(ICP) for valsalva windows (y=1) vs normal windows (y=0).
    Uses LOPO predictions: each patient was held-out during its own fold.
    Paired Wilcoxon across patients.
    """
    val_means, norm_means = [], []
    for rec in lopo_records:
        y_arr    = np.array(rec["y"])
        p_arr    = np.array(rec["probs"])
        p_val    = p_arr[y_arr == 1]
        p_norm   = p_arr[y_arr == 0]
        if len(p_val) > 0 and len(p_norm) > 0:
            val_means.append(float(p_val.mean()))
            norm_means.append(float(p_norm.mean()))

    if len(val_means) < 4:
        print("  Valsalva stats: insufficient paired subjects (<4)")
        return {}

    val_arr  = np.array(val_means)
    norm_arr = np.array(norm_means)
    stat, pval = _wilcoxon(val_arr, norm_arr, alternative="greater")
    pct_higher = 100 * (val_arr > norm_arr).mean()
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  VALSALVA ANALYSIS (paired Wilcoxon on LOPO predictions)")
    print(SEP)
    print(f"  Paired subjects                 : {len(val_arr)}")
    print(f"  Subjects where valsalva > normal: {int((val_arr>norm_arr).sum())}/{len(val_arr)}  "
          f"({pct_higher:.0f}%)")
    print(f"  Mean P(ICP) valsalva            : {val_arr.mean():.4f}")
    print(f"  Mean P(ICP) normal              : {norm_arr.mean():.4f}")
    print(f"  Wilcoxon statistic              : {stat:.2f}   p={pval:.6f}  {sig}")
    print(SEP)

    return {"n_subjects": len(val_arr),
            "pct_valsalva_higher": round(pct_higher, 1),
            "mean_valsalva_prob": round(float(val_arr.mean()), 4),
            "mean_normal_prob":   round(float(norm_arr.mean()), 4),
            "wilcoxon_stat": round(float(stat), 4),
            "wilcoxon_p":    round(float(pval), 6)}


# ── Final model ────────────────────────────────────────────────────────────────
def train_final_model(hw_X, hw_y, charis_X, charis_y, device,
                      mean_lopo_thr: float):
    """
    Train on all HW + subsampled CHARIS. Threshold = mean LOPO threshold.
    Returns (bst, qt, thr).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "hybrid_xgb.json"
    qt_path    = MODEL_DIR / "hybrid_qt.pkl"
    thr_path   = MODEL_DIR / "hybrid_thr.pkl"

    n_hw_norm = int((hw_y == 0).sum())
    n_hw_abn  = int((hw_y == 1).sum())
    X_c, y_c  = subsample_charis_abnormals(charis_X, charis_y, n_hw_norm, n_hw_abn)

    X = np.concatenate([hw_X, X_c])
    y = np.concatenate([hw_y, y_c])
    pos, neg = int((y==1).sum()), int((y==0).sum())
    print(f"\n  Final model: {len(X):,} windows  pos={pos:,} neg={neg:,}  ratio={pos/neg:.2f}:1")

    qt        = fit_qt(X)
    X_qt      = qt.transform(X).astype(np.float32)
    X_sm, y_sm = smote_global(X_qt, y)
    print(f"  After SMOTE: {len(X_sm):,} windows  pos={int((y_sm==1).sum()):,}")

    d_tr = xgb.DMatrix(X_sm, label=y_sm, feature_names=FEATURES)
    bst  = xgb.train(xgb_params(device), d_tr, num_boost_round=600, verbose_eval=False)
    thr  = mean_lopo_thr   # unbiased: mean of per-fold Youden thresholds

    bst.save_model(str(model_path))
    with open(qt_path,  "wb") as f: pickle.dump(qt,  f)
    with open(thr_path, "wb") as f: pickle.dump(thr, f)
    print(f"  Threshold (mean LOPO Youden): {thr:.4f}")
    print(f"  Model -> {model_path}")
    return bst, qt, thr


# ── Feature importance ─────────────────────────────────────────────────────────
def run_feature_importance(bst: xgb.Booster) -> dict:
    """XGBoost gain importance from final model + permutation importance on LOPO records."""
    gain = bst.get_score(importance_type="gain")
    total = sum(gain.values()) + 1e-12
    norm_gain = {f: round(gain.get(f, 0.0) / total, 4) for f in FEATURES}

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  FEATURE IMPORTANCE (gain, normalised)")
    print(SEP)
    for f, g in sorted(norm_gain.items(), key=lambda x: -x[1]):
        bar = "█" * int(g * 40)
        print(f"  {f:<28}: {g:.4f}  {bar}")
    return norm_gain


# ── DeLong + Wilcoxon vs CHARIS-only XGBoost (if results exist) ───────────────
def run_comparison_test(lopo_records, lopo_results) -> dict:
    """
    Compare hybrid LOPO AUCs against CHARIS-only LOPO AUCs (from full_pipeline_qt results).
    Uses Wilcoxon on paired per-fold AUCs if the previous results JSON is available.
    """
    charis_json = Path("results/qt_pipeline/results.json")
    if not charis_json.exists():
        print("  Comparison: full_pipeline_qt results not found — skipping")
        return {}

    with open(charis_json) as f:
        prev = json.load(f)

    try:
        charis_aucs = [r["auc"] for r in prev["lopo"]["per_patient"]]
    except (KeyError, TypeError):
        print("  Comparison: could not parse CHARIS LOPO per-patient results")
        return {}

    hybrid_aucs = [r["auc"] for r in lopo_results if not np.isnan(r["auc"])]
    n = min(len(charis_aucs), len(hybrid_aucs))
    if n < 4:
        return {}

    a, b = np.array(hybrid_aucs[:n]), np.array(charis_aucs[:n])
    try:
        _, p_wx = _wilcoxon(a - b, alternative="greater")
    except Exception:
        p_wx = float("nan")

    delta = float(np.mean(a - b))
    sig = "***" if p_wx < 0.001 else ("**" if p_wx < 0.01 else ("*" if p_wx < 0.05 else "ns"))
    print(f"\n  Hybrid vs CHARIS-only (Wilcoxon, one-tailed):")
    print(f"  ΔAUC (mean)   : {delta:+.4f}  ({np.mean(a):.4f} vs {np.mean(b):.4f})")
    print(f"  Wilcoxon p    : {p_wx:.4f}  {sig}")

    return {"delta_auc": round(delta, 4), "hybrid_auc_mean": round(float(np.mean(a)), 4),
            "charis_auc_mean": round(float(np.mean(b)), 4), "wilcoxon_p": round(float(p_wx), 6)}


# ── Plots ──────────────────────────────────────────────────────────────────────
def save_plots(results, bst, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold AUC bar chart
    aucs   = [r["auc"] for r in results if not np.isnan(r["auc"])]
    labels = [r["patient"].replace(".csv","").replace("icp_","") for r in results
              if not np.isnan(r["auc"])]
    fig, ax = plt.subplots(figsize=(max(12, len(aucs)*0.45), 4))
    colors = ["#e74c3c" if a < 0.75 else "#f39c12" if a < 0.85 else "#2ecc71" for a in aucs]
    ax.bar(range(len(aucs)), aucs, color=colors, alpha=0.85)
    ax.axhline(np.mean(aucs), color="k", ls="--", lw=1.5, label=f"Mean {np.mean(aucs):.4f}")
    ax.set_xticks(range(len(aucs))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUC"); ax.set_title("Hybrid LOPO AUC per Patient")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_lopo_auc.png", dpi=150, bbox_inches="tight"); plt.close()

    # XGBoost feature importance
    gain  = bst.get_score(importance_type="gain")
    total = sum(gain.values()) + 1e-12
    feats = FEATURES
    vals  = [gain.get(f, 0.0) / total for f in feats]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(feats, vals, color="#3498db", alpha=0.8)
    ax.set_xlabel("Normalised Gain"); ax.set_title("Feature Importance (Hybrid XGBoost)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots -> {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 65
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  Hybrid XGBoost Pipeline  (fixed: leak-free threshold + unbiased valsalva)")
    print("  CHARIS abnormals (5:1 cap) + HW normals + HW valsalva")
    print(SEP)

    # [1] Load CHARIS
    for p in [CACHE_X, CACHE_Y]:
        if not p.exists():
            print(f"  ERROR: {p} not found. Run full_pipeline_qt.py first.")
            sys.exit(1)
    charis_X = np.load(CACHE_X)
    charis_y = np.load(CACHE_Y)
    n_cabn   = int((charis_y == 1).sum())
    print(f"\n[1] CHARIS: {len(charis_X):,} windows  abnormal={n_cabn:,}")

    # [2] Load hardware
    print(f"\n[2] Loading hardware data from {HW_DIR}/")
    hw_X, hw_y, hw_pid, hw_sess, names = load_hw_labeled(HW_DIR)
    n_pts  = len(np.unique(hw_pid))
    n_norm = int((hw_y == 0).sum()); n_abn = int((hw_y == 1).sum())
    X_c_p, _ = subsample_charis_abnormals(charis_X, charis_y, n_norm, n_abn)
    ratio     = (len(X_c_p) + n_abn) / max(n_norm, 1)
    print(f"\n  HW patients : {n_pts}")
    print(f"  HW windows  : {len(hw_X):,}  normal={n_norm:,}  valsalva={n_abn:,}")
    print(f"  CHARIS used : {len(X_c_p):,} / {n_cabn:,}  (cap at {MAX_RATIO}:1)")
    print(f"  Effective ratio before SMOTE: {ratio:.2f}:1")

    device = get_device()
    print(f"  Device: {device.upper()}")

    # [3] Hybrid LOPO
    print(f"\n[3] Hybrid LOPO ({n_pts} folds) — leak-free threshold ...")
    lopo_results, lopo_m, lopo_records, mean_thr = run_hybrid_lopo(
        hw_X, hw_y, hw_pid, charis_X, charis_y, device, names)

    # [4] Final model
    print(f"\n[4] Training final hybrid model ...")
    bst, qt, thr = train_final_model(hw_X, hw_y, charis_X, charis_y, device, mean_thr)

    # [5] Valsalva stats (unbiased: LOPO predictions)
    print(f"\n[5] Valsalva statistics (LOPO predictions) ...")
    val_stats = run_valsalva_stats(lopo_records)

    # [6] Feature importance
    print(f"\n[6] Feature importance ...")
    feat_imp = run_feature_importance(bst)

    # [7] Comparison vs CHARIS-only XGBoost
    print(f"\n[7] Comparison vs CHARIS-only XGBoost ...")
    comp = run_comparison_test(lopo_records, lopo_results)

    # [8] Plots
    save_plots(lopo_results, bst, OUT_DIR)

    # Summary
    print(f"\n{SEP}")
    print("  HYBRID XGBoost SUMMARY")
    print(SEP)
    print(f"  HW patients          : {n_pts}")
    print(f"  LOPO AUC             : {lopo_m['auc_mean']:.4f} ± {lopo_m['auc_std']:.4f}  "
          f"95%CI {lopo_m['auc_ci']}")
    print(f"  LOPO F1              : {lopo_m['f1_mean']:.4f}")
    if val_stats:
        print(f"  Valsalva elevated    : {val_stats['pct_valsalva_higher']:.0f}% subjects  "
              f"p={val_stats['wilcoxon_p']:.6f}")
    if comp:
        print(f"  vs CHARIS-only ΔAUC  : {comp['delta_auc']:+.4f}  p={comp['wilcoxon_p']:.4f}")
    print(SEP)

    # Save JSON
    out = {"date": date.today().isoformat(), "n_hw_patients": n_pts,
           "max_charis_ratio": MAX_RATIO, "charis_abn_used": int(len(X_c_p)),
           "hw_windows": {"total": int(len(hw_X)), "normal": int(n_norm), "valsalva": int(n_abn)},
           "lopo": lopo_m, "lopo_per_fold": lopo_results,
           "valsalva_stats": val_stats, "feature_importance": feat_imp,
           "comparison_vs_charis": comp}
    out_path = OUT_DIR / "hybrid_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
