"""
hw_test_pipeline.py  --  Hardware CSV -> XGBoost ICP Prediction Pipeline
=========================================================================
Fixes applied vs test_hardware.py:
  1. Detrend ir_raw and disp_raw before all feature extraction (removes DC offset)
  2. QuantileTransformer alignment fitted on CHARIS cache (better than z-score)
  3. Clean per-session breakdown with verdict

Usage:
    python hw_test_pipeline.py
    python hw_test_pipeline.py --csv path1.csv path2.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from scipy import signal as sp_signal
from sklearn.preprocessing import QuantileTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CSVS = [
    Path("C:/Users/asus/Downloads/icp_1_27min.csv"),
    Path("C:/Users/asus/Downloads/icp_2.csv"),
]
MODEL_PATH = Path("C:/Users/asus/Documents/GitHub/Pran/models/xgb_clean.json")
CACHE_X    = Path("C:/Users/asus/Documents/GitHub/Pran/results/audit/cache/X.npy")
OUT_DIR    = Path("C:/Users/asus/Documents/GitHub/Pran/results/hardware")

# ── Constants ─────────────────────────────────────────────────────────────────
FS        = 50
WIN       = 500    # 10 s @ 50 Hz
STEP      = 250    # 50% overlap
THRESHOLD = 0.2552 # Youden's J from training

FEATURES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
]

SESSION_NAMES = {
    0: "supine",
    1: "head-up-30deg",
    2: "head-down-10deg",
    3: "valsalva+recovery",
}

# ── Filter coefficients (pre-computed once) ───────────────────────────────────
_nyq            = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS           = np.fft.rfftfreq(WIN, d=1.0/FS)
_FREQ_MASK       = (_FREQS >= 0.7) & (_FREQS <= 2.5)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_window(ir_raw: np.ndarray, disp_raw: np.ndarray) -> np.ndarray | None:
    """
    Extract 5 features from one 10-second window.

    FIX: both signals are detrended (mean-subtracted) before any computation.
    Without detrending, ir_raw ~47000 DC and disp_raw ~180 DC swamp the wavelet
    decomposition (slow_wave_power -> 0.9999, cardiac_power -> 0.0000).
    Detrending exposes the actual pulsatile content the model was trained on.
    """
    ir   = ir_raw.astype(np.float64)
    disp = disp_raw.astype(np.float64)

    # Reject flat / contact-lost windows
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None

    # Detrend: remove DC offset and linear drift
    ir_dt   = sp_signal.detrend(ir)
    disp_dt = sp_signal.detrend(disp)

    # cardiac_amplitude: P99-P1 of ir_dt in cardiac band (matches pipeline_clean)
    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    # cardiac_frequency: dominant freq in 0.7-2.5 Hz band of ir_dt
    pwr = np.abs(np.fft.rfft(ir_dt)) ** 2
    if not _FREQ_MASK.any():
        return None
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])

    # respiratory_amplitude: P99-P1 of disp_dt in 0.1-0.5 Hz band
    r        = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    # Wavelet on detrended displacement (db4, level 5 @ 50 Hz)
    # cA5: 0-0.78 Hz (slow wave)    cD4: 1.56-3.12 Hz (cardiac band)
    coeffs   = pywt.wavedec(disp_dt, "db4", level=5)
    energies = [float(np.sum(cc**2)) for cc in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total   # cA5
    cardiac_pow = energies[2] / total   # cD4

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, comment="#")

    required = {"ir_raw", "disp_raw", "artifact_flag"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  ERROR: {path.name} missing columns: {missing}")
        sys.exit(1)

    df = df[df["artifact_flag"] == 0].reset_index(drop=True)
    has_session = "session_label" in df.columns

    feats, sessions = [], []
    n_win  = (len(df) - WIN) // STEP + 1
    n_skip = 0

    for w in range(n_win):
        s, e = w * STEP, w * STEP + WIN
        sl   = df.iloc[s:e]
        feat = extract_window(
            sl["ir_raw"].values.astype(np.float32),
            sl["disp_raw"].values.astype(np.float32),
        )
        if feat is None:
            n_skip += 1
            continue
        feats.append(feat)
        sess = int(sl["session_label"].mode()[0]) if has_session else 0
        sessions.append(sess)

    X = np.array(feats,   dtype=np.float32)
    s = np.array(sessions, dtype=np.int32)
    return X, s, n_skip


# ── KL divergence ─────────────────────────────────────────────────────────────
def kl_div(p: np.ndarray, q: np.ndarray, n: int = 200) -> float:
    from scipy.stats import gaussian_kde
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    if lo == hi:
        return 0.0
    grid = np.linspace(lo, hi, n)
    try:
        pk = gaussian_kde(p)(grid) + 1e-10
        qk = gaussian_kde(q)(grid) + 1e-10
    except Exception:
        return float("nan")
    pk /= pk.sum(); qk /= qk.sum()
    return float(np.sum(pk * np.log(pk / qk)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main(csv_paths: list[Path]):
    SEP  = "=" * 65
    SEP2 = "-" * 65

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── 1. Load CHARIS cache and fit QuantileTransformer ──────────────────────
    print(f"\n{SEP}")
    print("  [1/4] Loading CHARIS cache + fitting QuantileTransformer ...")
    print(SEP)
    if not CACHE_X.exists():
        print(f"  ERROR: CHARIS cache not found at {CACHE_X}")
        print("  Run audit_plots.py first to generate the cache.")
        sys.exit(1)

    charis_X = np.load(CACHE_X)
    print(f"  CHARIS windows: {len(charis_X):,}  |  features: {charis_X.shape[1]}")

    qt = QuantileTransformer(output_distribution="normal", random_state=42, n_quantiles=1000)
    qt.fit(charis_X)
    charis_aligned = qt.transform(charis_X)
    print("  QuantileTransformer fitted on CHARIS (maps each feature to N(0,1))")

    # Z-score stats (for comparison)
    ch_mean = charis_X.mean(axis=0)
    ch_std  = charis_X.std(axis=0)
    ch_std[ch_std < 1e-8] = 1.0

    # ── 2. Load XGBoost model ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [2/4] Loading XGBoost model ...")
    print(SEP)
    if not MODEL_PATH.exists():
        print(f"  ERROR: model not found at {MODEL_PATH}")
        print("  Run pipeline_clean.py first.")
        sys.exit(1)

    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    print(f"  Model loaded: {MODEL_PATH.name}")
    print(f"  Threshold: {THRESHOLD:.4f} (Youden's J on CHARIS val set)")

    # ── 3. Process each CSV ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [3/4] Extracting features from hardware CSVs ...")
    print(SEP)

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        print(f"\n  {csv_path.name}")
        print(f"  {SEP2}")

        X_hw, sessions, n_skip = load_csv(csv_path)
        print(f"  Windows: {len(X_hw):,} extracted, {n_skip} skipped (flat/contact-lost)")

        # Feature stats after fix
        print(f"\n  Feature stats (after detrend):")
        print(f"  {'Feature':<28} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*68}")
        for i, f in enumerate(FEATURES):
            print(f"  {f:<28} {X_hw[:,i].mean():>10.4f} {X_hw[:,i].std():>10.4f} "
                  f"{X_hw[:,i].min():>10.4f} {X_hw[:,i].max():>10.4f}")

        # KL divergence
        print(f"\n  Feature alignment (KL divergence after QuantileTransform):")
        X_aligned = qt.transform(X_hw)
        kl_vals = []
        for i, f in enumerate(FEATURES):
            kl = kl_div(charis_aligned[:, i], X_aligned[:, i])
            kl_vals.append(kl)
            flag = "OK  " if kl < 0.1 else "WARN" if kl < 0.3 else "HIGH"
            bar  = "#" * min(int(kl * 10), 30)
            print(f"  [{flag}] {f:<28} KL={kl:.3f}  {bar}")

        # Predict
        dm = xgb.DMatrix(X_aligned, feature_names=FEATURES)
        probs = bst.predict(dm)
        preds = (probs >= THRESHOLD).astype(int)

        # Results
        print(f"\n{SEP}")
        print(f"  [RESULTS]  {csv_path.name}")
        print(SEP)
        print(f"  Total windows      : {len(probs):,}")
        print(f"  Mean P(abnormal)   : {probs.mean():.4f}")
        print(f"  Flagged as abnormal: {preds.sum():,} / {len(preds):,} "
              f"({100*preds.mean():.1f}%)")

        unique_sess = np.unique(sessions)
        sess_results = {}

        if len(unique_sess) > 1:
            print(f"\n  Per-session breakdown:")
            print(f"  {'Session':<22} {'Windows':>8} {'Flagged':>8} {'Flag%':>7} "
                  f"{'MeanP':>8}  Verdict")
            print(f"  {'-'*65}")
            for sess in sorted(unique_sess):
                m      = sessions == sess
                n_s    = m.sum()
                n_flag = preds[m].sum()
                pct    = 100 * preds[m].mean()
                mean_p = probs[m].mean()
                sname  = SESSION_NAMES.get(int(sess), f"session_{sess}")

                if sess == 3:
                    # Valsalva — expect elevated ICP, higher flag rate is GOOD
                    verdict = "expected elevated" if pct > 20 else "low (may need more Valsalva effort)"
                else:
                    verdict = "PASS" if pct < 10 else "borderline" if pct < 25 else "high"

                tag = " << Valsalva" if sess == 3 else ""
                print(f"  {sname:<22} {n_s:>8,} {n_flag:>8,} {pct:>6.1f}%  "
                      f"{mean_p:>8.4f}  {verdict}{tag}")

                sess_results[sname] = {
                    "n_windows": int(n_s),
                    "flagged": int(n_flag),
                    "pct_flagged": round(float(pct), 2),
                    "mean_prob": round(float(mean_p), 4),
                }

        # Overall verdict
        pct_total = 100 * preds.mean()
        print(f"\n  {'='*65}")
        print(f"  OVERALL VERDICT: ", end="")
        if pct_total < 5:
            verdict_str = "EXCELLENT -- model correctly classifies as normal (<5% flagged)"
        elif pct_total < 15:
            verdict_str = "GOOD -- minor false positives, within expected domain gap range"
        elif pct_total < 30:
            verdict_str = "BORDERLINE -- domain gap visible, feature alignment needed"
        else:
            verdict_str = "HIGH FALSE POSITIVE RATE -- check sensor contact + feature alignment"
        print(verdict_str)
        print(f"  {'='*65}")

        all_results[csv_path.name] = {
            "n_windows": len(probs),
            "mean_prob": round(float(probs.mean()), 4),
            "pct_flagged": round(float(pct_total), 2),
            "kl_divergence": {f: round(kl_vals[i], 4) for i, f in enumerate(FEATURES)},
            "per_session": sess_results,
            "verdict": verdict_str,
        }

    # ── 4. Save results ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [4/4] Saving results ...")
    out_json = OUT_DIR / "hw_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"  Results saved -> {out_json}")
    print(SEP)

    # Summary table
    print(f"\n  SUMMARY")
    print(f"  {'File':<30} {'%Flagged':>10}  {'MeanP':>8}  Verdict")
    print(f"  {'-'*70}")
    for fname, r in all_results.items():
        print(f"  {fname:<30} {r['pct_flagged']:>9.1f}%  {r['mean_prob']:>8.4f}  "
              f"{r['verdict'].split('--')[0].strip()}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware CSV -> XGBoost ICP test pipeline")
    parser.add_argument(
        "--csv", nargs="+", type=Path,
        default=DEFAULT_CSVS,
        help="One or more hardware CSV paths",
    )
    args = parser.parse_args()
    main(args.csv)
