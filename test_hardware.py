"""
test_hardware.py — Run XGBoost ICP classifier on hardware CSV files.

Feature extraction mirrors pipeline_clean.py (same bandpass, same wavelet).
Signals: ir_raw → cardiac features | disp_raw → respiratory + wavelet features
Z-score alignment: fitted on full CHARIS cache (results/audit/cache/X.npy).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from scipy import signal as sp_signal
from scipy.stats import gaussian_kde

# ── Constants (must match pipeline_clean.py) ──────────────────────────────────
FS   = 50
WIN  = 500   # 10 s
STEP = 250   # 50% overlap

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]

MODEL_PATH  = Path("C:/Users/asus/Documents/GitHub/Pran/models/xgb_clean.json")
CACHE_X     = Path("C:/Users/asus/Documents/GitHub/Pran/results/audit/cache/X.npy")

SESSION_NAMES = {0: "supine", 1: "head-up-30°", 2: "head-down-10°", 3: "valsalva+recovery"}

CSVS = {
    "icp_2":       Path("C:/Users/asus/Downloads/icp_2.csv"),
    "icp_1_27min": Path("C:/Users/asus/Downloads/icp_1_27min.csv"),
}

# ── Pre-compute filter coefficients ───────────────────────────────────────────
def _butter(lo, hi):
    nyq = FS / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return b, a

B_CARD, A_CARD = _butter(1.0, 2.5)
B_RESP, A_RESP = _butter(0.1, 0.5)
FREQS          = np.fft.rfftfreq(WIN, d=1.0 / FS)
FREQ_MASK      = (FREQS >= 0.7) & (FREQS <= 2.5)


# ── Feature extraction (hardware signals) ─────────────────────────────────────
def extract(ir_win: np.ndarray, disp_win: np.ndarray) -> np.ndarray | None:
    if ir_win.std() < 10 or disp_win.std() < 0.5:
        return None  # flat / contact-lost

    ir   = ir_win.astype(np.float64)
    disp = disp_win.astype(np.float64)

    # cardiac_amplitude: P99-P1 of IR in 1.0-2.5 Hz band (matches pipeline_clean)
    c = sp_signal.filtfilt(B_CARD, A_CARD, ir)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    # cardiac_frequency: dominant freq of IR in 0.7-2.5 Hz
    pwr = np.abs(np.fft.rfft(ir)) ** 2
    if not FREQ_MASK.any():
        return None
    card_freq = float(FREQS[FREQ_MASK][np.argmax(pwr[FREQ_MASK])])

    # respiratory_amplitude: P99-P1 of disp_raw in 0.1-0.5 Hz
    r = sp_signal.filtfilt(B_RESP, A_RESP, disp)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    # wavelet on disp_raw (db4 level 5 @ 50 Hz — matches pipeline_clean)
    coeffs   = pywt.wavedec(disp, "db4", level=5)
    energies = [float(np.sum(cc ** 2)) for cc in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total  # cA5
    cardiac_pow = energies[2] / total  # cD4

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Load hardware CSV and extract features ────────────────────────────────────
def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, comment="#")
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)

    feats, labels, sessions = [], [], []
    n_win = (len(df) - WIN) // STEP + 1

    for w in range(n_win):
        s, e = w * STEP, w * STEP + WIN
        sl = df.iloc[s:e]
        ir_win   = sl["ir_raw"].values.astype(np.float32)
        disp_win = sl["disp_raw"].values.astype(np.float32)
        session  = int(sl["session_label"].mode()[0])

        feat = extract(ir_win, disp_win)
        if feat is None:
            continue

        feats.append(feat)
        sessions.append(session)

    X   = np.array(feats,   dtype=np.float32)
    s   = np.array(sessions, dtype=np.int32)
    return X, s


# ── KL divergence check ───────────────────────────────────────────────────────
def kl_div(p, q, n=200):
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    grid = np.linspace(lo, hi, n)
    pk = gaussian_kde(p)(grid) + 1e-10
    qk = gaussian_kde(q)(grid) + 1e-10
    pk /= pk.sum(); qk /= qk.sum()
    return float(np.sum(pk * np.log(pk / qk)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 65

    # Load CHARIS cache for z-score stats
    print(f"\n{SEP}")
    print("  Loading CHARIS training stats for z-score alignment ...")
    charis_X  = np.load(CACHE_X)
    ch_mean   = charis_X.mean(axis=0)
    ch_std    = charis_X.std(axis=0)
    ch_std[ch_std < 1e-8] = 1.0

    print(f"  CHARIS feature means: {np.round(ch_mean, 4)}")
    print(f"  CHARIS feature stds : {np.round(ch_std, 4)}")

    # Load model
    print(f"\n  Loading XGBoost model ...")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))

    for csv_name, csv_path in CSVS.items():
        print(f"\n{SEP}")
        print(f"  FILE: {csv_path.name}")
        print(SEP)

        X_hw, sessions = load_csv(csv_path)
        print(f"  Extracted {len(X_hw):,} windows")
        print(f"\n  Raw hardware feature stats:")
        for i, f in enumerate(FEATURES):
            print(f"    {f:<28} mean={X_hw[:,i].mean():.4f}  std={X_hw[:,i].std():.4f}  "
                  f"range=[{X_hw[:,i].min():.4f}, {X_hw[:,i].max():.4f}]")

        # KL divergence before scaling
        print(f"\n  KL divergence (CHARIS vs hardware raw):")
        for i, f in enumerate(FEATURES):
            try:
                kl = kl_div(charis_X[:, i], X_hw[:, i])
                flag = "OK" if kl < 0.5 else "WARN" if kl < 1.5 else "HIGH"
                print(f"    {f:<28} KL={kl:.3f}  [{flag}]")
            except Exception:
                print(f"    {f:<28} KL=ERR")

        # Z-score using CHARIS stats
        X_scaled = (X_hw - ch_mean) / ch_std

        # KL divergence after scaling
        charis_scaled = (charis_X - ch_mean) / ch_std
        print(f"\n  KL divergence (CHARIS vs hardware after z-score):")
        for i, f in enumerate(FEATURES):
            try:
                kl = kl_div(charis_scaled[:, i], X_scaled[:, i])
                flag = "OK" if kl < 0.1 else "WARN" if kl < 0.3 else "HIGH"
                print(f"    {f:<28} KL={kl:.3f}  [{flag}]")
            except Exception:
                print(f"    {f:<28} KL=ERR")

        # Predict WITHOUT scaling (raw features — how model was trained)
        dm_raw    = xgb.DMatrix(X_hw,     feature_names=FEATURES)
        dm_scaled = xgb.DMatrix(X_scaled, feature_names=FEATURES)
        probs_raw    = bst.predict(dm_raw)
        probs_scaled = bst.predict(dm_scaled)

        print(f"\n  {'-'*60}")
        print(f"  PREDICTIONS (threshold = 0.2552 from training)")
        print(f"  {'-'*60}")

        for label, probs in [("RAW (no alignment)", probs_raw),
                              ("Z-SCORED (aligned)", probs_scaled)]:
            pred = (probs >= 0.2552).astype(int)
            pct_abn = 100 * pred.mean()
            print(f"\n  [{label}]")
            print(f"  Mean P(abnormal) : {probs.mean():.4f}")
            print(f"  % windows flagged: {pct_abn:.1f}%  "
                  f"(expect ~0% for healthy subject)")

            # Per session breakdown
            unique_sess = np.unique(sessions)
            if len(unique_sess) > 1:
                print(f"  Per session:")
                for sess in sorted(unique_sess):
                    m = sessions == sess
                    p_abn = 100 * pred[m].mean()
                    mean_p = probs[m].mean()
                    name = SESSION_NAMES.get(int(sess), f"sess_{sess}")
                    marker = " << Valsalva" if sess == 3 else ""
                    print(f"    [{name:<18}] flagged={p_abn:5.1f}%  "
                          f"mean_prob={mean_p:.4f}{marker}")

    print(f"\n{SEP}")
    print("  INTERPRETATION GUIDE")
    print(SEP)
    print("  Healthy subject  -> expect < 5% windows flagged as abnormal")
    print("  Valsalva session -> may show elevated P(abnormal) -- this is expected")
    print("  HIGH KL on amplitude features -> domain gap (different signal units)")
    print("  cardiac_frequency + power features -> unit-free, KL should be low")
    print(SEP)


if __name__ == "__main__":
    main()
