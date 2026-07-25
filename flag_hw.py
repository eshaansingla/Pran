"""
flag_hw.py
==========
Apply the CHARIS-trained XGBoost model to every hardware recording and flag
each subject as overall-abnormal (mean P(ICP) > CHARIS threshold) or normal.

Inputs
------
  models/xgb_qt.json                  CHARIS XGBoost model
  models/qt_scaler.pkl                CHARIS QuantileTransformer
  results/qt_pipeline/qt_results.json CHARIS decision threshold (main_split)
  hw-tests/*.csv                      hardware recordings

Output
------
  results/hw_charis_flags.json        per-subject flags + per-session breakdown

Run
---
  python flag_hw.py
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.signal as sp_signal
import pywt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HW_DIR   = Path("hw-tests")
MODEL    = Path("models/xgb_qt.json")
QT_PKL   = Path("models/qt_scaler.pkl")
RESULTS  = Path("results/qt_pipeline/qt_results.json")
OUT      = Path("results/hw_charis_flags.json")
REC_OUT  = Path("results/hw_charis_records.pkl")  # per-window probs+sessions for dose-response
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]

FS, WIN, STEP = 50, 500, 250
_nyq = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS     = np.fft.rfftfreq(WIN, d=1.0/FS)
_FREQ_MASK = (_FREQS >= 0.7) & (_FREQS <= 2.5)


def extract_window(ir, disp):
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))
    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))
    pwr      = np.abs(np.fft.rfft(ir_dt)) ** 2
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])
    r        = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))
    coeffs   = pywt.wavedec(disp_dt, "db4", level=5)
    energies = [float(np.sum(cc ** 2)) for cc in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total
    cardiac_pow = energies[2] / total
    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow], dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def main():
    bst = xgb.Booster(); bst.load_model(str(MODEL))
    qt  = pickle.load(open(QT_PKL, "rb"))
    thr = float(json.load(open(RESULTS))["main_split"]["threshold"])
    print(f"CHARIS model loaded  threshold={thr:.4f}\n")
    print(f"{'Subject':<24} {'Windows':>7} {'Mean P':>8} {'Frac>thr':>9} {'Flagged':>8}")
    print("-" * 62)

    flags = {}
    records = []   # per-window probs + sessions per subject (pure-CHARIS model on HW)
    for csv_path in sorted(HW_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, comment="#", low_memory=False)
        required = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
        if not required.issubset(df.columns):
            print(f"{csv_path.name:<24} [skip] missing columns")
            continue
        df = df[df["artifact_flag"] == 0].reset_index(drop=True)
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
        X     = np.array(feats)
        sess  = np.array(sessions)
        probs = bst.predict(xgb.DMatrix(qt.transform(X).astype(np.float32),
                                        feature_names=FEATURES))
        mean_p     = float(probs.mean())
        frac_abn   = float((probs > thr).mean())
        is_flagged = mean_p > thr
        sess_info  = {int(s): round(float(probs[sess == s].mean()), 4)
                      for s in sorted(np.unique(sess))}
        flags[csv_path.name] = {"flagged": bool(is_flagged),
                                "mean_prob": round(mean_p, 4),
                                "frac_above_thr": round(frac_abn, 4),
                                "n_windows": len(probs),
                                "per_session": sess_info}
        records.append({"name": csv_path.name,
                        "true_label": int(is_flagged),
                        "probs": probs.tolist(),
                        "sessions": sess.tolist(),
                        "mean_prob": round(mean_p, 4)})
        mark = "  <<< ABNORMAL" if is_flagged else ""
        print(f"{csv_path.name:<24} {len(probs):>7} {mean_p:>8.4f} "
              f"{frac_abn:>9.3f} {str(is_flagged):>8}{mark}")

    n_flagged = sum(1 for v in flags.values() if v["flagged"])
    print(f"\n{'='*62}")
    print(f"Flagged ABNORMAL : {n_flagged} / {len(flags)}")
    print(f"Flagged NORMAL   : {len(flags)-n_flagged} / {len(flags)}")
    print(f"CHARIS threshold : {thr:.4f}\n{'='*62}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"charis_threshold": thr, "n_flagged": n_flagged,
               "n_total": len(flags), "patients": flags},
              open(OUT, "w"), indent=2)
    print(f"Saved -> {OUT}")
    pickle.dump(records, open(REC_OUT, "wb"))
    print(f"Saved -> {REC_OUT}  ({len(records)} subject records)")


if __name__ == "__main__":
    main()
