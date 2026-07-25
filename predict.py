"""
predict.py  -  ICP-modulation inference
=======================================
Runs the honest, defensible model (the CHARIS clinical model applied ZERO-SHOT)
on a hardware recording and produces a WITHIN-SUBJECT ICP-modulation report -
the result this project actually stands behind.

It reports a *relative* ICP-elevation trend across the provocation manoeuvres.
It does NOT output a diagnosis or an absolute mmHg value.

Public API (used by the web demo, app.py)
-----------------------------------------
    from predict import analyse_csv, VERDICT_TRACKS
    result = analyse_csv("hw-tests/icp_100_20_M.csv")   # -> dict

CLI
---
    python predict.py hw-tests/icp_100_20_M.csv     # one subject, detailed report
    python predict.py --all                         # every hw-tests/*.csv, summary
"""
from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy.signal as sp_signal
import xgboost as xgb
from scipy.stats import spearmanr, wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths / constants ────────────────────────────────────────────────────────
MODEL_PATH = Path("models/xgb_qt.json")
QT_PATH = Path("models/qt_scaler.pkl")
THR_PATH = Path("results/qt_pipeline/qt_results.json")

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]
REQUIRED_COLS = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
FS, WIN, STEP = 50, 500, 250
SLOW_IDX = FEATURES.index("slow_wave_power")

SESSION_NAME = {1: "head-up 30°", 0: "supine", 2: "head-down 10°", 3: "Valsalva"}
ORDER = [1, 0, 2, 3]                       # ascending expected ICP
RHO_TRACK_THRESHOLD = 0.5                  # within-subject rho considered "tracking"

# Pre-computed DSP kernels
_NYQ = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0 / _NYQ, 2.5 / _NYQ], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1 / _NYQ, 0.5 / _NYQ], btype="band")
_FREQS = np.fft.rfftfreq(WIN, d=1.0 / FS)
_CARD_MASK = (_FREQS >= 0.7) & (_FREQS <= 2.5)


# ── Feature extraction (identical to the training pipeline) ──────────────────
def extract_window(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    """Return the 5-feature vector for one window, or None if unusable."""
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))

    card = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(card, 99) - np.percentile(card, 1))

    power = np.abs(np.fft.rfft(ir_dt)) ** 2
    card_freq = float(_FREQS[_CARD_MASK][np.argmax(power[_CARD_MASK])])

    resp = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(resp, 99) - np.percentile(resp, 1))

    energies = [float(np.sum(c ** 2)) for c in pywt.wavedec(disp_dt, "db4", level=5)]
    total = sum(energies) + 1e-12
    feat = np.array([card_amp, card_freq, resp_amp,
                     energies[0] / total, energies[2] / total], dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Model (loaded once, cached) ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    """Load the zero-shot CHARIS model, quantile scaler, and screening threshold."""
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    scaler = pickle.load(open(QT_PATH, "rb"))
    threshold = float(json.load(open(THR_PATH))["main_split"]["threshold"])
    return booster, scaler, threshold


# ── Core analysis ────────────────────────────────────────────────────────────
def _windows_from_df(df: pd.DataFrame):
    """Slice a recording into windows -> (feature matrix, session array)."""
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)
    feats, sess = [], []
    n_windows = (len(df) - WIN) // STEP + 1
    for w in range(max(n_windows, 0)):
        s, e = w * STEP, w * STEP + WIN
        chunk = df.iloc[s:e]
        f = extract_window(chunk["ir_raw"].to_numpy(np.float32),
                           chunk["disp_raw"].to_numpy(np.float32))
        if f is not None:
            feats.append(f)
            sess.append(int(chunk["session_label"].mode()[0]))
    return np.array(feats), np.array(sess)


def analyse_csv(csv_path: str | Path) -> dict:
    """Analyse one recording; return a structured within-subject ICP-modulation report."""
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = ", ".join(sorted(REQUIRED_COLS - set(df.columns)))
        return {"ok": False, "error": f"missing required columns: {missing}"}

    X, sess = _windows_from_df(df)
    if len(X) == 0:
        return {"ok": False, "error": "no valid signal windows in this recording"}

    booster, scaler, threshold = load_model()
    probs = booster.predict(xgb.DMatrix(scaler.transform(X).astype(np.float32),
                                        feature_names=FEATURES))

    present = [s for s in ORDER if (sess == s).sum() > 0]
    per_session = {s: float(probs[sess == s].mean()) for s in present}
    slow_wave = {s: float(X[sess == s, SLOW_IDX].mean()) for s in present}

    # within-subject dose-response along the ICP ladder
    rho = monotonic = None
    if len(present) >= 3:
        vec = [per_session[s] for s in present]
        rho = float(spearmanr(range(len(vec)), vec)[0])
        monotonic = all(vec[i] < vec[i + 1] for i in range(len(vec) - 1))

    # Valsalva vs baseline
    valsalva = baseline = valsalva_higher = None
    if 3 in per_session and any(s in per_session for s in (0, 1, 2)):
        valsalva = per_session[3]
        baseline = float(np.mean([per_session[s] for s in (1, 0, 2) if s in per_session]))
        valsalva_higher = bool(valsalva > baseline)

    tracks = (rho is not None and rho > RHO_TRACK_THRESHOLD) or (valsalva_higher is True)
    mean_score = float(probs.mean())
    return {
        "ok": True,
        "n_windows": int(len(X)),
        "mean_score": mean_score,
        "screening_prob": mean_score,                       # P(ICP elevated), 0..1
        "screening_threshold": float(threshold),
        "screening_flag": bool(mean_score >= threshold),    # True = FLAGGED (elevated)
        "sessions_present": present,
        "per_session": per_session,
        "slow_wave": slow_wave,
        "rho": rho,
        "monotonic": monotonic,
        "valsalva": valsalva,
        "baseline": baseline,
        "valsalva_higher": valsalva_higher,
        "tracks": bool(tracks),
        "verdict": VERDICT_TRACKS if tracks else VERDICT_WEAK,
    }


VERDICT_TRACKS = "ICP modulation tracked - output follows the physiological ladder."
VERDICT_WEAK = "Weak / absent modulation signal in this recording."
DISCLAIMER = ("Relative ICP-modulation proxy - validated by physiology, not calibrated "
              "to mmHg. Not a diagnosis and not a medical device.")


# ── CLI reporting ────────────────────────────────────────────────────────────
def print_report(name: str, r: dict) -> None:
    if not r["ok"]:
        print(f"  {name}: [skip] {r['error']}")
        return
    line = "=" * 62
    print(line)
    print(f"  ICP-MODULATION REPORT - {name}")
    print(line)
    print(f"  Windows analysed        : {r['n_windows']}")
    print(f"  Mean ICP-elevation score: {r['mean_score']:.3f}   (relative proxy, not mmHg)")
    print("\n  Within-subject dose-response (model output per session):")
    print(f"  {'session':<16}{'ICP score':>11}{'slow-wave':>12}")
    print(f"  {'-' * 39}")
    for s in ORDER:
        if s in r["per_session"]:
            print(f"  {SESSION_NAME[s]:<16}{r['per_session'][s]:>11.3f}{r['slow_wave'][s]:>12.3f}")
    if r["rho"] is not None:
        print(f"\n  Within-subject Spearman rho (output vs ICP ladder): {r['rho']:+.2f}")
        print(f"  Strictly monotonic with ICP ladder               : {'YES' if r['monotonic'] else 'no'}")
    if r["valsalva_higher"] is not None:
        d = r["valsalva"] - r["baseline"]
        print(f"  Valsalva > baseline: {'YES' if r['valsalva_higher'] else 'no'}  "
              f"({r['valsalva']:.3f} vs {r['baseline']:.3f}, delta {d:+.3f})")
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  ({DISCLAIMER})")


def _run_all() -> None:
    files = sorted(Path("hw-tests").glob("*.csv"))
    out, rows = {}, []
    for f in files:
        r = analyse_csv(f)
        out[f.name] = r
        if r["ok"]:
            rows.append(r)
    rhos = [r["rho"] for r in rows if r["rho"] is not None]
    mono = [r["monotonic"] for r in rows if r["monotonic"] is not None]
    vh = [r["valsalva_higher"] for r in rows if r["valsalva_higher"] is not None]
    print(f"Tested {len(rows)} subjects (all hw-tests):")
    print(f"  Mean within-subject rho          : {np.mean(rhos):+.3f}")
    print(f"  Subjects strictly monotonic      : {sum(mono)}/{len(mono)}  ({100 * sum(mono) / len(mono):.0f}%)")
    print(f"  Subjects Valsalva > baseline     : {sum(vh)}/{len(vh)}  ({100 * sum(vh) / len(vh):.0f}%)")
    diffs = [r["valsalva"] - r["baseline"] for r in rows if r["valsalva"] is not None]
    if len(diffs) >= 2:
        try:
            p = wilcoxon(diffs, alternative="greater")[1]
            print(f"  Valsalva>baseline (Wilcoxon p)   : {p:.2e}")
        except ValueError:
            pass
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/predict_all.json", "w"), indent=2)
    print("\n  Per-subject results -> results/predict_all.json")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    print(f"Model: CHARIS clinical (zero-shot) · threshold {load_model()[2]:.3f}\n")
    if args[0] == "--all":
        _run_all()
    else:
        path = Path(args[0])
        print_report(path.stem, analyse_csv(path))


if __name__ == "__main__":
    main()
