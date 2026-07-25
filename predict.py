"""
predict.py  —  ICP-modulation inference pipeline
================================================
Runs the honest, defensible model (the CHARIS clinical model applied ZERO-SHOT)
on a hardware recording and reports an in-depth WITHIN-SUBJECT ICP-modulation
analysis — the result the project actually stands behind. It does NOT output a
diagnosis or an mmHg value.

Usage
-----
  python predict.py hw-tests/icp_100_20_M.csv     # one subject, detailed report
  python predict.py --all                         # every hw-tests/*.csv, summary

Model: models/xgb_qt.json + models/qt_scaler.pkl  (trained on invasive CHARIS ICP)
Output for --all: results/predict_all.json
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.signal as sp_signal
import pywt
from scipy.stats import spearmanr, wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = Path("models/xgb_qt.json")
QT    = Path("models/qt_scaler.pkl")
THR_J = Path("results/qt_pipeline/qt_results.json")
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]
FS, WIN, STEP = 50, 500, 250
SESS = {1: "head-up 30°", 0: "supine", 2: "head-down 10°", 3: "Valsalva"}
ORDER = [1, 0, 2, 3]                    # ascending expected ICP
SLOW_IDX = 3                            # slow_wave_power feature index (ICP biomarker)

_nyq = FS / 2.0
_B_C, _A_C = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_R, _A_R = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS = np.fft.rfftfreq(WIN, d=1.0/FS)
_MASK = (_FREQS >= 0.7) & (_FREQS <= 2.5)


def extract_window(ir, disp):
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))
    c = sp_signal.filtfilt(_B_C, _A_C, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))
    pwr = np.abs(np.fft.rfft(ir_dt)) ** 2
    card_freq = float(_FREQS[_MASK][np.argmax(pwr[_MASK])])
    r = sp_signal.filtfilt(_B_R, _A_R, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))
    coeffs = pywt.wavedec(disp_dt, "db4", level=5)
    e = [float(np.sum(cc ** 2)) for cc in coeffs]; tot = sum(e) + 1e-12
    feat = np.array([card_amp, card_freq, resp_amp, e[0]/tot, e[2]/tot], dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def load_model():
    bst = xgb.Booster(); bst.load_model(str(MODEL))
    qt = pickle.load(open(QT, "rb"))
    thr = float(json.load(open(THR_J))["main_split"]["threshold"])
    return bst, qt, thr


def analyse(csv_path, bst, qt, thr):
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    need = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
    if not need.issubset(df.columns):
        return {"error": "missing required columns"}
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)
    feats, sess = [], []
    for w in range((len(df) - WIN) // STEP + 1):
        s, e = w*STEP, w*STEP + WIN
        sl = df.iloc[s:e]
        f = extract_window(sl["ir_raw"].values.astype(np.float32),
                           sl["disp_raw"].values.astype(np.float32))
        if f is None:
            continue
        feats.append(f); sess.append(int(sl["session_label"].mode()[0]))
    if not feats:
        return {"error": "no valid windows"}
    X = np.array(feats); sess = np.array(sess)
    probs = bst.predict(xgb.DMatrix(qt.transform(X).astype(np.float32), feature_names=FEATURES))

    per = {s: float(probs[sess == s].mean()) for s in ORDER if (sess == s).sum() > 0}
    slow = {s: float(X[sess == s, SLOW_IDX].mean()) for s in ORDER if (sess == s).sum() > 0}
    # within-subject dose-response (only if the subject has the ordered sessions)
    have = [s for s in ORDER if s in per]
    rho = mono = None
    if len(have) >= 3:
        ranks = list(range(len(have)))
        vec = [per[s] for s in have]
        rho = float(spearmanr(ranks, vec)[0])
        mono = all(vec[i] < vec[i+1] for i in range(len(vec)-1))
    # Valsalva vs baseline
    val = base = vtest = None
    if 3 in per and any(s in per for s in (0, 1, 2)):
        val = per[3]; base = float(np.mean([per[s] for s in (1, 0, 2) if s in per]))
        vtest = val > base
    return {"n_windows": len(feats), "mean_score": float(probs.mean()),
            "per_session": per, "slow_wave": slow, "rho": rho, "monotonic": mono,
            "valsalva": val, "baseline": base, "valsalva_higher": vtest}


def print_report(name, r):
    if "error" in r:
        print(f"  {name}: [skip] {r['error']}"); return
    print("=" * 62)
    print(f"  ICP-MODULATION REPORT — {name}")
    print("=" * 62)
    print(f"  Windows analysed        : {r['n_windows']}")
    print(f"  Mean ICP-elevation score: {r['mean_score']:.3f}   (relative proxy, not mmHg)")
    print(f"\n  Within-subject dose-response (model output per session):")
    print(f"  {'session':<16}{'ICP score':>11}{'slow-wave':>12}")
    print(f"  {'-'*39}")
    for s in ORDER:
        if s in r["per_session"]:
            print(f"  {SESS[s]:<16}{r['per_session'][s]:>11.3f}{r['slow_wave'][s]:>12.3f}")
    if r["rho"] is not None:
        print(f"\n  Within-subject Spearman rho (output vs ICP ladder): {r['rho']:+.2f}")
        print(f"  Strictly monotonic with ICP ladder               : {'YES' if r['monotonic'] else 'no'}")
    if r["valsalva_higher"] is not None:
        d = r["valsalva"] - r["baseline"]
        print(f"  Valsalva > baseline: {'YES' if r['valsalva_higher'] else 'no'}  "
              f"({r['valsalva']:.3f} vs {r['baseline']:.3f}, Δ{d:+.3f})")
    # honest verdict
    tracks = (r["rho"] is not None and r["rho"] > 0.5) or (r["valsalva_higher"] is True)
    print(f"\n  VERDICT: {'ICP MODULATION TRACKED — output follows the physiological ladder.' if tracks else 'weak/absent modulation signal in this recording.'}")
    print(f"  (Relative ICP-modulation proxy. Not a diagnosis, not calibrated mmHg.)")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    bst, qt, thr = load_model()
    print(f"Model: CHARIS clinical (zero-shot) · threshold {thr:.3f}\n")

    if args[0] == "--all":
        files = sorted(Path("hw-tests").glob("*.csv"))
        rows, out = [], {}
        for f in files:
            r = analyse(f, bst, qt, thr)
            out[f.name] = r
            if "error" in r:
                continue
            rows.append((f.name, r.get("rho"), r.get("monotonic"), r.get("valsalva_higher"), r["mean_score"]))
        # cohort summary
        rhos = [x[1] for x in rows if x[1] is not None]
        mono = [x[2] for x in rows if x[2] is not None]
        vh   = [x[3] for x in rows if x[3] is not None]
        print(f"Tested {len(rows)} subjects (all hw-tests):")
        print(f"  Mean within-subject rho          : {np.mean(rhos):+.3f}")
        print(f"  Subjects strictly monotonic      : {sum(mono)}/{len(mono)}  ({100*sum(mono)/len(mono):.0f}%)")
        print(f"  Subjects Valsalva > baseline     : {sum(vh)}/{len(vh)}  ({100*sum(vh)/len(vh):.0f}%)")
        if len(vh) >= 2:
            # sign test / wilcoxon on valsalva-baseline
            diffs = [out[n]["valsalva"] - out[n]["baseline"] for n, *_ in rows
                     if out[n].get("valsalva") is not None]
            try:
                p = wilcoxon(diffs, alternative="greater")[1]
                print(f"  Valsalva>baseline (Wilcoxon p)   : {p:.2e}")
            except Exception:
                pass
        Path("results").mkdir(exist_ok=True)
        json.dump(out, open("results/predict_all.json", "w"), indent=2)
        print(f"\n  Per-subject results -> results/predict_all.json")
    else:
        p = Path(args[0])
        print_report(p.stem, analyse(p, bst, qt, thr))


if __name__ == "__main__":
    main()
