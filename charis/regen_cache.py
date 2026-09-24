"""Regenerate CHARIS feature cache with unit-free amplitude features."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FS, WIN, STEP = 50, 500, 250
THRESH, LABEL_FRAC = 20.0, 0.60
ICP_CH = {"ICP", "ICP1", "ICP2", "ICPC"}

CHARIS_DIR = Path("C:/Users/asus/Documents/GitHub/Pran/data/raw/charis")
CACHE      = Path("results/audit/cache")

_nyq = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS     = np.fft.rfftfreq(WIN, d=1.0/FS)
_FREQ_MASK = (_FREQS >= 0.7) & (_FREQS <= 2.5)

CACHE.mkdir(parents=True, exist_ok=True)
records = sorted([f.stem for f in CHARIS_DIR.glob("*.hea")])
Xa, ya, pa = [], [], []

for rec_name in records:
    pid = int("".join(filter(str.isdigit, rec_name)) or 0)
    if pid == 0: continue
    try: rec = wfdb.rdrecord(str(CHARIS_DIR / rec_name))
    except Exception as e: print(f"  SKIP {rec_name}: {e}"); continue
    sig = [s.upper() for s in rec.sig_name]
    ii  = next((i for i, s in enumerate(sig) if s in ICP_CH), None)
    if ii is None: print(f"  SKIP {rec_name}: no ICP channel"); continue
    icp = rec.p_signal[:, ii].astype(np.float64)
    if int(rec.fs) != FS:
        n = int(len(icp) * FS / int(rec.fs))
        icp = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(icp)), icp)
    bad = (icp < -5.0) | (icp > 50.0); icp[bad] = np.nan
    nan_m = np.isnan(icp)
    if nan_m.any():
        idx = np.where(~nan_m, np.arange(len(icp)), 0)
        np.maximum.accumulate(idx, out=idx); icp = icp[idx]
    icp = np.nan_to_num(icp, nan=0.0)

    card_sig = sp_signal.filtfilt(_B_CARD, _A_CARD, icp)
    resp_sig = sp_signal.filtfilt(_B_RESP, _A_RESP, icp)

    n_ok = n_skip = n0 = n1 = 0
    for w in range((len(icp) - WIN) // STEP + 1):
        s, e = w*STEP, w*STEP+WIN
        win  = icp[s:e]
        if win.std() < 0.02: n_skip += 1; continue
        label = 1 if (win >= THRESH).mean() > LABEL_FRAC else 0

        c_win    = card_sig[s:e]
        card_amp = float(np.percentile(c_win,99) - np.percentile(c_win,1))

        pwr = np.abs(np.fft.rfft(win))**2
        if not _FREQ_MASK.any(): n_skip += 1; continue
        card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])

        r_win    = resp_sig[s:e]
        resp_amp = float(np.percentile(r_win,99) - np.percentile(r_win,1))

        coeffs   = pywt.wavedec(win, "db4", level=5)
        energies = [float(np.sum(c**2)) for c in coeffs]
        total    = sum(energies) + 1e-12
        slow_pow    = energies[0] / total
        cardiac_pow = energies[2] / total

        feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow], dtype=np.float32)
        if not np.all(np.isfinite(feat)): n_skip += 1; continue

        Xa.append(feat); ya.append(label); pa.append(pid)
        n_ok += 1
        if label == 0: n0 += 1
        else: n1 += 1

    print(f"  {rec_name}: {n_ok:,} windows [norm={n0} abn={n1}] skip={n_skip}", flush=True)

X = np.array(Xa, dtype=np.float32)
y = np.array(ya, dtype=np.int64)
p = np.array(pa, dtype=np.int32)

np.save(CACHE / "X.npy", X)
np.save(CACHE / "y.npy", y)
np.save(CACHE / "pid.npy", p)

print(f"\nCache saved: {len(X):,} windows | {len(np.unique(p))} patients")
print(f"Normal: {(y==0).sum():,}  Abnormal: {(y==1).sum():,}")
print(f"cardiac_amplitude  mean={X[:,0].mean():.4f} std={X[:,0].std():.4f}")
print(f"respiratory_amp    mean={X[:,2].mean():.4f} std={X[:,2].std():.4f}")
