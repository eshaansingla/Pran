"""
extract_charis_v1.py
====================
Extract CHARIS ICP features at 125 Hz (v1 pipeline format).

Reads local .dat files from data/raw/charis/ and produces:
  data/processed/features.npy        (N_charis, 6)  float32
  data/processed/labels.npy          (N_charis,)    int64
  data/processed/patient_ids.npy     (N_charis,)    int32

Feature order (must match v1 config.py and train_binary.py KEEP=[0..5]):
  0  cardiac_amplitude      peak-to-peak ICP in 0.7-2.5 Hz band
  1  cardiac_frequency      dominant freq in cardiac band (Hz)
  2  respiratory_amplitude  peak-to-peak ICP in 0.1-0.5 Hz band
  3  slow_wave_power        wavelet energy fraction (cA5+cD5, db4 level-5, 125 Hz)
  4  cardiac_power          wavelet energy fraction (cD4, db4 level-5, 125 Hz)
  5  mean_arterial_pressure real MAP from ABP waveform (not constant 90)

ICP >= 15 mmHg → label 1 (Abnormal), else 0 (Normal).
CHARIS patient IDs: 1 to 13.

Usage:
    python extract_charis_v1.py
    python extract_charis_v1.py --charis_dir data/raw/charis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal

# ── Constants (must match v1 config.py) ──────────────────────────────────────
TARGET_FS      = 125
WINDOW_SAMPLES = 1250          # 10 s × 125 Hz
WAVELET        = "db4"
WAVELET_LEVEL  = 5
ICP_THRESHOLD  = 15.0          # mmHg

ICP_CHANNELS   = {"ICP", "ICP1", "ICP2"}
ABP_CHANNELS   = {"ABP", "ART", "ABPI", "ABP1"}
ICP_VALID_RANGE = (0.0, 60.0)
ABP_VALID_RANGE = (40.0, 200.0)
MAX_MISSING_FRAC = 0.20

# Feature clip ranges (from v1 config.py FEATURE_RANGES)
CLIP_RANGES = [
    (5.0,  120.0),   # cardiac_amplitude
    (0.7,  2.5),     # cardiac_frequency
    (1.0,  50.0),    # respiratory_amplitude
    (0.30, 1.0),     # slow_wave_power
    (0.0,  0.40),    # cardiac_power
    (40.0, 200.0),   # mean_arterial_pressure
]


# ── Signal helpers ────────────────────────────────────────────────────────────

def _resample(sig: np.ndarray, orig_fs: int) -> np.ndarray:
    if orig_fs == TARGET_FS:
        return sig.astype(np.float32)
    n_new = int(len(sig) * TARGET_FS / orig_fs)
    return np.interp(
        np.linspace(0, 1, n_new),
        np.linspace(0, 1, len(sig)),
        sig,
    ).astype(np.float32)


def _bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    nyq = TARGET_FS / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig).astype(np.float32)


def _is_valid(icp_win: np.ndarray) -> bool:
    if np.isnan(icp_win).mean() > MAX_MISSING_FRAC:
        return False
    clean = icp_win[~np.isnan(icp_win)]
    if len(clean) < WINDOW_SAMPLES * 0.7:
        return False
    if clean.std() < 0.05:
        return False
    if not (ICP_VALID_RANGE[0] <= clean.mean() <= ICP_VALID_RANGE[1]):
        return False
    return True


def _extract_features(icp_win: np.ndarray, abp_win: np.ndarray | None) -> np.ndarray | None:
    icp = icp_win.copy()
    icp[np.isnan(icp)] = np.nanmean(icp)

    # cardiac_amplitude
    cardiac = _bandpass(icp, 0.7, 2.5)
    card_amp = float(cardiac.max() - cardiac.min())

    # cardiac_frequency
    freqs = np.fft.rfftfreq(len(cardiac), d=1.0 / TARGET_FS)
    fft   = np.abs(np.fft.rfft(cardiac))
    mask  = (freqs >= 0.7) & (freqs <= 2.5)
    if not mask.any():
        return None
    card_freq = float(freqs[mask][np.argmax(fft[mask])])

    # respiratory_amplitude
    resp = _bandpass(icp, 0.1, 0.5)
    resp_amp = float(resp.max() - resp.min())

    # wavelet decomposition (db4, level 5, at 125 Hz)
    coeffs   = pywt.wavedec(icp, WAVELET, level=WAVELET_LEVEL)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total_e  = sum(energies) + 1e-9
    slow_power    = (energies[0] + energies[1]) / total_e  # cA5 + cD5
    cardiac_power = energies[2] / total_e                  # cD4

    # MAP from ABP (real value, not constant)
    if abp_win is not None and not np.isnan(abp_win).all():
        abp = abp_win.copy()
        abp[np.isnan(abp)] = np.nanmean(abp)
        map_est = float(
            np.percentile(abp, 33) +
            (np.percentile(abp, 90) - np.percentile(abp, 10)) / 3.0
        )
    else:
        # Physiological fallback: CPP ≈ 70 → MAP ≈ ICP_mean + 70
        map_est = float(np.nanmean(icp) + 70.0)

    feat = np.array([
        card_amp, card_freq, resp_amp,
        slow_power, cardiac_power, map_est,
    ], dtype=np.float32)

    for i, (lo, hi) in enumerate(CLIP_RANGES):
        feat[i] = float(np.clip(feat[i], lo, hi))

    return feat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--charis_dir",   type=Path, default=Path("data/raw/charis"))
    parser.add_argument("--out_dir",      type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.charis_dir.exists():
        print(f"ERROR: {args.charis_dir} not found. Run download_charis.py first.")
        sys.exit(1)

    records = sorted([f.stem for f in args.charis_dir.glob("*.hea")])
    if not records:
        print(f"ERROR: No .hea files in {args.charis_dir}")
        sys.exit(1)

    print(f"\n-- Extracting CHARIS at 125 Hz ({len(records)} records) -------------")
    all_feats, all_labels, all_ids = [], [], []

    for rec_name in records:
        pid_str = "".join(filter(str.isdigit, rec_name))
        if not pid_str:
            continue
        pid = int(pid_str)

        try:
            rec = wfdb.rdrecord(str(args.charis_dir / rec_name))
        except Exception as e:
            print(f"  SKIP {rec_name}: {e}")
            continue

        sig_names = [s.upper() for s in rec.sig_name]
        orig_fs   = rec.fs

        icp_idx = next((i for i, n in enumerate(sig_names) if n in ICP_CHANNELS), None)
        abp_idx = next((i for i, n in enumerate(sig_names) if n in ABP_CHANNELS), None)

        if icp_idx is None:
            print(f"  SKIP {rec_name}: no ICP channel ({sig_names})")
            continue

        icp_raw = rec.p_signal[:, icp_idx].astype(np.float32)
        abp_raw = rec.p_signal[:, abp_idx].astype(np.float32) if abp_idx is not None else None

        icp_125 = _resample(icp_raw, orig_fs)
        abp_125 = _resample(abp_raw, orig_fs) if abp_raw is not None else None

        n_windows = len(icp_125) // WINDOW_SAMPLES
        n_ok = n_skip = 0

        for w in range(n_windows):
            s, e = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
            icp_win = icp_125[s:e]
            abp_win = abp_125[s:e] if abp_125 is not None else None

            if not _is_valid(icp_win):
                n_skip += 1
                continue

            label = 1 if float(np.nanmean(icp_win)) >= ICP_THRESHOLD else 0
            feat  = _extract_features(icp_win, abp_win)
            if feat is None:
                n_skip += 1
                continue

            all_feats.append(feat)
            all_labels.append(label)
            all_ids.append(pid)
            n_ok += 1

        n_pat = sum(1 for l in all_labels[-n_ok:] if l == 0)
        n_abn = n_ok - n_pat
        abp_tag = "ABP ok" if abp_idx is not None else "no ABP"
        print(f"  {rec_name} (fs={orig_fs}Hz, {abp_tag}): "
              f"{n_ok} windows (skip={n_skip}) | normal={n_pat} abnormal={n_abn}")

    if not all_feats:
        print("ERROR: No valid windows extracted from CHARIS.")
        sys.exit(1)

    X   = np.array(all_feats,  dtype=np.float32)
    y   = np.array(all_labels, dtype=np.int64)
    ids = np.array(all_ids,    dtype=np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "features.npy",    X)
    np.save(args.out_dir / "labels.npy",      y)
    np.save(args.out_dir / "patient_ids.npy", ids)

    print(f"\nCHARIS extraction complete:")
    print(f"  Windows:  {len(X):,}")
    print(f"  Patients: {len(np.unique(ids))}")
    print(f"  Normal:   {(y==0).sum():,}   Abnormal: {(y==1).sum():,}")
    print(f"  MAP range: [{X[:,5].min():.1f}, {X[:,5].max():.1f}] mmHg  "
          f"(std={X[:,5].std():.2f})")
    print(f"  Saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
