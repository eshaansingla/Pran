"""
reextract_clinical_50hz.py
==========================
Re-extract CHARIS + MIMIC features at 50 Hz using the SAME feature extraction
pipeline as preprocess_hardware.py.

WHY THIS MATTERS:
  The v1 pipeline extracted features at 125 Hz. At 50 Hz, db4 level-5 wavelet
  bands are completely different (slow_wave_power covers 0-0.78 Hz at 50 Hz
  vs 0-1.95 Hz at 125 Hz). Mixing features extracted at different rates means
  the model learns the extraction artifact, not the physiology.

CHARIS: reads from data/raw/charis/ (local .dat files — no download needed)
MIMIC:  streams from PhysioNet (requires PHYSIONET_USERNAME + PHYSIONET_PASSWORD
        in environment). Set credentials in .env file.

Output:
  data/processed/charis_50hz_features.npy
  data/processed/charis_50hz_labels.npy
  data/processed/charis_50hz_patient_ids.npy
  data/processed/mimic_50hz_features.npy    (if credentials available)
  data/processed/mimic_50hz_labels.npy
  data/processed/mimic_50hz_patient_ids.npy
  data/processed/clinical_50hz_features.npy  (merged)
  data/processed/clinical_50hz_labels.npy
  data/processed/clinical_50hz_patient_ids.npy

Usage:
  python reextract_clinical_50hz.py --charis_only   # just CHARIS (no credentials needed)
  python reextract_clinical_50hz.py                 # CHARIS + MIMIC
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal as sp_signal
import pywt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import (
    TARGET_FS, WINDOW_SAMPLES, WAVELET, WAVELET_LEVEL,
    FEATURE_NAMES, FEATURE_RANGES, DATA_PROCESSED,
    ICP_THRESHOLD,
)

# ── Import feature extractor from preprocess_hardware (shared logic) ──────────
# We reuse the exact same bandpass + wavelet extraction so features are
# numerically identical to hardware features at the same sampling rate.
from preprocess_hardware import _bandpass

CHARIS_DIR   = ROOT / "data" / "raw" / "charis"
ICP_CHANNELS = {"ICP", "ICP1", "ICP2"}
ABP_CHANNELS = {"ABP", "ART", "ABPI", "ABP1"}

MAX_MISSING_FRAC = 0.20
ICP_VALID_RANGE  = (0.0, 60.0)   # mmHg — physiological range for clinical ICP
ABP_VALID_RANGE  = (40.0, 200.0) # mmHg

CHARIS_ID_OFFSET = 0    # CHARIS patient IDs: 1–13
MIMIC_ID_OFFSET  = 100  # MIMIC patient IDs: 101–253


# ── Feature extraction (identical logic to preprocess_hardware.py) ────────────

def _resample(sig: np.ndarray, orig_fs: int, target_fs: int = TARGET_FS) -> np.ndarray:
    if orig_fs == target_fs:
        return sig.astype(np.float32)
    n_new = int(len(sig) * target_fs / orig_fs)
    return np.interp(
        np.linspace(0, 1, n_new),
        np.linspace(0, 1, len(sig)),
        sig,
    ).astype(np.float32)


def _is_valid_window(icp_win: np.ndarray) -> bool:
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


def extract_clinical_features(icp_win: np.ndarray, abp_win: np.ndarray | None) -> np.ndarray | None:
    """
    Extract the same 6 features from clinical ICP/ABP waveforms.
    ICP waveform drives cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
    slow_wave_power, cardiac_power — all at 50 Hz, same wavelet bands as hardware.
    ABP drives mean_arterial_pressure.
    """
    icp = icp_win.copy()
    nan_mask = np.isnan(icp)
    if nan_mask.any():
        icp[nan_mask] = np.nanmean(icp)

    # ── cardiac_amplitude: peak-to-peak of ICP in cardiac band (0.7–2.5 Hz) ──
    icp_cardiac  = _bandpass(icp, 0.7, 2.5, TARGET_FS)
    cardiac_amp  = float(icp_cardiac.max() - icp_cardiac.min())

    # ── cardiac_frequency ────────────────────────────────────────────────────
    freqs = np.fft.rfftfreq(len(icp_cardiac), d=1.0 / TARGET_FS)
    fft   = np.abs(np.fft.rfft(icp_cardiac))
    mask  = (freqs >= 0.7) & (freqs <= 2.5)
    if mask.sum() == 0:
        return None
    cardiac_freq = float(freqs[mask][np.argmax(fft[mask])])

    # ── respiratory_amplitude: ICP in 0.1–0.5 Hz ─────────────────────────────
    icp_resp  = _bandpass(icp, 0.1, 0.5, TARGET_FS)
    resp_amp  = float(icp_resp.max() - icp_resp.min())

    # ── Wavelet (same as preprocess_hardware.py: db4, level 5, at 50 Hz) ─────
    coeffs     = pywt.wavedec(icp, WAVELET, level=WAVELET_LEVEL)
    energies   = [np.sum(c ** 2) for c in coeffs]
    total_e    = sum(energies) + 1e-9
    slow_power    = float((energies[0] + energies[1]) / total_e)  # 0–1.56 Hz
    cardiac_power = float(energies[2] / total_e)                  # 1.56–3.125 Hz

    # ── MAP: from ABP if available, else ICP mean as proxy ───────────────────
    if abp_win is not None and not np.isnan(abp_win).all():
        abp = abp_win.copy()
        abp[np.isnan(abp)] = np.nanmean(abp)
        # MAP ≈ diastolic + 1/3 pulse pressure
        map_est = float(np.percentile(abp, 33) + (np.percentile(abp, 90) - np.percentile(abp, 10)) / 3)
    else:
        # Crude fallback: CPP ≈ MAP - ICP; if mean ICP ≈ 15, and CPP ≈ 70, MAP ≈ 85
        map_est = float(np.nanmean(icp) + 70.0)

    map_est = float(np.clip(map_est, 40.0, 200.0))

    feat = np.array([
        cardiac_amp, cardiac_freq, resp_amp,
        slow_power, cardiac_power, map_est,
    ], dtype=np.float32)

    # Range clip (same as preprocess_hardware.py)
    for i, (_, (lo, hi)) in enumerate(FEATURE_RANGES.items()):
        feat[i] = float(np.clip(feat[i], lo, hi))

    return feat


# ── CHARIS extractor ──────────────────────────────────────────────────────────

def process_charis():
    print("\n── Extracting CHARIS at 50 Hz ──────────────────────────────────────")
    if not CHARIS_DIR.exists():
        print(f"  ERROR: {CHARIS_DIR} not found.")
        return None, None, None

    records = [f.stem for f in CHARIS_DIR.glob("*.hea")]
    if not records:
        print("  ERROR: No .hea files in charis dir.")
        return None, None, None

    all_feats, all_labels, all_ids = [], [], []

    for rec_name in sorted(records):
        pid_str = "".join(filter(str.isdigit, rec_name))
        if not pid_str:
            continue
        pid = int(pid_str) + CHARIS_ID_OFFSET

        try:
            rec = wfdb.rdrecord(str(CHARIS_DIR / rec_name))
        except Exception as e:
            print(f"  SKIP {rec_name}: {e}")
            continue

        sig_names = [s.upper() for s in rec.sig_name]
        orig_fs   = rec.fs

        icp_idx = next((i for i, n in enumerate(sig_names) if n in ICP_CHANNELS), None)
        abp_idx = next((i for i, n in enumerate(sig_names) if n in ABP_CHANNELS), None)

        if icp_idx is None:
            print(f"  SKIP {rec_name}: no ICP channel (found: {sig_names})")
            continue

        icp_raw = rec.p_signal[:, icp_idx].astype(np.float32)
        abp_raw = rec.p_signal[:, abp_idx].astype(np.float32) if abp_idx is not None else None

        icp_50 = _resample(icp_raw, orig_fs)
        abp_50 = _resample(abp_raw, orig_fs) if abp_raw is not None else None

        n_windows = len(icp_50) // WINDOW_SAMPLES
        n_feat, n_skip = 0, 0

        for w in range(n_windows):
            s, e = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
            icp_win = icp_50[s:e]
            abp_win = abp_50[s:e] if abp_50 is not None else None

            if not _is_valid_window(icp_win):
                n_skip += 1
                continue

            mean_icp = float(np.nanmean(icp_win))
            label    = 1 if mean_icp >= ICP_THRESHOLD else 0

            feat = extract_clinical_features(icp_win, abp_win)
            if feat is not None:
                all_feats.append(feat)
                all_labels.append(label)
                all_ids.append(pid)
                n_feat += 1
            else:
                n_skip += 1

        print(f"  {rec_name}: {n_feat} windows (skipped {n_skip}) — "
              f"normal={sum(1 for i in range(len(all_labels)-n_feat, len(all_labels)) if all_labels[i]==0)}, "
              f"abnormal={sum(1 for i in range(len(all_labels)-n_feat, len(all_labels)) if all_labels[i]==1)}")

    if not all_feats:
        print("  No valid windows from CHARIS.")
        return None, None, None

    return (np.array(all_feats,  dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_ids,    dtype=np.int32))


# ── MIMIC extractor ───────────────────────────────────────────────────────────

def process_mimic(target_patients: int = 95):
    print("\n── Extracting MIMIC at 50 Hz ───────────────────────────────────────")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    username = os.environ.get("PHYSIONET_USERNAME", "")
    password = os.environ.get("PHYSIONET_PASSWORD", "")
    if not username or not password:
        print("  SKIP: PHYSIONET_USERNAME / PHYSIONET_PASSWORD not set.")
        print("  Set credentials in .env file and re-run without --charis_only.")
        return None, None, None

    import requests
    auth = (username, password)
    base = "https://physionet.org/files/mimic3wdb/1.0"

    # Fetch record list
    try:
        r = requests.get(f"{base}/RECORDS", auth=auth, timeout=30)
        r.raise_for_status()
        records = [line.strip() for line in r.text.splitlines() if line.strip()]
    except Exception as e:
        print(f"  ERROR fetching RECORDS: {e}")
        return None, None, None

    all_feats, all_labels, all_ids = [], [], []
    processed = 0

    for rec_path in records:
        if processed >= target_patients:
            break
        try:
            rec = wfdb.rdrecord(rec_path, pn_dir="mimic3wdb/1.0")
        except Exception:
            continue

        sig_names = [s.upper() for s in rec.sig_name]
        icp_idx = next((i for i, n in enumerate(sig_names) if n in ICP_CHANNELS), None)
        if icp_idx is None:
            continue

        abp_idx = next((i for i, n in enumerate(sig_names) if n in ABP_CHANNELS), None)
        orig_fs  = rec.fs

        icp_raw = rec.p_signal[:, icp_idx].astype(np.float32)
        abp_raw = rec.p_signal[:, abp_idx].astype(np.float32) if abp_idx is not None else None

        icp_50 = _resample(icp_raw, orig_fs)
        abp_50 = _resample(abp_raw, orig_fs) if abp_raw is not None else None

        pid       = processed + MIMIC_ID_OFFSET + 1
        n_windows = len(icp_50) // WINDOW_SAMPLES
        n_feat    = 0

        for w in range(n_windows):
            s, e = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
            icp_win = icp_50[s:e]
            abp_win = abp_50[s:e] if abp_50 is not None else None
            if not _is_valid_window(icp_win):
                continue
            mean_icp = float(np.nanmean(icp_win))
            label    = 1 if mean_icp >= ICP_THRESHOLD else 0
            feat = extract_clinical_features(icp_win, abp_win)
            if feat is not None:
                all_feats.append(feat)
                all_labels.append(label)
                all_ids.append(pid)
                n_feat += 1

        if n_feat > 0:
            print(f"  {rec_path}: {n_feat} windows")
            processed += 1

    if not all_feats:
        return None, None, None

    return (np.array(all_feats,  dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_ids,    dtype=np.int32))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--charis_only", action="store_true",
                        help="Only process CHARIS (no PhysioNet credentials needed)")
    parser.add_argument("--mimic_patients", type=int, default=95)
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sources = []

    # CHARIS
    cx, cy, cids = process_charis()
    if cx is not None:
        np.save(DATA_PROCESSED / "charis_50hz_features.npy",    cx)
        np.save(DATA_PROCESSED / "charis_50hz_labels.npy",      cy)
        np.save(DATA_PROCESSED / "charis_50hz_patient_ids.npy", cids)
        print(f"\n  CHARIS: {len(cx)} windows | normal={( cy==0).sum()} | abnormal={(cy==1).sum()}")
        sources.append((cx, cy, cids))

    # MIMIC
    if not args.charis_only:
        mx, my, mids = process_mimic(target_patients=args.mimic_patients)
        if mx is not None:
            np.save(DATA_PROCESSED / "mimic_50hz_features.npy",    mx)
            np.save(DATA_PROCESSED / "mimic_50hz_labels.npy",      my)
            np.save(DATA_PROCESSED / "mimic_50hz_patient_ids.npy", mids)
            print(f"\n  MIMIC: {len(mx)} windows | normal={(my==0).sum()} | abnormal={(my==1).sum()}")
            sources.append((mx, my, mids))

    if not sources:
        print("\nERROR: No clinical features extracted.")
        sys.exit(1)

    # Merge
    X   = np.concatenate([s[0] for s in sources], axis=0)
    y   = np.concatenate([s[1] for s in sources], axis=0)
    ids = np.concatenate([s[2] for s in sources], axis=0)

    np.save(DATA_PROCESSED / "clinical_50hz_features.npy",    X)
    np.save(DATA_PROCESSED / "clinical_50hz_labels.npy",      y)
    np.save(DATA_PROCESSED / "clinical_50hz_patient_ids.npy", ids)

    print(f"\nMerged clinical (50 Hz): {len(X)} windows from {len(np.unique(ids))} patients")
    print(f"  Normal (0): {(y==0).sum()}  |  Abnormal (1): {(y==1).sum()}")
    print("\nNow update build_hybrid_dataset.py to use clinical_50hz_features.npy")
    print("instead of features.npy (which was extracted at 125 Hz).")


if __name__ == "__main__":
    main()
