"""
preprocess_hardware.py
======================
Convert raw hardware session CSVs into the 6-feature matrix used for training.

Drop all subject CSVs into:  data/raw/hardware/<subject_id>/session_*.csv
  OR                          data/raw/hardware/session_*.csv (auto-assigns IDs)

Subject ID assignment:
  - If CSV is inside a subfolder (e.g. data/raw/hardware/subj_01/), folder name = patient ID.
  - If CSV is directly in data/raw/hardware/, filename prefix = patient ID.

Output:
  data/processed/hw_features.npy   (N, 6)  float32
  data/processed/hw_labels.npy     (N,)    int64   — all 0 (normal)
  data/processed/hw_patient_ids.npy(N,)    int32

Usage:
  python preprocess_hardware.py
  python preprocess_hardware.py --valsalva   # include valsalva windows as label=1
  python preprocess_hardware.py --input_dir path/to/csvs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy import signal as sp_signal

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import (
    TARGET_FS, WINDOW_SAMPLES, WAVELET, WAVELET_LEVEL,
    FEATURE_NAMES, FEATURE_RANGES,
    DATA_RAW_HW, DATA_PROCESSED,
    SESSION_NORMAL, SESSION_VALSALVA, SESSION_RECOVERY,
    LABEL_NORMAL, LABEL_ABNORMAL,
)

# ── Feature extraction ────────────────────────────────────────────────────────

def _bandpass(sig: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, hi / nyq
    hi_n = min(hi_n, 0.999)
    b, a = sp_signal.butter(3, [lo_n, hi_n], btype="band")
    return sp_signal.filtfilt(b, a, sig)


def extract_features(window: pd.DataFrame, fs: float = TARGET_FS) -> np.ndarray | None:
    """
    window: DataFrame slice of WINDOW_SAMPLES rows with columns
            [ir_raw, disp_raw, ...]
    Returns: (6,) float32 feature vector, or None if window is invalid.
    """
    if len(window) < WINDOW_SAMPLES:
        return None

    ir   = window["ir_raw"].values.astype(np.float32)
    disp = window["disp_raw"].values.astype(np.float32)

    # Reject flat / contact-lost windows
    if ir.std() < 10 or disp.std() < 1:
        return None

    # ── cardiac_amplitude: peak-to-peak of IR in cardiac band (0.7–2.5 Hz) ──
    ir_cardiac  = _bandpass(ir, 0.7, 2.5, fs)
    cardiac_amp = float(ir_cardiac.max() - ir_cardiac.min())

    # ── cardiac_frequency: dominant FFT frequency in 0.7–2.5 Hz ─────────────
    freqs = np.fft.rfftfreq(len(ir_cardiac), d=1.0 / fs)
    fft   = np.abs(np.fft.rfft(ir_cardiac))
    mask  = (freqs >= 0.7) & (freqs <= 2.5)
    if mask.sum() == 0:
        return None
    cardiac_freq = float(freqs[mask][np.argmax(fft[mask])])

    # ── respiratory_amplitude: peak-to-peak of displacement in 0.1–0.5 Hz ──
    disp_resp  = _bandpass(disp, 0.1, 0.5, fs)
    resp_amp   = float(disp_resp.max() - disp_resp.min())

    # ── Wavelet decomposition on displacement (db4, level 5) ─────────────────
    coeffs     = pywt.wavedec(disp, WAVELET, level=WAVELET_LEVEL)
    energies   = [np.sum(c ** 2) for c in coeffs]
    total_e    = sum(energies) + 1e-9
    # approx (cA5) + detail-5 (cD5) ≈ 0–1.56 Hz → slow waves
    slow_power = float((energies[0] + energies[1]) / total_e)
    # detail-4 (cD4) ≈ 1.56–3.125 Hz → cardiac band
    cardiac_power = float(energies[2] / total_e)

    # ── mean_arterial_pressure: PPG amplitude proxy ───────────────────────────
    ir_amp      = float(ir.max() - ir.min())
    map_estimate = 60.0 + (ir_amp - 5000.0) * (60.0 / 75000.0)
    map_estimate = float(np.clip(map_estimate, 40.0, 200.0))

    feat = np.array([
        cardiac_amp, cardiac_freq, resp_amp,
        slow_power, cardiac_power, map_estimate,
    ], dtype=np.float32)

    # Range validation
    for i, (fname, (lo, hi)) in enumerate(FEATURE_RANGES.items()):
        if not (lo <= feat[i] <= hi):
            feat[i] = float(np.clip(feat[i], lo, hi))

    return feat


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_session(csv_path: Path, include_valsalva: bool = False):
    """Returns (features, labels) for one session CSV."""
    required = {"timestamp_ms", "ir_raw", "disp_raw", "artifact_flag", "session_label"}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read ({e})")
        return [], []

    missing = required - set(df.columns)
    if missing:
        print(f"  SKIP {csv_path.name}: missing columns {missing}")
        return [], []

    # Remove artifact rows
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)

    feats, labels = [], []
    n_windows = len(df) // WINDOW_SAMPLES

    for w in range(n_windows):
        sl = df.iloc[w * WINDOW_SAMPLES : (w + 1) * WINDOW_SAMPLES]
        # Determine label from majority session_label in window
        majority_label = sl["session_label"].mode()[0]

        if majority_label == SESSION_NORMAL:
            target = LABEL_NORMAL
        elif majority_label == SESSION_VALSALVA and include_valsalva:
            target = LABEL_ABNORMAL
        else:
            continue   # skip recovery and valsalva windows when not requested

        feat = extract_features(sl)
        if feat is not None:
            feats.append(feat)
            labels.append(target)

    return feats, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DATA_RAW_HW)
    parser.add_argument("--valsalva", action="store_true",
                        help="Include valsalva windows as abnormal (label=1)")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        print(f"ERROR: {input_dir} does not exist. Create it and drop CSVs inside.")
        sys.exit(1)

    all_feats, all_labels, all_ids = [], [], []
    patient_id = 1000   # hardware patient IDs start at 1000 to avoid overlap with MIMIC (1xx)

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files.")

    # Track subfolder-based patient IDs
    folder_id_map: dict[str, int] = {}

    for csv_path in csv_files:
        # Assign patient ID by parent folder name (if subject-organised)
        folder_key = csv_path.parent.name
        if folder_key not in folder_id_map:
            folder_id_map[folder_key] = patient_id
            patient_id += 1
        pid = folder_id_map[folder_key]

        print(f"  Processing {csv_path.name} (patient {pid}) ...", end=" ")
        feats, labels = load_session(csv_path, include_valsalva=args.valsalva)
        print(f"{len(feats)} windows")

        for f, l in zip(feats, labels):
            all_feats.append(f)
            all_labels.append(l)
            all_ids.append(pid)

    if not all_feats:
        print("No valid windows extracted. Check sensor contact and artifact rates.")
        sys.exit(1)

    X = np.array(all_feats,  dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    ids = np.array(all_ids,  dtype=np.int32)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / "hw_features.npy",    X)
    np.save(DATA_PROCESSED / "hw_labels.npy",      y)
    np.save(DATA_PROCESSED / "hw_patient_ids.npy", ids)

    print(f"\nSaved {len(X)} windows from {len(folder_id_map)} subjects.")
    print(f"  Normal (0): {(y==0).sum()}  |  Abnormal (1): {(y==1).sum()}")
    print(f"  Files: data/processed/hw_features.npy, hw_labels.npy, hw_patient_ids.npy")


if __name__ == "__main__":
    main()
