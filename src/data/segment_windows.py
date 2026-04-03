"""
segment_windows.py
==================
Loads raw wfdb records and splits them into clean 10-second windows.

Usage:
    python segment_windows.py
"""

import logging
import numpy as np
import wfdb
from pathlib import Path
from typing import Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SECONDS      = 10
TARGET_FS           = 125            # resample everything to this rate
SAMPLES_PER_WINDOW  = WINDOW_SECONDS * TARGET_FS   # 1 250 samples
MAX_MISSING_FRAC    = 0.30           # skip window if >30 % NaN
ICP_MIN_MMHG        = 0.0
ICP_MAX_MMHG        = 50.0
FLATLINE_STD_THRESH = 0.01           # mmHg; below this → flatline
np.random.seed(42)


def _resample(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample a 1-D signal from orig_fs to target_fs using linear interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1-D waveform.
    orig_fs : int
        Original sampling frequency (Hz).
    target_fs : int
        Desired output frequency (Hz).

    Returns
    -------
    np.ndarray
        Resampled signal.
    """
    if orig_fs == target_fs:
        return signal
    n_orig   = len(signal)
    n_target = int(n_orig * target_fs / orig_fs)
    x_orig   = np.linspace(0, 1, n_orig)
    x_new    = np.linspace(0, 1, n_target)
    return np.interp(x_new, x_orig, signal)


def _is_valid_window(window: np.ndarray) -> bool:
    """Return True if the window passes quality checks.

    Parameters
    ----------
    window : np.ndarray
        Single 10-second ICP window (1 250 samples at 125 Hz).

    Returns
    -------
    bool
        True when the window is usable.
    """
    missing_frac = np.isnan(window).mean()
    if missing_frac > MAX_MISSING_FRAC:
        return False

    clean = window[~np.isnan(window)]
    if len(clean) == 0:
        return False

    # Physiologically impossible values
    if np.any(clean < ICP_MIN_MMHG) or np.any(clean > ICP_MAX_MMHG):
        return False

    # Flatline detection
    if clean.std() < FLATLINE_STD_THRESH:
        return False

    return True


def load_record_signal(
    record_path: str,
    signal_name: str = "ICP",
) -> tuple[np.ndarray | None, int]:
    """Load a single-channel signal from a wfdb record.

    Parameters
    ----------
    record_path : str
        Path to the record (without extension).
    signal_name : str
        Channel name to extract (case-insensitive).

    Returns
    -------
    tuple[np.ndarray | None, int]
        (signal array in mmHg, sampling frequency) or (None, 0) on failure.
    """
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as exc:
        logger.warning("Cannot read %s: %s", record_path, exc)
        return None, 0

    sig_names_upper = [s.upper() for s in record.sig_name]
    # Accept ICP or ABP as the target channel
    for candidate in [signal_name.upper(), "ABP", "ART", "AP"]:
        if candidate in sig_names_upper:
            idx = sig_names_upper.index(candidate)
            signal = record.p_signal[:, idx].astype(np.float32)
            return signal, int(record.fs)

    logger.warning("No suitable signal channel in %s (available: %s)",
                   record_path, record.sig_name)
    return None, 0


def segment_record(
    record_path: str,
    patient_id: int,
    signal_name: str = "ICP",
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """Yield valid 10-second windows from a single record.

    Parameters
    ----------
    record_path : str
        wfdb record path (no extension).
    patient_id : int
        Numeric patient identifier to tag each window.
    signal_name : str
        Preferred channel name.

    Yields
    ------
    tuple[np.ndarray, int, int]
        (window_array [1250], patient_id, start_sample_index)
    """
    signal, orig_fs = load_record_signal(record_path, signal_name)
    if signal is None:
        return

    if orig_fs != TARGET_FS:
        signal = _resample(signal, orig_fs, TARGET_FS)

    n_windows = len(signal) // SAMPLES_PER_WINDOW
    for i in range(n_windows):
        start = i * SAMPLES_PER_WINDOW
        window = signal[start : start + SAMPLES_PER_WINDOW].copy()
        if _is_valid_window(window):
            yield window, patient_id, start


def segment_all_records(
    charis_dir: str = "data/raw/charis",
    mimic_dir:  str = "data/raw/mimic",
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Segment all downloaded records into 10-second windows.

    Parameters
    ----------
    charis_dir : str
        Directory containing CHARIS records.
    mimic_dir : str
        Directory containing MIMIC-III records.

    Returns
    -------
    tuple[list[np.ndarray], list[int], list[int]]
        (windows, patient_ids, start_indices)
    """
    windows: list[np.ndarray] = []
    pids:    list[int]        = []
    starts:  list[int]        = []

    patient_counter = 1

    for source_dir, sig_name in [(charis_dir, "ICP"), (mimic_dir, "ABP")]:
        p = Path(source_dir)
        if not p.exists():
            logger.warning("Directory not found: %s", source_dir)
            continue
        hea_files = sorted(p.rglob("*.hea"))
        logger.info("Segmenting %d records from %s …", len(hea_files), source_dir)

        for hea in hea_files:
            record_path = str(hea.with_suffix(""))
            n_before = len(windows)
            for win, pid, st in segment_record(record_path, patient_counter, sig_name):
                windows.append(win)
                pids.append(pid)
                starts.append(st)
            n_added = len(windows) - n_before
            if n_added > 0:
                logger.info("  patient %d | %s | %d windows", patient_counter,
                            hea.stem, n_added)
                patient_counter += 1
            else:
                logger.warning("  No valid windows for %s", hea.stem)

    logger.info("Segmentation complete: %d total windows, %d patients.",
                len(windows), patient_counter - 1)
    return windows, pids, starts


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    wins, pids, starts = segment_all_records()
    # Quick sanity check
    if wins:
        arr = np.stack(wins)
        print(f"Windows shape : {arr.shape}")
        print(f"Unique patients: {len(set(pids))}")
