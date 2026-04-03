"""
extract_features.py
===================
Extracts 11 physiological features from a 10-second ICP/ABP window.

Feature vector (in order):
  [0]  cardiac_amplitude      – peak-to-peak TM displacement at 1-2 Hz (μm)
  [1]  cardiac_frequency      – dominant frequency in cardiac band (Hz)
  [2]  respiratory_amplitude  – peak-to-peak displacement at 0.1-0.5 Hz (μm)
  [3]  slow_wave_power        – normalised cA5 wavelet energy (0-1)
  [4]  cardiac_power          – normalised cD5 wavelet energy (0-1)
  [5]  mean_arterial_pressure – estimated MAP from ABP or default 90 mmHg
  [6]  head_angle             – 0 deg (supine; IMU not available in PhysioNet)
  [7]  motion_artifact_flag   – 0 (clean data; no ToF sensor in PhysioNet)
  [8]  phase_lag_mean         – circular mean of ICP-PPG phase difference (rad)
  [9]  phase_lag_std          – circular std of phase difference (rad)
  [10] phase_coherence        – phase locking value |mean(exp(i·Δφ))| (0-1)

Input: 1 250-sample window at 125 Hz (10 seconds), ICP or ABP in mmHg.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

# ── Constants ─────────────────────────────────────────────────────────────────
FS: int = 125                   # target sampling rate (Hz)
WINDOW_SAMPLES: int = 1250      # 10 s × 125 Hz
CARDIAC_LOW: float = 1.0        # Hz – cardiac bandpass low
CARDIAC_HIGH: float = 2.0       # Hz – cardiac bandpass high
RESP_LOW: float = 0.1           # Hz – respiratory bandpass low
RESP_HIGH: float = 0.5          # Hz – respiratory bandpass high
WAVELET_RESAMPLE_FS: int = 100  # Hz – resample before wavelet so bands align
PPG_DELAY_SAMPLES: int = 25     # 0.2 s at 125 Hz – simulated peripheral delay
DEFAULT_MAP: float = 90.0       # mmHg – population mean if ABP unavailable
ICP_SCALE_UM_PER_MMHG: float = 10.0  # 1 mmHg ≈ 10 μm TM displacement

try:
    import pywt
    _PYWT_OK = True
except ImportError:
    _PYWT_OK = False


# ── Signal processing helpers ──────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: float, low: float, high: float,
              order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.9999)
    if low_n >= high_n:
        return signal.copy()
    sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sosfiltfilt(sos, signal)


def _robust_peak_to_peak(signal: np.ndarray) -> float:
    """Robust peak-to-peak: 99th percentile minus 1st percentile."""
    return float(np.percentile(signal, 99) - np.percentile(signal, 1))


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * angles))))


def _circular_std(angles: np.ndarray) -> float:
    r = float(np.abs(np.mean(np.exp(1j * angles))))
    return float(np.sqrt(-2.0 * np.log(max(r, 1e-10))))


def _plv(angles: np.ndarray) -> float:
    """Phase locking value: |mean(exp(i·Δφ))|."""
    return float(np.abs(np.mean(np.exp(1j * angles))))


# ── Individual feature extractors ──────────────────────────────────────────────

def extract_cardiac_amplitude(icp_window: np.ndarray, fs: int = FS) -> float:
    """
    Bandpass 1-2 Hz and return peak-to-peak amplitude in μm.

    Parameters
    ----------
    icp_window : np.ndarray, shape (1250,) in mmHg
    fs : int

    Returns
    -------
    float – amplitude in μm (1 mmHg ≈ 10 μm)

    Examples
    --------
    >>> t = np.linspace(0, 10, 1250)
    >>> w = 3.0 * np.sin(2 * np.pi * 1.2 * t)
    >>> round(extract_cardiac_amplitude(w), 0)
    60.0
    """
    filtered = _bandpass(icp_window, fs, CARDIAC_LOW, CARDIAC_HIGH)
    return _robust_peak_to_peak(filtered) * ICP_SCALE_UM_PER_MMHG


def extract_cardiac_frequency(icp_window: np.ndarray, fs: int = FS) -> float:
    """
    Return dominant frequency in the 0.7-2.5 Hz band (Hz).

    Parameters
    ----------
    icp_window : np.ndarray, shape (1250,)
    fs : int

    Returns
    -------
    float – frequency in Hz

    Examples
    --------
    >>> t = np.linspace(0, 10, 1250)
    >>> w = np.sin(2 * np.pi * 1.1 * t)
    >>> abs(extract_cardiac_frequency(w) - 1.1) < 0.1
    True
    """
    filtered = _bandpass(icp_window, fs, 0.7, 2.5)
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    power = np.abs(np.fft.rfft(filtered)) ** 2
    mask = (freqs >= 0.7) & (freqs <= 2.5)
    if not mask.any():
        return 1.0
    return float(freqs[mask][np.argmax(power[mask])])


def extract_respiratory_amplitude(icp_window: np.ndarray, fs: int = FS) -> float:
    """
    Bandpass 0.1-0.5 Hz and return peak-to-peak amplitude in μm.

    Parameters
    ----------
    icp_window : np.ndarray, shape (1250,)
    fs : int

    Returns
    -------
    float – amplitude in μm
    """
    filtered = _bandpass(icp_window, fs, RESP_LOW, RESP_HIGH)
    return _robust_peak_to_peak(filtered) * ICP_SCALE_UM_PER_MMHG


def extract_wavelet_powers(
    icp_window: np.ndarray,
    fs: int = FS,
) -> tuple[float, float]:
    """
    5-level db4 wavelet decomposition. Returns normalised energies of
    cA5 (slow-wave band, 0-1.56 Hz) and cD5 (cardiac band, 1.56-3.12 Hz).

    Resamples the input to 100 Hz so frequency bands are consistent
    regardless of the original acquisition rate.

    Parameters
    ----------
    icp_window : np.ndarray, shape (N,)
    fs : int

    Returns
    -------
    slow_wave_power : float  in [0, 1]
    cardiac_power   : float  in [0, 1]
    """
    if _PYWT_OK:
        # Resample to 100 Hz for consistent wavelet frequency bins
        n_100 = int(len(icp_window) * WAVELET_RESAMPLE_FS / fs)
        x_100 = np.interp(
            np.linspace(0, 1, n_100),
            np.linspace(0, 1, len(icp_window)),
            icp_window,
        )
        coeffs = pywt.wavedec(x_100, "db4", level=5)
        # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total = sum(energies) + 1e-10
        slow_wave_power = energies[0] / total   # cA5
        cardiac_power = energies[1] / total      # cD5
    else:
        # FFT fallback: estimate power in corresponding frequency bands
        freqs = np.fft.rfftfreq(len(icp_window), d=1.0 / fs)
        power = np.abs(np.fft.rfft(icp_window)) ** 2
        total = power.sum() + 1e-10
        slow_wave_power = float(power[(freqs >= 0.0) & (freqs <= 1.56)].sum() / total)
        cardiac_power = float(power[(freqs > 1.56) & (freqs <= 3.12)].sum() / total)

    return (
        float(np.clip(slow_wave_power, 0.0, 1.0)),
        float(np.clip(cardiac_power, 0.0, 1.0)),
    )


def extract_map(
    icp_window: np.ndarray,
    abp_window: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate mean arterial pressure (mmHg).

    Uses ABP window mean if provided. If the ICP window looks like an ABP
    signal (mean > 50 mmHg), returns its mean as a MAP proxy. Otherwise
    returns the population default (90 mmHg).

    Parameters
    ----------
    icp_window : np.ndarray
    abp_window : np.ndarray or None

    Returns
    -------
    float – MAP in mmHg
    """
    if abp_window is not None:
        return float(np.nanmean(abp_window))
    sig_mean = float(np.nanmean(icp_window))
    if sig_mean > 50.0:           # signal is ABP not ICP
        return sig_mean
    return DEFAULT_MAP


def extract_phase_lag_features(
    icp_window: np.ndarray,
    ppg_window: Optional[np.ndarray] = None,
    fs: int = FS,
) -> tuple[float, float, float]:
    """
    Compute phase-lag features between ICP and PPG at the cardiac frequency.

    If PPG is unavailable, a synthetic PPG is simulated by time-shifting
    the ICP signal by 0.2 s (25 samples at 125 Hz), representing the
    peripheral transmission delay.

    Parameters
    ----------
    icp_window : np.ndarray, shape (1250,)
    ppg_window : np.ndarray or None
    fs : int

    Returns
    -------
    phase_lag_mean : float – circular mean of Δφ (radians)
    phase_lag_std  : float – circular std of Δφ (radians)
    phase_coherence: float – PLV in [0, 1]
    """
    icp_filt = _bandpass(icp_window, fs, CARDIAC_LOW, CARDIAC_HIGH)

    if ppg_window is not None:
        ppg_filt = _bandpass(ppg_window, fs, CARDIAC_LOW, CARDIAC_HIGH)
    else:
        # Simulate peripheral delay: shift ICP by PPG_DELAY_SAMPLES
        ppg_filt = np.roll(icp_filt, PPG_DELAY_SAMPLES)
        ppg_filt[:PPG_DELAY_SAMPLES] = 0.0

    phase_icp = np.angle(hilbert(icp_filt))
    phase_ppg = np.angle(hilbert(ppg_filt))
    delta_phi = phase_icp - phase_ppg

    return (
        _circular_mean(delta_phi),
        _circular_std(delta_phi),
        _plv(delta_phi),
    )


# ── Master extraction function ─────────────────────────────────────────────────

def extract_all_features(
    icp_window: np.ndarray,
    abp_window: Optional[np.ndarray] = None,
    ppg_window: Optional[np.ndarray] = None,
    fs: int = FS,
) -> np.ndarray:
    """
    Extract all 11 features from a single 10-second window.

    Parameters
    ----------
    icp_window : np.ndarray, shape (1250,)
        ICP or ABP proxy values in mmHg, sampled at ``fs`` Hz.
    abp_window : np.ndarray or None
        Simultaneous ABP window for MAP estimation.
    ppg_window : np.ndarray or None
        Simultaneous PPG window for phase-lag features.
    fs : int
        Sampling frequency (default 125 Hz).

    Returns
    -------
    features : np.ndarray, shape (11,), dtype float32

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> win = rng.normal(12, 2, 1250).astype(np.float32)
    >>> feats = extract_all_features(win)
    >>> feats.shape
    (11,)
    """
    # Replace any NaN with window median (simple imputation)
    if np.any(np.isnan(icp_window)):
        icp_window = icp_window.copy()
        icp_window[np.isnan(icp_window)] = np.nanmedian(icp_window)

    cardiac_amp = extract_cardiac_amplitude(icp_window, fs)
    cardiac_freq = extract_cardiac_frequency(icp_window, fs)
    resp_amp = extract_respiratory_amplitude(icp_window, fs)
    slow_pwr, card_pwr = extract_wavelet_powers(icp_window, fs)
    map_val = extract_map(icp_window, abp_window)
    phase_mean, phase_std, plv = extract_phase_lag_features(icp_window, ppg_window, fs)

    return np.array([
        cardiac_amp,   # [0]
        cardiac_freq,  # [1]
        resp_amp,      # [2]
        slow_pwr,      # [3]
        card_pwr,      # [4]
        map_val,       # [5]
        0.0,           # [6] head_angle – supine (no IMU in PhysioNet)
        0.0,           # [7] motion_artifact_flag – clean data
        phase_mean,    # [8]
        phase_std,     # [9]
        plv,           # [10]
    ], dtype=np.float32)


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_features_batch(
    windows: list[np.ndarray],
    fs: int = FS,
) -> np.ndarray:
    """
    Extract features for a list of windows.

    Parameters
    ----------
    windows : list[np.ndarray]
        Each element is a (1250,) array.
    fs : int

    Returns
    -------
    features : np.ndarray, shape (N, 11), dtype float32
    """
    from tqdm import tqdm
    results = []
    for win in tqdm(windows, desc="Extracting features", unit="window", leave=False):
        results.append(extract_all_features(win, fs=fs))
    return np.vstack(results).astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(42)
    test_win = rng.normal(12, 2, WINDOW_SAMPLES).astype(np.float32)
    feats = extract_all_features(test_win)
    names = [
        "cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
        "slow_wave_power", "cardiac_power", "mean_arterial_pressure",
        "head_angle", "motion_artifact_flag",
        "phase_lag_mean", "phase_lag_std", "phase_coherence",
    ]
    print("Feature extraction test:")
    for n, v in zip(names, feats):
        print(f"  {n:<28} {v:.4f}")
