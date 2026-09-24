"""Shared window features for the hardware recordings (identical at training and inference time).

Each 10-second window (500 samples at 50 Hz, 50 % overlap) becomes a 32-value vector:
  5 CHARIS-style features (unit-free, detrended): cardiac amplitude / frequency, respiratory amplitude,
    slow-wave power, cardiac power                                  -> BASE_NAMES
  27 optical features: for each of the IR, displacement and red channels, 6 relative band powers
    (0.1-0.3, 0.3-0.7, 0.7-1.2, 1.2-2.5, 2.5-6, 6-25 Hz), log-amplitude, skewness, kurtosis.
The motion sensor (IMU) is deliberately NOT used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pywt
from scipy import signal as sp
from scipy.stats import kurtosis, skew

FS, WIN, STEP = 50, 500, 250
REQUIRED_COLUMNS = {"ir_raw", "disp_raw", "red_raw", "artifact_flag"}
SESSION_NAMES = {0: "supine", 1: "head-up 30°", 2: "head-down 10°", 3: "Valsalva"}
ICP_ORDER = [1, 0, 2, 3]  # ascending expected ICP: head-up, supine, head-down, Valsalva

_NYQ = FS / 2.0
_B_CARD, _A_CARD = sp.butter(4, [1.0 / _NYQ, 2.5 / _NYQ], btype="band")
_B_RESP, _A_RESP = sp.butter(4, [0.1 / _NYQ, 0.5 / _NYQ], btype="band")
_FREQS = np.fft.rfftfreq(WIN, d=1.0 / FS)
_MASK = (_FREQS >= 0.7) & (_FREQS <= 2.5)
_EDGES = [0.1, 0.3, 0.7, 1.2, 2.5, 6.0, 25.0]

BASE_NAMES = ["cardiac_amp_rel", "cardiac_frequency", "resp_amp_rel", "slow_wave_power", "cardiac_power"]
_CH_NAMES = ["b0.1-0.3", "b0.3-0.7", "b0.7-1.2", "b1.2-2.5", "b2.5-6", "b6-25", "log_amp", "skew", "kurt"]
FEATURE_NAMES = BASE_NAMES + [f"{ch}_{n}" for ch in ("ir", "disp", "red") for n in _CH_NAMES]


def _base_features(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    a = sp.detrend(ir.astype(np.float64)); b = sp.detrend(disp.astype(np.float64))
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return None
    c = sp.filtfilt(_B_CARD, _A_CARD, a)
    card_amp = (np.percentile(c, 99) - np.percentile(c, 1)) / sa
    card_freq = _FREQS[_MASK][np.argmax((np.abs(np.fft.rfft(a)) ** 2)[_MASK])]
    r = sp.filtfilt(_B_RESP, _A_RESP, b)
    resp_amp = (np.percentile(r, 99) - np.percentile(r, 1)) / sb
    e = [float(np.sum(x ** 2)) for x in pywt.wavedec(b, "db4", level=5)]
    tot = sum(e) + 1e-12
    f = np.array([card_amp, card_freq, resp_amp, e[0] / tot, e[2] / tot], dtype=np.float32)
    return f if np.all(np.isfinite(f)) else None


def _channel_features(x: np.ndarray) -> list[float]:
    x = sp.detrend(x.astype(np.float64))
    p = np.abs(np.fft.rfft(x)) ** 2
    tot = p[(_FREQS >= 0.1) & (_FREQS <= 25)].sum() + 1e-12
    bands = [p[(_FREQS >= lo) & (_FREQS < hi)].sum() / tot for lo, hi in zip(_EDGES[:-1], _EDGES[1:])]
    return bands + [float(np.log10(x.std() + 1e-9)), float(skew(x)), float(kurtosis(x))]


def window_features(ir: np.ndarray, disp: np.ndarray, red: np.ndarray) -> np.ndarray | None:
    """32-value feature vector for one window, or None if the window is unusable (flat / dead channel)."""
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    base = _base_features(ir, disp)
    if base is None:
        return None
    vec = np.r_[base, _channel_features(ir), _channel_features(disp), _channel_features(red)]
    return np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def recording_windows(df: pd.DataFrame):
    """Slice one recording into windows -> (features [n,32], session labels [n] or None)."""
    df = df[df["artifact_flag"] == 0].reset_index(drop=True)
    ir, dp, rd = (df[c].to_numpy(float) for c in ("ir_raw", "disp_raw", "red_raw"))
    sl = df["session_label"].to_numpy() if "session_label" in df.columns else None
    feats, sess = [], []
    for w in range(max((len(df) - WIN) // STEP + 1, 0)):
        a, b = w * STEP, w * STEP + WIN
        f = window_features(ir[a:b], dp[a:b], rd[a:b])
        if f is None:
            continue
        feats.append(f)
        if sl is not None:
            sess.append(int(np.bincount(sl[a:b].astype(int)).argmax()))
    X = np.array(feats, dtype=np.float32) if feats else np.empty((0, len(FEATURE_NAMES)), np.float32)
    return X, (np.array(sess, dtype=int) if sl is not None else None)
