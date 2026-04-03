"""
generate_labels.py
==================
Assigns ICP class labels from raw ICP waveform windows.

Classes:
  0 – Normal   (median ICP < 15 mmHg)
  1 – Elevated (15 ≤ median ICP < 20 mmHg)
  2 – Critical (median ICP ≥ 20 mmHg)
"""

from __future__ import annotations
import numpy as np

# ── Thresholds ──────────────────────────────────────────────────────────────
ICP_NORMAL_MAX: float = 15.0    # mmHg – upper bound for Normal class
ICP_CRITICAL_MIN: float = 20.0  # mmHg – lower bound for Critical class


def assign_label(icp_median: float) -> int:
    """
    Map a scalar ICP median value to a class index.

    Parameters
    ----------
    icp_median : float
        Median ICP (or ABP proxy) over a 10-second window in mmHg.

    Returns
    -------
    int – 0 (Normal), 1 (Elevated), or 2 (Critical)

    Examples
    --------
    >>> assign_label(10.0)
    0
    >>> assign_label(17.5)
    1
    >>> assign_label(22.0)
    2
    >>> assign_label(15.0)   # boundary: exactly 15 → Elevated
    1
    >>> assign_label(20.0)   # boundary: exactly 20 → Critical
    2
    """
    if icp_median < ICP_NORMAL_MAX:
        return 0
    elif icp_median < ICP_CRITICAL_MIN:
        return 1
    else:
        return 2


def generate_labels(windows: list[np.ndarray]) -> np.ndarray:
    """
    Generate a label array from a list of raw ICP windows.

    Parameters
    ----------
    windows : list[np.ndarray]
        Each element is a (1250,) array of ICP values in mmHg.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype int64

    Examples
    --------
    >>> wins = [np.full(1250, v) for v in [10.0, 17.5, 22.0]]
    >>> list(generate_labels(wins))
    [0, 1, 2]
    """
    labels = np.array(
        [assign_label(float(np.nanmedian(w))) for w in windows],
        dtype=np.int64,
    )
    return labels


if __name__ == "__main__":
    # Sanity checks
    for val, expected in [(5.0, 0), (14.9, 0), (15.0, 1), (19.9, 1), (20.0, 2), (35.0, 2)]:
        got = assign_label(val)
        status = "OK" if got == expected else f"FAIL (got {got})"
        print(f"  assign_label({val:5.1f}) = {got}  [{status}]")
