"""Shared test fixtures for ICP monitoring tests."""
import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_features():
    """A valid 6-feature vector within physiological ranges."""
    return [35.0, 1.2, 12.0, 0.95, 0.003, 80.0]


@pytest.fixture
def sample_features_abnormal():
    """A 6-feature vector suggesting elevated ICP."""
    return [65.0, 0.9, 8.0, 0.78, 0.015, 105.0]


@pytest.fixture
def sample_csv_bytes():
    """Valid CSV bytes with header + 5 data rows."""
    header = "cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,cardiac_power,mean_arterial_pressure"
    rows = [
        "35.0,1.2,12.0,0.95,0.003,80.0",
        "40.0,1.1,14.0,0.93,0.005,85.0",
        "55.0,0.9,9.0,0.80,0.012,100.0",
        "30.0,1.3,15.0,0.97,0.002,75.0",
        "45.0,1.0,11.0,0.88,0.008,90.0",
    ]
    return (header + "\n" + "\n".join(rows)).encode("utf-8")


@pytest.fixture
def sample_30_windows():
    """30 valid feature windows for LSTM forecasting."""
    np.random.seed(42)
    windows = []
    for _ in range(30):
        w = [
            np.random.uniform(25, 50),   # cardiac_amplitude
            np.random.uniform(0.9, 1.5), # cardiac_frequency
            np.random.uniform(8, 20),    # respiratory_amplitude
            np.random.uniform(0.90, 0.99), # slow_wave_power
            np.random.uniform(0.001, 0.01), # cardiac_power
            np.random.uniform(70, 95),   # mean_arterial_pressure
        ]
        windows.append(w)
    return windows
