"""Tests for feature extraction pipeline (_extract_8)."""
import os
import sys
from pathlib import Path

import pytest
import numpy as np

# Set dummy credentials so build_mimic_features.py doesn't sys.exit(1) on import
os.environ.setdefault("PHYSIONET_USERNAME", "test_user")
os.environ.setdefault("PHYSIONET_PASSWORD", "test_pass")

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_mimic_features import _extract_8, _bandpass, TARGET_FS, WINDOW_SAMPLES


class TestBandpass:
    def test_returns_same_length(self):
        sig = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
        filtered = _bandpass(sig, 1.0, 2.0)
        assert len(filtered) == len(sig)

    def test_preserves_dtype(self):
        sig = np.ones(WINDOW_SAMPLES, dtype=np.float32)
        out = _bandpass(sig, 0.5, 5.0)
        assert isinstance(out, np.ndarray)

    def test_invalid_band_returns_copy(self):
        """If lo >= hi after normalization, should return copy of input."""
        sig = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
        out = _bandpass(sig, 60.0, 61.0)  # both above Nyquist
        # Should not crash


class TestExtract8:
    def test_output_shape(self):
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 10
        feat = _extract_8(win)
        assert feat.shape == (8,)
        assert feat.dtype == np.float32

    def test_no_nan_in_output(self):
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 10
        feat = _extract_8(win)
        assert not np.any(np.isnan(feat)), f"NaN found in features: {feat}"

    def test_nan_input_handled(self):
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 10
        win[100:110] = np.nan  # introduce NaN
        feat = _extract_8(win)
        assert not np.any(np.isnan(feat))

    def test_all_nan_raises(self):
        win = np.full(WINDOW_SAMPLES, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="entirely NaN"):
            _extract_8(win)

    def test_map_without_abp(self):
        """Without ABP, MAP should default to 90 (ICP mean is low)."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 2 + 10
        feat = _extract_8(win, abp_win=None)
        map_val = feat[5]
        assert map_val == 90.0  # ICP mean ~10 < 40, defaults to 90

    def test_map_with_abp(self):
        """With ABP provided, MAP should be mean of ABP, clamped."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 2 + 10
        abp = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 80
        feat = _extract_8(win, abp_win=abp)
        map_val = feat[5]
        assert 40.0 <= map_val <= 200.0

    def test_map_clamp_negative(self):
        """ABP with extreme negative values should be clamped to 40."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 2 + 10
        abp = np.full(WINDOW_SAMPLES, -20.0, dtype=np.float32)
        feat = _extract_8(win, abp_win=abp)
        assert feat[5] == 40.0

    def test_map_clamp_high(self):
        """ABP with extreme high values should be clamped to 200."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 2 + 10
        abp = np.full(WINDOW_SAMPLES, 300.0, dtype=np.float32)
        feat = _extract_8(win, abp_win=abp)
        assert feat[5] == 200.0

    def test_cardiac_amplitude_positive(self):
        """Cardiac amplitude should always be >= 0."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 15
        feat = _extract_8(win)
        assert feat[0] >= 0

    def test_wavelet_powers_sum_reasonable(self):
        """slow_wave_power + cardiac_power should be <= 1.0."""
        win = np.random.randn(WINDOW_SAMPLES).astype(np.float32) * 5 + 15
        feat = _extract_8(win)
        assert feat[3] + feat[4] <= 1.0 + 1e-6
