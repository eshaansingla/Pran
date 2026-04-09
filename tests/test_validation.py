"""Tests for validation.py — CSV parsing and range validation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "icp-monitor-web" / "backend"))
from validation import validate_feature_vector, parse_csv_bytes


class TestValidateFeatureVector:
    def test_valid_features(self, sample_features):
        errors = validate_feature_vector(sample_features)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_wrong_count(self):
        errors = validate_feature_vector([1.0, 2.0, 3.0])
        assert len(errors) == 1
        assert "Expected 6" in errors[0]

    def test_out_of_range_low(self):
        features = [1.0, 1.2, 12.0, 0.95, 0.003, 80.0]  # cardiac_amp too low
        errors = validate_feature_vector(features)
        assert any("cardiac_amplitude" in e for e in errors)

    def test_out_of_range_high(self):
        features = [35.0, 1.2, 12.0, 0.95, 0.5, 80.0]  # cardiac_power too high
        errors = validate_feature_vector(features)
        assert any("cardiac_power" in e for e in errors)

    def test_empty_list(self):
        errors = validate_feature_vector([])
        assert len(errors) > 0

    def test_boundary_values(self):
        """Features at exact boundaries should be valid."""
        features = [5.0, 0.7, 1.0, 0.30, 0.0, 40.0]
        errors = validate_feature_vector(features)
        assert errors == [], f"Boundary values should be valid: {errors}"


class TestParseCsvBytes:
    def test_valid_csv(self, sample_csv_bytes):
        rows, errors = parse_csv_bytes(sample_csv_bytes)
        assert len(rows) == 5
        assert all(len(r) == 6 for r in rows)

    def test_empty_csv(self):
        rows, errors = parse_csv_bytes(b"")
        assert len(rows) == 0
        assert any("empty" in e.lower() for e in errors)

    def test_header_only(self):
        csv = b"cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,cardiac_power,mean_arterial_pressure\n"
        rows, errors = parse_csv_bytes(csv)
        assert len(rows) == 0

    def test_no_header(self):
        csv = b"35.0,1.2,12.0,0.95,0.003,80.0\n40.0,1.1,14.0,0.93,0.005,85.0\n"
        rows, errors = parse_csv_bytes(csv)
        assert len(rows) == 2

    def test_wrong_columns(self):
        csv = b"1.0,2.0,3.0\n"
        rows, errors = parse_csv_bytes(csv)
        assert len(rows) == 0
        assert any("column" in e.lower() for e in errors)

    def test_nan_values(self):
        csv = b"35.0,NaN,12.0,0.95,0.003,80.0\n"
        rows, errors = parse_csv_bytes(csv)
        assert len(rows) == 0  # NaN rows are skipped
        assert any("Missing" in e for e in errors)

    def test_trailing_commas(self):
        """CSV with trailing commas (common artefact) should still parse."""
        csv = b"35.0,1.2,12.0,0.95,0.003,80.0,\n"
        rows, errors = parse_csv_bytes(csv)
        assert len(rows) == 1
