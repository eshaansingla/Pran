"""
validation.py
=============
Input validation for ICP feature vectors.
"""
from __future__ import annotations

import io
import csv
from typing import Any

from model_loader import FEATURE_NAMES, FEATURE_RANGES

N_FEATURES = len(FEATURE_NAMES)


class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_feature_vector(features: list[float]) -> list[str]:
    """Return list of error strings (empty if valid)."""
    errors: list[str] = []

    if len(features) != N_FEATURES:
        errors.append(
            f"Expected {N_FEATURES} features, got {len(features)}"
        )
        return errors  # no point checking ranges

    for i, (name, val) in enumerate(zip(FEATURE_NAMES, features)):
        lo, hi = FEATURE_RANGES[name]
        if not (lo <= val <= hi):
            errors.append(
                f"Feature '{name}' = {val} is outside physiological range "
                f"[{lo}, {hi}]"
            )

    return errors


def parse_csv_bytes(raw: bytes) -> tuple[list[list[float]], list[str]]:
    """
    Parse raw CSV bytes into a list of feature rows.

    Returns (rows, errors).  'rows' may be empty if errors are fatal.
    Expected columns (header optional):
        cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
        slow_wave_power, cardiac_power, mean_arterial_pressure
    """
    errors: list[str] = []
    rows: list[list[float]] = []

    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    lines = list(reader)

    if not lines:
        errors.append("CSV file is empty")
        return rows, errors

    # Detect header row
    start = 0
    try:
        [float(v) for v in lines[0] if v.strip()]
    except ValueError:
        start = 1   # first row is a header

    if len(lines) - start == 0:
        errors.append("CSV contains header but no data rows")
        return rows, errors

    # Check column count from first data row
    first = [v.strip() for v in lines[start] if v.strip() != ""]
    if len(first) != N_FEATURES:
        errors.append(
            f"Invalid CSV: expected {N_FEATURES} columns, "
            f"found {len(first)} in row {start + 1}"
        )
        return rows, errors

    missing_rows: list[int] = []

    for lineno, line in enumerate(lines[start:], start=start + 1):
        cells = [c.strip() for c in line]
        # Skip blank lines
        if not any(cells):
            continue

        if len(cells) != N_FEATURES:
            errors.append(
                f"Row {lineno}: expected {N_FEATURES} values, "
                f"found {len(cells)}"
            )
            continue

        row: list[float] = []
        row_ok = True
        for j, cell in enumerate(cells):
            if cell in ("", "NA", "NaN", "nan", "null"):
                missing_rows.append(lineno)
                row_ok = False
                break
            try:
                row.append(float(cell))
            except ValueError:
                errors.append(
                    f"Row {lineno}, column '{FEATURE_NAMES[j]}': "
                    f"cannot parse '{cell}' as a number"
                )
                row_ok = False
                break

        if not row_ok:
            continue

        # Per-row range validation
        range_errors = validate_feature_vector(row)
        if range_errors:
            for e in range_errors:
                errors.append(f"Row {lineno}: {e}")
            # Still include the row (backend can choose to reject or warn)

        rows.append(row)

    if missing_rows:
        batch = missing_rows[:10]
        tail = f" (and {len(missing_rows)-10} more)" if len(missing_rows) > 10 else ""
        errors.append(f"Missing values in rows: {', '.join(map(str, batch))}{tail}")

    return rows, errors
