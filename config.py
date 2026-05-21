"""
config.py — Hybrid pipeline configuration (v2: hardware-normal + MIMIC/CHARIS-abnormal)
"""
from __future__ import annotations

FEATURE_NAMES = [
    "cardiac_amplitude",
    "cardiac_frequency",
    "respiratory_amplitude",
    "slow_wave_power",
    "cardiac_power",
    "mean_arterial_pressure",
]
N_FEATURES = len(FEATURE_NAMES)

# Hardware sampling parameters
TARGET_FS      = 50       # Hz — hardware timer interrupt rate
WINDOW_SAMPLES = 500      # 10 s × 50 Hz
WINDOW_SECONDS = 10.0

# Wavelet config (db4 level-5 at 50 Hz)
WAVELET        = "db4"
WAVELET_LEVEL  = 5
# At 50 Hz: approx band ≈ 0–0.78 Hz (slow waves + respiratory)
#           detail-4  band ≈ 1.56–3.125 Hz (cardiac)

FEATURE_RANGES = {
    "cardiac_amplitude":     (5.0, 120.0),   # μm proxy
    "cardiac_frequency":     (0.7, 2.5),     # Hz (42–150 bpm)
    "respiratory_amplitude": (1.0, 50.0),    # μm proxy
    "slow_wave_power":       (0.20, 1.0),    # dimensionless wavelet energy
    "cardiac_power":         (0.0, 0.50),    # dimensionless wavelet energy
    "mean_arterial_pressure":(40.0, 200.0),  # mmHg
}

# Labels
LABEL_NORMAL   = 0
LABEL_ABNORMAL = 1
CLASS_NAMES    = ["Normal", "Abnormal"]
ICP_THRESHOLD  = 15.0  # mmHg clinical threshold

# Hardware CSV session labels
SESSION_NORMAL   = 0
SESSION_VALSALVA = 3
SESSION_RECOVERY = 4

# Paths (relative to repo root)
from pathlib import Path
ROOT              = Path(__file__).parent
DATA_RAW_HW       = ROOT / "data" / "raw" / "hardware"    # drop hardware CSVs here
DATA_PROCESSED    = ROOT / "data" / "processed"
MODEL_DIR         = ROOT / "models"
RESULTS_DIR       = ROOT / "results"

# Existing MIMIC/CHARIS processed features (from v1 pipeline)
MIMIC_FEATURES    = DATA_PROCESSED / "mimic_features.npy"
MIMIC_LABELS      = DATA_PROCESSED / "mimic_labels.npy"
MIMIC_IDS         = DATA_PROCESSED / "mimic_patient_ids.npy"

# Hybrid dataset output
HW_FEATURES       = DATA_PROCESSED / "hw_features.npy"
HW_LABELS         = DATA_PROCESSED / "hw_labels.npy"
HW_IDS            = DATA_PROCESSED / "hw_patient_ids.npy"

HYBRID_FEATURES   = DATA_PROCESSED / "hybrid_features.npy"
HYBRID_LABELS     = DATA_PROCESSED / "hybrid_labels.npy"
HYBRID_IDS        = DATA_PROCESSED / "hybrid_patient_ids.npy"
