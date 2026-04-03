"""
save_processed_data.py
======================
Assembles features, labels, patient IDs, and metadata into the final
arrays consumed by the XGBoost training pipeline.

Normal mode  – reads data/raw/ records via segment_windows, extracts features
Synthetic    – generates a physiologically correlated synthetic dataset

Usage:
    python src/data/save_processed_data.py --synthetic          # no real data needed
    python src/data/save_processed_data.py                      # real PhysioNet data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/processed")
LOG_DIR    = Path("logs")

FEATURE_NAMES = [
    "cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
    "slow_wave_power", "cardiac_power", "mean_arterial_pressure",
    "head_angle", "motion_artifact_flag",
    "phase_lag_mean", "phase_lag_std", "phase_coherence",
]

# Physiologically motivated feature distributions per ICP class
# Each entry: (mean, std) for the 11 features in order
_SYNTH_PARAMS: dict[int, list[tuple[float, float]]] = {
    0: [  # Normal  ICP < 15 mmHg  ──  high cardiac amp, low slow-wave, low MAP
        (35.0, 5.0),   # cardiac_amplitude (μm)
        (1.20, 0.10),  # cardiac_frequency (Hz)
        (8.00, 1.50),  # respiratory_amplitude (μm)
        (0.12, 0.04),  # slow_wave_power
        (0.35, 0.06),  # cardiac_power
        (82.0, 8.0),   # MAP (mmHg)
        (0.00, 2.00),  # head_angle
        (0.00, 0.00),  # motion_artifact_flag (will be 0)
        (-0.30, 0.30), # phase_lag_mean (rad)
        (0.50, 0.10),  # phase_lag_std (rad)
        (0.75, 0.08),  # phase_coherence
    ],
    1: [  # Elevated  15-20 mmHg
        (20.0, 4.0),
        (1.10, 0.10),
        (6.00, 1.20),
        (0.40, 0.07),
        (0.50, 0.07),
        (97.0, 9.0),
        (0.00, 2.00),
        (0.00, 0.00),
        (-0.10, 0.30),
        (0.70, 0.10),
        (0.60, 0.10),
    ],
    2: [  # Critical  > 20 mmHg  ──  low cardiac amp, high slow-wave, high MAP
        (8.00, 3.00),
        (1.00, 0.10),
        (4.00, 1.00),
        (0.72, 0.08),
        (0.65, 0.06),
        (111.0, 10.0),
        (0.00, 2.00),
        (0.00, 0.00),
        (0.20, 0.40),
        (0.90, 0.10),
        (0.45, 0.12),
    ],
}

_FEATURE_CLIPS: list[tuple[float, float]] = [
    (1.0,   80.0),   # cardiac_amplitude
    (0.5,    2.5),   # cardiac_frequency
    (0.5,   20.0),   # respiratory_amplitude
    (0.0,    1.0),   # slow_wave_power
    (0.0,    1.0),   # cardiac_power
    (50.0, 180.0),   # MAP
    (-90.0, 90.0),   # head_angle
    (0.0,    0.0),   # motion_artifact_flag (always 0 for synthetic PhysioNet)
    (-3.15,  3.15),  # phase_lag_mean
    (0.0,    3.15),  # phase_lag_std
    (0.0,    1.0),   # phase_coherence
]


# ── Synthetic data generation ──────────────────────────────────────────────────

def generate_synthetic(
    n_patients: int = 50,
    windows_per_patient: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate a physiologically correlated synthetic dataset.

    Each patient is assigned a severity profile (mild / moderate / severe) that
    biases its ICP class distribution. Features are drawn from Gaussian
    distributions whose means differ substantially between classes, giving
    XGBoost enough signal to learn (expected macro-F1 ≈ 0.80+).

    Parameters
    ----------
    n_patients : int
    windows_per_patient : int
    seed : int

    Returns
    -------
    features    : np.ndarray, shape (N, 11), float32
    labels      : np.ndarray, shape (N,),    int64
    patient_ids : np.ndarray, shape (N,),    int64
    metadata    : pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    N = n_patients * windows_per_patient

    # Patient severity determines label probability profile
    severity_label_probs = {
        "mild":     [0.80, 0.15, 0.05],
        "moderate": [0.35, 0.45, 0.20],
        "severe":   [0.10, 0.30, 0.60],
    }
    severities = rng.choice(
        list(severity_label_probs.keys()),
        size=n_patients,
        p=[0.40, 0.35, 0.25],
    )

    features    = np.zeros((N, 11), dtype=np.float32)
    labels      = np.zeros(N,       dtype=np.int64)
    patient_ids = np.zeros(N,       dtype=np.int64)

    idx = 0
    for pat_i in range(n_patients):
        probs = severity_label_probs[severities[pat_i]]
        pat_labels = rng.choice([0, 1, 2], size=windows_per_patient, p=probs)

        for lbl in pat_labels:
            params = _SYNTH_PARAMS[lbl]
            feat = np.array(
                [rng.normal(m, max(s, 1e-6)) for m, s in params],
                dtype=np.float32,
            )
            # Clip each feature to physiological range
            for fi, (lo, hi) in enumerate(_FEATURE_CLIPS):
                feat[fi] = np.clip(feat[fi], lo, hi)
            feat[7] = 0.0  # motion_artifact_flag always 0

            features[idx]    = feat
            labels[idx]      = lbl
            patient_ids[idx] = pat_i + 1
            idx += 1

    # ICP medians consistent with labels (for metadata)
    icp_lookup = {0: 10.0, 1: 17.5, 2: 25.0}
    icp_medians = np.array(
        [icp_lookup[int(l)] + rng.normal(0, 1.0) for l in labels],
        dtype=np.float32,
    )
    timestamps = pd.date_range("2023-01-01", periods=N, freq="10s")
    metadata = pd.DataFrame({
        "window_id":  np.arange(N),
        "patient_id": patient_ids,
        "timestamp":  timestamps,
        "icp_median": icp_medians,
    })

    return features, labels, patient_ids, metadata


# ── Real-data assembly ─────────────────────────────────────────────────────────

def process_real_data(
    charis_dir: Path = Path("data/raw/charis"),
    mimic_dir: Path  = Path("data/raw/mimic"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run segmentation → feature extraction → label generation on real records.

    Returns arrays in the same format as ``generate_synthetic()``.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from segment_windows import segment_all_records
    from extract_features import extract_features_batch
    from generate_labels import generate_labels

    logging.info("Segmenting raw records …")
    windows, pids, starts = segment_all_records(
        charis_dir=str(charis_dir),
        mimic_dir=str(mimic_dir),
    )

    if not windows:
        raise RuntimeError(
            "No usable windows found in data/raw/. "
            "Run download_physionet.py first or use --synthetic."
        )

    logging.info("Extracting features from %d windows …", len(windows))
    features = extract_features_batch(windows)

    logging.info("Generating labels …")
    labels = generate_labels(windows)

    patient_ids = np.array(pids, dtype=np.int64)

    # Build metadata
    icp_medians = np.array([float(np.nanmedian(w)) for w in windows], dtype=np.float32)
    metadata = pd.DataFrame({
        "window_id":  np.arange(len(windows)),
        "patient_id": patient_ids,
        "timestamp":  starts,
        "icp_median": icp_medians,
    })

    return features, labels, patient_ids, metadata


# ── Saving ─────────────────────────────────────────────────────────────────────

def validate_and_save(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Validate shapes / types and save to output_dir.

    Parameters
    ----------
    features    : np.ndarray, shape (N, 11), float32
    labels      : np.ndarray, shape (N,),    int64
    patient_ids : np.ndarray, shape (N,),    int64
    metadata    : pd.DataFrame
    output_dir  : Path
    """
    N = len(features)
    assert features.shape == (N, 11),  f"Bad features shape: {features.shape}"
    assert labels.shape   == (N,),     f"Bad labels shape: {labels.shape}"
    assert patient_ids.shape == (N,),  f"Bad patient_ids shape: {patient_ids.shape}"
    assert set(np.unique(labels)).issubset({0, 1, 2}), "Unexpected label values"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "features.npy",    features.astype(np.float32))
    np.save(output_dir / "labels.npy",      labels.astype(np.int64))
    np.save(output_dir / "patient_ids.npy", patient_ids.astype(np.int64))
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # Report
    dist = {c: int((labels == c).sum()) for c in [0, 1, 2]}
    class_names = {0: "Normal", 1: "Elevated", 2: "Critical"}
    total = len(labels)

    print(f"\n{'='*50}")
    print(f"  Processed dataset saved to {output_dir}/")
    print(f"{'='*50}")
    print(f"  Total windows  : {total:,}")
    print(f"  Total patients : {len(np.unique(patient_ids))}")
    print(f"\n  Class distribution:")
    for cls, cnt in dist.items():
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    {class_names[cls]:<10} (class {cls}): {cnt:>7,}  ({100*cnt/total:5.1f}%)  {bar}")
    print()
    for fname in ["features.npy", "labels.npy", "patient_ids.npy", "metadata.csv"]:
        fpath = output_dir / fname
        size_mb = fpath.stat().st_size / 1024 / 1024
        print(f"  {fname:<22} {size_mb:.2f} MB")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Assemble and save processed ICP dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--synthetic",    action="store_true",
                   help="Generate correlated synthetic data (no PhysioNet needed).")
    p.add_argument("--n_patients",   type=int, default=50,
                   help="[synthetic] Number of virtual patients.")
    p.add_argument("--windows_each", type=int, default=200,
                   help="[synthetic] Windows per patient.")
    p.add_argument("--charis_dir",   type=Path, default=Path("data/raw/charis"))
    p.add_argument("--mimic_dir",    type=Path, default=Path("data/raw/mimic"))
    p.add_argument("--output_dir",   type=Path, default=OUTPUT_DIR)
    p.add_argument("--log_dir",      type=Path, default=LOG_DIR)
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_dir / "preprocessing.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if args.synthetic:
        logging.info("Generating synthetic correlated dataset (%d patients × %d windows) …",
                     args.n_patients, args.windows_each)
        features, labels, patient_ids, metadata = generate_synthetic(
            n_patients=args.n_patients,
            windows_per_patient=args.windows_each,
        )
    else:
        logging.info("Processing real PhysioNet records …")
        try:
            features, labels, patient_ids, metadata = process_real_data(
                charis_dir=args.charis_dir,
                mimic_dir=args.mimic_dir,
            )
        except RuntimeError as exc:
            logging.error("%s", exc)
            sys.exit(1)

    validate_and_save(features, labels, patient_ids, metadata, args.output_dir)


if __name__ == "__main__":
    main()
