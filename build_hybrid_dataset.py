"""
build_hybrid_dataset.py
=======================
Merge hardware-normal features with MIMIC/CHARIS-abnormal features.
Runs distribution alignment check (KL divergence + MMD) before saving.

Inputs (must exist):
  data/processed/hw_features.npy          from preprocess_hardware.py
  data/processed/hw_labels.npy
  data/processed/hw_patient_ids.npy
  data/processed/mimic_features.npy       from v1 pipeline (already exists)
  data/processed/mimic_labels.npy
  data/processed/mimic_patient_ids.npy

Output:
  data/processed/hybrid_features.npy
  data/processed/hybrid_labels.npy
  data/processed/hybrid_patient_ids.npy

Usage:
  python build_hybrid_dataset.py
  python build_hybrid_dataset.py --skip_mmd    # faster, skip MMD check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import (
    FEATURE_NAMES, DATA_PROCESSED,
    HW_FEATURES, HW_LABELS, HW_IDS,
    MIMIC_FEATURES, MIMIC_LABELS, MIMIC_IDS,
    HYBRID_FEATURES, HYBRID_LABELS, HYBRID_IDS,
)


# ── Distribution checks ───────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    p_hist, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)
    p_hist = p_hist + 1e-9
    q_hist = q_hist + 1e-9
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(entropy(p_hist, q_hist))


def mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear-time MMD approximation (chunk to avoid OOM)."""
    n = min(500, len(X), len(Y))
    X_s = X[np.random.choice(len(X), n, replace=False)]
    Y_s = Y[np.random.choice(len(Y), n, replace=False)]
    XX = np.mean(np.dot(X_s, X_s.T))
    YY = np.mean(np.dot(Y_s, Y_s.T))
    XY = np.mean(np.dot(X_s, Y_s.T))
    return float(XX - 2 * XY + YY)


def check_distributions(hw_X: np.ndarray, clinical_X: np.ndarray, skip_mmd: bool = False):
    print("\n── Distribution Alignment Check ─────────────────────────────────")
    print(f"{'Feature':<28}  {'KL div':>8}  {'Status':>8}")
    print("-" * 52)

    kl_threshold = 0.5   # warn if KL > 0.5 nats
    any_fail = False

    for i, fname in enumerate(FEATURE_NAMES):
        kl = kl_divergence(hw_X[:, i], clinical_X[:, i])
        status = "OK" if kl < kl_threshold else "WARN"
        if status == "WARN":
            any_fail = True
        print(f"  {fname:<26}  {kl:>8.4f}  {status:>8}")

    if not skip_mmd:
        # Normalize before MMD
        mu  = hw_X.mean(axis=0)
        std = hw_X.std(axis=0) + 1e-9
        hw_norm  = (hw_X - mu) / std
        cli_norm = (clinical_X - mu) / std
        mmd = mmd_linear(hw_norm, cli_norm)
        print(f"\n  MMD (linear, normalized): {mmd:.6f}")
        if mmd > 0.1:
            print("  WARNING: MMD > 0.1 — consider hardware re-calibration.")
            any_fail = True

    if any_fail:
        print("\n  Some features show distribution mismatch.")
        print("  Check sensor calibration or consider domain adaptation.")
    else:
        print("\n  Distributions look aligned. Proceeding with merge.")
    print("─" * 52)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mmd", action="store_true")
    args = parser.parse_args()

    # Load hardware features
    for path in [HW_FEATURES, HW_LABELS, HW_IDS]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run preprocess_hardware.py first.")
            sys.exit(1)

    hw_X   = np.load(HW_FEATURES)
    hw_y   = np.load(HW_LABELS)
    hw_ids = np.load(HW_IDS)

    # Load MIMIC/CHARIS — MUST use 50 Hz re-extracted features for consistency.
    # The v1 features.npy was extracted at 125 Hz; wavelet bands differ from
    # hardware (50 Hz) so they CANNOT be mixed. Always prefer clinical_50hz_*.
    clinical_50hz = DATA_PROCESSED / "clinical_50hz_features.npy"
    combined_feat = DATA_PROCESSED / "features.npy"   # v1 (125 Hz) — fallback only

    if clinical_50hz.exists():
        cli_X   = np.load(clinical_50hz)
        cli_y   = np.load(DATA_PROCESSED / "clinical_50hz_labels.npy")
        cli_ids = np.load(DATA_PROCESSED / "clinical_50hz_patient_ids.npy")
        print(f"Loaded clinical (50 Hz re-extracted): {len(cli_X)} windows")
    elif combined_feat.exists():
        print("WARNING: Using v1 clinical features extracted at 125 Hz.")
        print("  slow_wave_power and cardiac_power wavelet bands WILL NOT match hardware.")
        print("  Run: python reextract_clinical_50hz.py --charis_only")
        print("  Then re-run this script.")
        cli_X   = np.load(combined_feat)
        cli_y   = np.load(DATA_PROCESSED / "labels.npy")
        cli_ids = np.load(DATA_PROCESSED / "patient_ids.npy")
        print(f"Loaded v1 MIMIC+CHARIS (125 Hz): {len(cli_X)} windows")
    elif MIMIC_FEATURES.exists():
        print("WARNING: Using MIMIC-only features at 125 Hz — run reextract_clinical_50hz.py")
        cli_X   = np.load(MIMIC_FEATURES)
        cli_y   = np.load(MIMIC_LABELS)
        cli_ids = np.load(MIMIC_IDS)
        print(f"Loaded MIMIC only (125 Hz): {len(cli_X)} windows")
    else:
        print("ERROR: No clinical features found. Run reextract_clinical_50hz.py first.")
        sys.exit(1)

    # Keep only abnormal from clinical (should already be labelled correctly)
    abn_mask = cli_y == 1
    print(f"  Clinical abnormal windows: {abn_mask.sum()} / {len(cli_y)}")

    # Use only normal from hardware
    norm_mask = hw_y == 0
    print(f"  Hardware normal windows:   {norm_mask.sum()} / {len(hw_y)}")

    hw_X_norm   = hw_X[norm_mask]
    hw_y_norm   = hw_y[norm_mask]
    hw_ids_norm = hw_ids[norm_mask]

    cli_X_abn   = cli_X[abn_mask]
    cli_y_abn   = cli_y[abn_mask]
    cli_ids_abn = cli_ids[abn_mask]

    # Distribution check: compare hardware-normal vs clinical-abnormal
    # (we check feature-by-feature to catch calibration issues)
    check_distributions(hw_X_norm, cli_X_abn, skip_mmd=args.skip_mmd)

    # Merge
    X   = np.concatenate([hw_X_norm,   cli_X_abn],   axis=0).astype(np.float32)
    y   = np.concatenate([hw_y_norm,   cli_y_abn],   axis=0).astype(np.int64)
    ids = np.concatenate([hw_ids_norm, cli_ids_abn], axis=0).astype(np.int32)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(HYBRID_FEATURES, X)
    np.save(HYBRID_LABELS,   y)
    np.save(HYBRID_IDS,      ids)

    print(f"\nHybrid dataset saved: {len(X)} windows total")
    print(f"  Normal (0): {(y==0).sum()}  |  Abnormal (1): {(y==1).sum()}")
    print(f"  Unique patients: {len(np.unique(ids))}")
    print(f"  Files: data/processed/hybrid_{{features,labels,patient_ids}}.npy")


if __name__ == "__main__":
    main()
