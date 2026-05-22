"""
fix_charis_map.py
=================
Patch the MAP column (index 5) in data/processed/features.npy for CHARIS
patients (pid <= 13) using real ABP waveforms instead of the constant 90 mmHg
that was written by the original extraction script.

CHARIS has simultaneous ICP + ABP waveforms — the constant was an extraction
bug. This script replays the exact same windowing used in the v1 pipeline
(TARGET_FS=125, WINDOW_SAMPLES=1250) and computes per-window MAP from ABP.

Safety checks:
  - Only patches rows where patient_ids <= 13 (CHARIS)
  - Verifies window count per patient matches before patching
  - Saves backup to features_before_map_fix.npy

Usage:
    python fix_charis_map.py
    python fix_charis_map.py --dry_run   # print changes without saving
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data" / "processed"
CHARIS_DIR = ROOT / "data" / "raw" / "charis"

TARGET_FS     = 125
WINDOW_SAMPLES = 1250           # 10 s × 125 Hz
ICP_CHANNELS  = {"ICP", "ICP1", "ICP2"}
ABP_CHANNELS  = {"ABP", "ART", "ABPI", "ABP1"}
ICP_VALID_RANGE = (0.0, 60.0)   # mmHg
MAX_MISSING_FRAC = 0.20


def _resample(sig: np.ndarray, orig_fs: int, target_fs: int = TARGET_FS) -> np.ndarray:
    if orig_fs == target_fs:
        return sig.astype(np.float32)
    n_new = int(len(sig) * target_fs / orig_fs)
    return np.interp(
        np.linspace(0, 1, n_new),
        np.linspace(0, 1, len(sig)),
        sig,
    ).astype(np.float32)


def _is_valid_icp(icp_win: np.ndarray) -> bool:
    if np.isnan(icp_win).mean() > MAX_MISSING_FRAC:
        return False
    clean = icp_win[~np.isnan(icp_win)]
    if len(clean) < WINDOW_SAMPLES * 0.7:
        return False
    if clean.std() < 0.05:
        return False
    if not (ICP_VALID_RANGE[0] <= clean.mean() <= ICP_VALID_RANGE[1]):
        return False
    return True


def _map_from_abp(abp_win: np.ndarray) -> float:
    """MAP = diastolic + pulse_pressure/3, estimated from percentiles."""
    abp = abp_win.copy()
    abp[np.isnan(abp)] = np.nanmean(abp)
    map_est = float(
        np.percentile(abp, 33) +
        (np.percentile(abp, 90) - np.percentile(abp, 10)) / 3.0
    )
    return float(np.clip(map_est, 40.0, 200.0))


def extract_maps_for_record(rec_name: str) -> list[float] | None:
    """
    Replay the v1 windowing for one CHARIS record and return the per-window MAP
    list in the same order windows would have been extracted by the original pipeline.
    """
    try:
        rec = wfdb.rdrecord(str(CHARIS_DIR / rec_name))
    except Exception as e:
        print(f"  ERROR reading {rec_name}: {e}")
        return None

    sig_names = [s.upper() for s in rec.sig_name]
    orig_fs   = rec.fs

    icp_idx = next((i for i, n in enumerate(sig_names) if n in ICP_CHANNELS), None)
    abp_idx = next((i for i, n in enumerate(sig_names) if n in ABP_CHANNELS), None)

    if icp_idx is None:
        print(f"  SKIP {rec_name}: no ICP channel (found: {sig_names})")
        return None
    if abp_idx is None:
        print(f"  SKIP {rec_name}: no ABP channel (found: {sig_names})")
        return None

    icp_raw = rec.p_signal[:, icp_idx].astype(np.float32)
    abp_raw = rec.p_signal[:, abp_idx].astype(np.float32)

    icp_125 = _resample(icp_raw, orig_fs)
    abp_125 = _resample(abp_raw, orig_fs)

    maps: list[float] = []
    n_windows = len(icp_125) // WINDOW_SAMPLES

    for w in range(n_windows):
        s, e = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
        icp_win = icp_125[s:e]
        abp_win = abp_125[s:e]

        if not _is_valid_icp(icp_win):
            continue

        maps.append(_map_from_abp(abp_win))

    return maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print changes without saving")
    args = parser.parse_args()

    feat_path = DATA_DIR / "features.npy"
    pid_path  = DATA_DIR / "patient_ids.npy"

    for p in [feat_path, pid_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run the v1 extraction pipeline first.")
            sys.exit(1)

    if not CHARIS_DIR.exists():
        print(f"ERROR: {CHARIS_DIR} not found. Run download_charis.py first.")
        sys.exit(1)

    X   = np.load(feat_path).astype(np.float32)
    pid = np.load(pid_path).astype(np.int32)

    ch_mask = pid <= 13
    print(f"CHARIS rows in features.npy: {ch_mask.sum()} / {len(X)}")
    print(f"MAP before fix: mean={X[ch_mask, 5].mean():.2f}  "
          f"std={X[ch_mask, 5].std():.4f}  "
          f"min={X[ch_mask, 5].min():.2f}  max={X[ch_mask, 5].max():.2f}")

    if not args.dry_run:
        backup = DATA_DIR / "features_before_map_fix.npy"
        if not backup.exists():
            np.save(backup, X)
            print(f"Backup saved: {backup}")
        else:
            print(f"Backup already exists: {backup}")

    records = sorted([f.stem for f in CHARIS_DIR.glob("*.hea")])
    if not records:
        print(f"ERROR: No .hea files in {CHARIS_DIR}")
        sys.exit(1)

    X_patched = X.copy()
    total_patched = 0
    total_skipped_patients = 0

    print()
    for rec_name in records:
        pid_str = "".join(filter(str.isdigit, rec_name))
        if not pid_str:
            continue
        rec_pid = int(pid_str)

        patient_mask = pid == rec_pid
        n_expected   = patient_mask.sum()

        if n_expected == 0:
            print(f"  {rec_name}: no rows in features.npy (pid={rec_pid}) — skip")
            continue

        maps = extract_maps_for_record(rec_name)
        if maps is None:
            print(f"  {rec_name}: extraction failed — skipping patient (MAP stays 90)")
            total_skipped_patients += 1
            continue

        n_extracted = len(maps)

        if n_extracted != n_expected:
            print(f"  {rec_name}: window count MISMATCH "
                  f"(features.npy={n_expected}, re-extracted={n_extracted}) — "
                  f"skipping patient to avoid corruption")
            total_skipped_patients += 1
            continue

        maps_arr = np.array(maps, dtype=np.float32)
        before_mean = X_patched[patient_mask, 5].mean()

        X_patched[patient_mask, 5] = maps_arr

        after_mean = X_patched[patient_mask, 5].mean()
        print(f"  {rec_name}: {n_expected} windows patched | "
              f"MAP before={before_mean:.1f}  after={after_mean:.1f}  "
              f"range=[{maps_arr.min():.1f}, {maps_arr.max():.1f}]")
        total_patched += n_expected

    print(f"\nTotal windows patched: {total_patched}")
    if total_skipped_patients:
        print(f"Patients skipped (count mismatch or extract error): {total_skipped_patients}")

    if total_patched == 0:
        print("Nothing patched — no changes written.")
        sys.exit(0)

    # Verify the patch removed the constant
    print(f"\nMAP after fix: mean={X_patched[ch_mask, 5].mean():.2f}  "
          f"std={X_patched[ch_mask, 5].std():.4f}  "
          f"min={X_patched[ch_mask, 5].min():.2f}  max={X_patched[ch_mask, 5].max():.2f}")

    if X_patched[ch_mask, 5].std() < 0.1:
        print("WARNING: MAP is still nearly constant after fix — check ABP extraction.")

    if args.dry_run:
        print("\nDRY RUN — no files written.")
    else:
        np.save(feat_path, X_patched)
        print(f"\nPatched features saved: {feat_path}")
        print("NEXT STEP: retrain the model with:  python train_binary.py")


if __name__ == "__main__":
    main()
