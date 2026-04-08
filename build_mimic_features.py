"""
build_mimic_features.py
=======================
1. Scan MIMIC-III WDB for records with ICP channel (layout header check)
2. Stream-read ICP data via wfdb remote access (no full download)
3. Extract 8 features per 10-second window
4. Save:
   data/processed/mimic_features.npy     (N, 8)  float32
   data/processed/mimic_labels.npy       (N,)    int64
   data/processed/mimic_patient_ids.npy  (N,)    int32

Usage:
    python build_mimic_features.py
    python build_mimic_features.py --target_patients 50 --scan_step 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Credentials ────────────────────────────────────────────────────────────────
USERNAME = "eshaansingla2005"
PASSWORD = "+5Q5,,jdcy_ty8"
AUTH     = (USERNAME, PASSWORD)

os.environ["PHYSIONET_USERNAME"] = USERNAME
os.environ["PHYSIONET_PASSWORD"] = PASSWORD

BASE_URL = "https://physionet.org/files/mimic3wdb/1.0"
MIMIC_DB = "mimic3wdb/1.0"

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FS          = 125
WINDOW_SAMPLES     = 1250           # 10 s x 125 Hz
MAX_MISSING_FRAC   = 0.30
ICP_MIN_MMHG       = 0.0
ICP_MAX_MMHG       = 50.0
FLATLINE_STD       = 0.01
PATIENT_ID_OFFSET  = 100            # MIMIC IDs start at 101
ICP_CHANNELS       = {"ICP"}        # exact channel names to accept

# ── Signal helpers ─────────────────────────────────────────────────────────────

def _resample(sig: np.ndarray, orig_fs: int) -> np.ndarray:
    if orig_fs == TARGET_FS:
        return sig
    n_new = int(len(sig) * TARGET_FS / orig_fs)
    return np.interp(np.linspace(0, 1, n_new),
                     np.linspace(0, 1, len(sig)), sig).astype(np.float32)


def _is_valid_window(win: np.ndarray) -> bool:
    missing = np.isnan(win).mean()
    if missing > MAX_MISSING_FRAC:
        return False
    clean = win[~np.isnan(win)]
    if len(clean) < WINDOW_SAMPLES * 0.5:
        return False
    if np.any(clean < ICP_MIN_MMHG) or np.any(clean > ICP_MAX_MMHG):
        return False
    if clean.std() < FLATLINE_STD:
        return False
    return True


def _assign_label(icp_median: float) -> int:
    if icp_median < 15.0:
        return 0
    elif icp_median < 20.0:
        return 1
    return 2


# ── Feature extraction (no scipy/pywt dependency on import) ───────────────────

def _bandpass(sig: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    import warnings
    nyq = TARGET_FS / 2.0
    lo = max(low / nyq, 1e-4)
    hi = min(high / nyq, 0.9999)
    if lo >= hi:
        return sig.copy()
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sosfiltfilt(sos, sig)


def _extract_8(win: np.ndarray, abp_win: np.ndarray | None = None) -> np.ndarray:
    """Extract the 8 validated features (no phase-lag)."""
    if np.any(np.isnan(win)):
        win = win.copy()
        win[np.isnan(win)] = float(np.nanmedian(win))

    # 0 cardiac_amplitude  1-2 Hz peak-to-peak in um
    cb = _bandpass(win, 1.0, 2.0)
    cardiac_amp = (float(np.percentile(cb, 99) - np.percentile(cb, 1))) * 10.0

    # 1 cardiac_frequency  dominant freq 0.7-2.5 Hz
    fb = _bandpass(win, 0.7, 2.5)
    freqs = np.fft.rfftfreq(len(fb), d=1.0 / TARGET_FS)
    power = np.abs(np.fft.rfft(fb)) ** 2
    mask  = (freqs >= 0.7) & (freqs <= 2.5)
    cardiac_freq = float(freqs[mask][np.argmax(power[mask])]) if mask.any() else 1.0

    # 2 respiratory_amplitude  0.1-0.5 Hz
    rb = _bandpass(win, 0.1, 0.5)
    resp_amp = (float(np.percentile(rb, 99) - np.percentile(rb, 1))) * 10.0

    # 3 & 4 slow_wave_power, cardiac_power (wavelet)
    try:
        import pywt
        n100 = int(len(win) * 100 / TARGET_FS)
        x100 = np.interp(np.linspace(0, 1, n100), np.linspace(0, 1, len(win)), win)
        coeffs  = pywt.wavedec(x100, "db4", level=5)
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total    = sum(energies) + 1e-10
        slow_pwr = float(np.clip(energies[0] / total, 0, 1))
        card_pwr = float(np.clip(energies[1] / total, 0, 1))
    except ImportError:
        fq  = np.fft.rfftfreq(len(win), d=1.0 / TARGET_FS)
        pw  = np.abs(np.fft.rfft(win)) ** 2
        tot = pw.sum() + 1e-10
        slow_pwr = float(pw[(fq <= 1.56)].sum() / tot)
        card_pwr = float(pw[(fq > 1.56) & (fq <= 3.12)].sum() / tot)

    # 5 mean_arterial_pressure
    if abp_win is not None:
        map_val = float(np.nanmean(abp_win))
    else:
        sig_mean = float(np.nanmean(win))
        map_val  = sig_mean if sig_mean > 50.0 else 90.0

    return np.array([cardiac_amp, cardiac_freq, resp_amp, slow_pwr, card_pwr,
                     map_val, 0.0, 0.0], dtype=np.float32)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, auth=AUTH, timeout=20)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def _subject_id_from_rec_dir(rec_dir: str) -> int:
    """
    Extract the real MIMIC subject_id from a record path like '30/3000063'.
    The last component of the path IS the subject_id in MIMIC-III WDB.
    Strips any segment suffix (e.g. '3000063_0001' -> 3000063).
    """
    rec_name = rec_dir.split("/")[-1].split("_")[0]
    try:
        return int(rec_name)
    except ValueError:
        return abs(hash(rec_name)) % (10 ** 7)


def _parse_layout_signals(hea_text: str) -> list[str]:
    """Parse last token from lines 1.. of a header file as signal names."""
    sigs = []
    for line in hea_text.strip().split("\n")[1:]:
        parts = line.strip().split()
        if parts:
            sigs.append(parts[-1].upper())
    return sigs


# ── Phase 1a: scan ALL records, rank by estimated length ──────────────────────

def scan_all_icp_ranked(
    all_records: list[str],
    top_n: int = 100,
    workers: int = 16,
    step: int = 3,
) -> list[str]:
    """
    Scan every `step`-th record for ICP + estimate recording length.
    Returns top_n unique patients sorted by estimated window count (longest first).
    step=3 covers ~7000 records — thorough but finishes in ~15 min.
    """
    candidates = all_records[::step]
    print(f"  Scanning {len(candidates)} records (every {step}th of {len(all_records)}) for ICP + length ...")

    results: list[tuple[str, int]] = []   # (rec_dir, estimated_windows)
    done = 0
    total = len(all_records)

    def check_record(rec_dir: str) -> tuple[str, int]:
        rec_id = rec_dir.split("/")[-1]
        resp = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_layout.hea")
        if resp is None:
            resp = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_0001.hea")
        if resp is None:
            return rec_dir, 0
        text = resp.text
        sigs = _parse_layout_signals(text)
        if not (ICP_CHANNELS & set(sigs)):
            return rec_dir, 0
        # Estimate total samples from first header line: "rec n_segs fs n_samp"
        try:
            first = text.strip().split("\n")[0].split()
            # Layout header first line: recname nseg fs n_samp
            # Signal header first line: recname nsig fs n_samp
            n_samp = int(first[3]) if len(first) >= 4 else 0
            fs     = int(float(first[2])) if len(first) >= 3 else TARGET_FS
            est_wins = n_samp // (WINDOW_SAMPLES * max(1, fs // TARGET_FS))
        except Exception:
            est_wins = 1   # unknown length — include but rank low
        return rec_dir, max(est_wins, 1)

    total = len(candidates)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_record, rec): rec for rec in candidates}
        for fut in as_completed(futures):
            rec_dir, est_wins = fut.result()
            done += 1
            if est_wins > 0:
                results.append((rec_dir, est_wins))
            if done % 500 == 0:
                print(f"  ... {done}/{total} checked, {len(results)} ICP found so far")

    # Deduplicate: for each real subject_id keep only the longest record
    subject_best: dict[int, tuple[str, int]] = {}
    for rec_dir, est_wins in results:
        subj_id = _subject_id_from_rec_dir(rec_dir)
        if subj_id not in subject_best or est_wins > subject_best[subj_id][1]:
            subject_best[subj_id] = (rec_dir, est_wins)

    deduped = sorted(subject_best.values(), key=lambda x: x[1], reverse=True)
    n_dupes = len(results) - len(deduped)

    print(f"\n  Found {len(results)} ICP records -> {len(deduped)} unique patients"
          f" ({n_dupes} duplicate stays removed).")
    print(f"  Top 5 by estimated windows:")
    for rd, ew in deduped[:5]:
        print(f"    {rd}  (~{ew} windows, subject_id={_subject_id_from_rec_dir(rd)})")

    return [rd for rd, _ in deduped[:top_n]]


# ── Phase 1b: scan for ICP records (original fast scan) ───────────────────────

def scan_for_icp_records(
    all_records: list[str],
    scan_step: int = 20,
    target: int = 50,
    workers: int = 8,
) -> list[str]:
    """
    Check every `scan_step`-th record header for ICP channel.
    Returns list of record directory paths that have ICP.
    """
    candidates = all_records[::scan_step]
    print(f"  Scanning {len(candidates)} records (every {scan_step}th of {len(all_records)}) ...")

    icp_dirs: list[str] = []
    done = 0

    def check_record(rec_dir: str) -> tuple[str, bool]:
        rec_id  = rec_dir.split("/")[-1]
        # Try layout header first (multi-segment)
        resp = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_layout.hea")
        if resp is None:
            # Try first segment header
            resp = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_0001.hea")
        if resp is None:
            return rec_dir, False
        sigs = _parse_layout_signals(resp.text)
        return rec_dir, bool(ICP_CHANNELS & set(sigs))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_record, rec): rec for rec in candidates}
        for fut in as_completed(futures):
            rec_dir, has_icp = fut.result()
            done += 1
            if has_icp:
                icp_dirs.append(rec_dir)
                print(f"  [+] ICP found: {rec_dir}  (total: {len(icp_dirs)})")
            if done % 200 == 0:
                print(f"  ... {done}/{len(candidates)} checked, {len(icp_dirs)} ICP found")
            if len(icp_dirs) >= target:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    # If not enough, expand scan
    if len(icp_dirs) < target and scan_step > 1:
        print(f"  Found only {len(icp_dirs)}, expanding scan (step=5) ...")
        remaining = [r for r in all_records[::5] if r not in icp_dirs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(check_record, rec): rec
                       for rec in remaining[:3000]}
            for fut in as_completed(futures):
                rec_dir, has_icp = fut.result()
                done += 1
                if has_icp and rec_dir not in icp_dirs:
                    icp_dirs.append(rec_dir)
                    print(f"  [+] ICP found: {rec_dir}  (total: {len(icp_dirs)})")
                if len(icp_dirs) >= target:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    return icp_dirs[:target]


# ── Phase 2: stream-read and extract features ─────────────────────────────────

def extract_from_record(rec_dir: str, patient_id: int) -> tuple[list, list]:
    """
    Remote-read one MIMIC waveform record and extract 8 features.
    Returns (features_list, labels_list).
    """
    import wfdb
    rec_id = rec_dir.split("/")[-1]

    # wfdb pn_dir format: "mimic3wdb/1.0/30/3040891"
    pn_dir = f"mimic3wdb/1.0/{rec_dir}"

    try:
        header = wfdb.rdheader(f"{rec_id}_layout", pn_dir=pn_dir)
        seg_name_to_use = None
        # Multi-segment: find a segment that has ICP
        if hasattr(header, "seg_name") and header.seg_name:
            for sn, sl in zip(header.seg_name, header.seg_len):
                if sn and not sn.startswith("~") and sl and int(sl) >= WINDOW_SAMPLES:
                    # Read segment header to check for ICP
                    try:
                        sh = wfdb.rdheader(sn, pn_dir=pn_dir)
                        upper = [s.upper() for s in sh.sig_name]
                        if "ICP" in upper:
                            seg_name_to_use = sn
                            break
                    except Exception:
                        continue
    except Exception:
        # Not multi-segment - use rec_id directly
        seg_name_to_use = None

    feats: list[np.ndarray] = []
    labels: list[int] = []

    # Build list of segments to try
    segments_to_try = []
    if seg_name_to_use:
        segments_to_try.append(seg_name_to_use)
    else:
        # Try individual _000N segments
        for i in range(1, 30):
            segments_to_try.append(f"{rec_id}_{i:04d}")

    for seg in segments_to_try:
        try:
            # Read header only to check signals
            sh = wfdb.rdheader(seg, pn_dir=pn_dir)
            upper = [s.upper() for s in sh.sig_name]
            if "ICP" not in upper:
                continue

            icp_idx = upper.index("ICP")
            abp_idx = next((i for i, s in enumerate(upper)
                            if s in {"ABP", "ART", "AP", "ABPM", "ARTM"}), None)

            channels = [icp_idx] if abp_idx is None else [icp_idx, abp_idx]
            record = wfdb.rdrecord(seg, pn_dir=pn_dir, channels=channels)
            fs     = int(record.fs)
            p_sig  = record.p_signal  # (N, n_channels)

            icp_raw = p_sig[:, 0].astype(np.float32)
            abp_raw = p_sig[:, 1].astype(np.float32) if p_sig.shape[1] > 1 else None

            if fs != TARGET_FS:
                icp_raw = _resample(icp_raw, fs)
                if abp_raw is not None:
                    abp_raw = _resample(abp_raw, fs)

            n_wins = len(icp_raw) // WINDOW_SAMPLES
            for i in range(n_wins):
                s = i * WINDOW_SAMPLES
                win_icp = icp_raw[s:s + WINDOW_SAMPLES].copy()
                win_abp = (abp_raw[s:s + WINDOW_SAMPLES].copy()
                           if abp_raw is not None else None)
                if not _is_valid_window(win_icp):
                    continue
                try:
                    feat  = _extract_8(win_icp, win_abp)
                    label = _assign_label(float(np.nanmedian(win_icp)))
                    feats.append(feat)
                    labels.append(label)
                except Exception:
                    continue

        except Exception:
            continue

        if feats:
            break  # got enough from one segment

    return feats, labels


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(target_patients: int, scan_step: int, out_dir: Path, best: bool = False) -> None:
    import wfdb

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  MIMIC-III ICP Feature Extraction Pipeline")
    mode = "best (full scan, ranked by length)" if best else f"first-found (scan_step={scan_step})"
    print(f"  Target: {target_patients} patients | mode: {mode}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*62}\n")

    # ---- 1. Get full RECORDS list ----
    print("[1/4] Fetching RECORDS list from mimic3wdb ...")
    resp = requests.get(f"{BASE_URL}/RECORDS", auth=AUTH, timeout=30)
    if resp.status_code != 200:
        print(f"  FATAL: HTTP {resp.status_code} - cannot access MIMIC-III")
        print("  Ensure credentials are correct and you have PhysioNet access.")
        sys.exit(1)
    all_records = [l.rstrip("/") for l in resp.text.strip().split("\n")]
    print(f"  Total records: {len(all_records)}")

    # ---- 2. Scan for ICP ----
    if best:
        print(f"\n[2/4] Full scan — ranking all records by length, picking best {target_patients} ...")
        icp_records = scan_all_icp_ranked(all_records, top_n=target_patients, workers=16)
    else:
        print(f"\n[2/4] Scanning for ICP signal ...")
        icp_records = scan_for_icp_records(
            all_records, scan_step=scan_step, target=target_patients
        )
    print(f"\n  Found {len(icp_records)} ICP records.")
    if not icp_records:
        print("  ERROR: No ICP records found!")
        sys.exit(1)

    # ---- 3. Extract features ----
    print(f"\n[3/4] Extracting features (streaming, no full download) ...")
    all_features:    list[np.ndarray] = []
    all_labels:      list[int]        = []
    all_patient_ids: list[int]        = []

    MAX_NORMAL_PATIENTS = 5
    normal_patients_kept = 0

    for i, rec_dir in enumerate(icp_records):
        pid = _subject_id_from_rec_dir(rec_dir)   # real MIMIC subject_id
        print(f"  [{i+1:02d}/{len(icp_records)}] {rec_dir} (subject_id={pid}) ...", end=" ", flush=True)

        feats, labels = extract_from_record(rec_dir, pid)

        if not feats:
            print("no valid windows")
            continue

        counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
        print(f"{len(feats)} windows  N={counts[0]} E={counts[1]} C={counts[2]}")

        # Selective Balancing Logic for LSTM:
        # Drop patients completely lacking abnormal windows once we have enough negative references
        total_abnormal = counts[1] + counts[2]
        if total_abnormal < 5:
            if normal_patients_kept >= MAX_NORMAL_PATIENTS:
                print("  -> SKIPPING strictly Normal patient to balance dataset...")
                continue
            normal_patients_kept += 1

        all_features.extend(feats)
        all_labels.extend(labels)
        all_patient_ids.extend([pid] * len(feats))

    if not all_features:
        print("\nFATAL: No features extracted! Check network / credentials.")
        sys.exit(1)

    # ---- 4. Save ----
    print(f"\n[4/4] Saving ...")
    feat_arr = np.vstack(all_features).astype(np.float32)
    lab_arr  = np.array(all_labels, dtype=np.int64)
    pid_arr  = np.array(all_patient_ids, dtype=np.int32)

    np.save(out_dir / "mimic_features.npy",    feat_arr)
    np.save(out_dir / "mimic_labels.npy",      lab_arr)
    np.save(out_dir / "mimic_patient_ids.npy", pid_arr)

    print(f"\n{'='*62}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Patients  : {len(set(all_patient_ids))}")
    print(f"  Windows   : {len(all_features)}")
    lc = np.bincount(lab_arr, minlength=3)
    for cls, name in enumerate(["Normal", "Elevated", "Critical"]):
        print(f"  {name:<10}: {lc[cls]:>7,}  ({100*lc[cls]/len(lab_arr):.1f}%)")
    print(f"  Saved to  : {out_dir}/")
    print(f"{'='*62}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ICP features from MIMIC-III WDB (streaming)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target_patients", type=int, default=50,
                        help="Number of ICP patients to collect.")
    parser.add_argument("--scan_step",       type=int, default=20,
                        help="Check every Nth record header during scan.")
    parser.add_argument("--out_dir",         type=Path,
                        default=Path("data/processed"),
                        help="Output directory for mimic_*.npy files.")
    parser.add_argument("--best",            action="store_true",
                        help="Scan ALL records, rank by recording length, pick top N longest.")
    args = parser.parse_args()
    main(args.target_patients, args.scan_step, args.out_dir, best=args.best)
