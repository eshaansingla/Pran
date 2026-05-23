"""
build_mimic_features.py
=======================
1. Scan MIMIC-III WDB for records with ICP channel (layout header check)
2. Stream-read ICP data via wfdb remote access (no full download)
3. Extract 6 features per 10-second window
4. Save:
   data/processed/mimic_features.npy     (N, 6)  float32
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

# ── Credentials (from environment — NEVER hardcode) ───────────────────────────
# Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD in your shell or .env file.
# See .env.example for the template.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

USERNAME = os.environ.get("PHYSIONET_USERNAME", "")
PASSWORD = os.environ.get("PHYSIONET_PASSWORD", "")
if not USERNAME or not PASSWORD:
    print("ERROR: Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables.")
    print("       See .env.example for the template.")
    sys.exit(1)
AUTH = (USERNAME, PASSWORD)

BASE_URL = "https://physionet.org/files/mimic3wdb/1.0"
MIMIC_DB = "mimic3wdb/1.0"

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FS          = 125
WINDOW_SAMPLES     = 1250           # 10 s x 125 Hz
MAX_MISSING_FRAC   = 0.30
ICP_MIN_MMHG       = 0.0
ICP_MAX_MMHG       = 60.0   # matches extract_charis_v1.py ICP_VALID_RANGE upper bound
FLATLINE_STD       = 0.01
PATIENT_ID_OFFSET  = 100            # MIMIC IDs start at 101
ICP_CHANNELS       = {"ICP"}        # exact channel names to accept

# Feature clip ranges (must match extract_charis_v1.py and model_loader.py)
CLIP_RANGES = [
    (5.0,  120.0),   # cardiac_amplitude
    (0.7,  2.5),     # cardiac_frequency
    (1.0,  50.0),    # respiratory_amplitude
    (0.30, 1.0),     # slow_wave_power
    (0.0,  0.40),    # cardiac_power
    (40.0, 200.0),   # mean_arterial_pressure
]

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
    # Check mean is in physiological range (matches extract_charis_v1.py approach)
    if not (ICP_MIN_MMHG <= clean.mean() <= ICP_MAX_MMHG):
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


def _extract_6(win: np.ndarray, abp_win: np.ndarray | None = None) -> np.ndarray:
    """Extract the 6 validated features (no phase-lag).

    Returns np.ndarray of shape (6,) or raises ValueError if features
    contain NaN or no ABP channel is available (caller should skip that window).
    """
    if np.any(np.isnan(win)):
        win = win.copy()
        median_val = float(np.nanmedian(win))
        if np.isnan(median_val):
            raise ValueError("Window is entirely NaN")
        win[np.isnan(win)] = median_val

    # 0 cardiac_amplitude  1.0-2.5 Hz peak-to-peak in um (matches CHARIS extractor)
    cb = _bandpass(win, 1.0, 2.5)
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

    # 3 & 4 slow_wave_power, cardiac_power (wavelet at 125 Hz — no resampling)
    try:
        import pywt
        coeffs  = pywt.wavedec(win, "db4", level=5)
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total    = sum(energies) + 1e-10
        slow_pwr = float(np.clip(energies[0] / total, 0, 1))  # cA5 only
        card_pwr = float(np.clip(energies[1] / total, 0, 1))  # cD5
    except ImportError:
        fq  = np.fft.rfftfreq(len(win), d=1.0 / TARGET_FS)
        pw  = np.abs(np.fft.rfft(win)) ** 2
        tot = pw.sum() + 1e-10
        slow_pwr = float(pw[(fq <= 1.56)].sum() / tot)
        card_pwr = float(pw[(fq > 1.56) & (fq <= 3.12)].sum() / tot)

    # 5 mean_arterial_pressure — clamp to physiological range [40, 200] mmHg
    # Formula matches CHARIS extractor (extract_charis_v1.py) for dataset consistency.
    if abp_win is not None:
        abp_clean = abp_win[~np.isnan(abp_win)] if np.any(np.isnan(abp_win)) else abp_win
        if len(abp_clean) > 0:
            map_val = float(np.clip(
                np.percentile(abp_clean, 33) + (np.percentile(abp_clean, 90) - np.percentile(abp_clean, 10)) / 3.0,
                40.0, 200.0
            ))
        else:
            raise ValueError("No ABP channel — window unusable for MAP-dependent features")
    else:
        raise ValueError("No ABP channel — window unusable for MAP-dependent features")

    feat = np.array([cardiac_amp, cardiac_freq, resp_amp, slow_pwr, card_pwr,
                     map_val], dtype=np.float32)

    # Guard: reject windows that produce NaN features
    if np.any(np.isnan(feat)):
        raise ValueError(f"NaN in extracted features: {feat}")

    # Clip to physiological ranges (matches CHARIS extractor)
    for i, (lo, hi) in enumerate(CLIP_RANGES):
        feat[i] = float(np.clip(feat[i], lo, hi))

    return feat


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.auth = AUTH
    # Allow up to 8 connections per host for scan throughput
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    return s

_SESSION = _make_session()


def _get(url: str, retries: int = 2, timeout: int = 10) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(5.0)   # back off on rate-limit
                continue
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.0)
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
    step: int = 1,
) -> list[str]:
    """
    Scan every `step`-th record for ICP + estimate recording length.
    Returns top_n unique patients sorted by estimated window count (longest first).
    step=1 scans all records — complete but takes ~30-60 min at 16 workers.
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

    # Deduplicate by subject_id — keep first occurrence per patient
    seen_subjects = set()
    deduped_dirs = []
    for rec_dir in icp_dirs:
        subj = _subject_id_from_rec_dir(rec_dir)
        if subj not in seen_subjects:
            seen_subjects.add(subj)
            deduped_dirs.append(rec_dir)
    icp_dirs = deduped_dirs

    return icp_dirs[:target]


# ── Phase 2: stream-read and extract features ─────────────────────────────────

def extract_from_record(rec_dir: str, patient_id: int) -> tuple[list, list, bool]:
    """
    Remote-read one MIMIC waveform record and extract features.
    Returns (features_list, labels_list, was_truncated).
    was_truncated=True means the recording had more data beyond MAX_SAMPLES_PER_SEG.
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
    was_truncated = False

    # Build list of segments to try — pre-check existence with fast HTTP HEAD
    segments_to_try = []
    if seg_name_to_use:
        segments_to_try.append(seg_name_to_use)
    else:
        # Try individual _000N segments; stop at first 404 (sequential numbering)
        for i in range(1, 11):   # max 10 segments to bound HTTP overhead
            seg_candidate = f"{rec_id}_{i:04d}"
            resp = _get(f"{BASE_URL}/{rec_dir}/{seg_candidate}.hea", retries=1, timeout=5)
            if resp is None:
                break  # segment doesn't exist — no point trying higher numbers
            segments_to_try.append(seg_candidate)

    for seg in segments_to_try:
        try:
            # Read header to check signals and detect truncation
            sh = wfdb.rdheader(seg, pn_dir=pn_dir)
            upper = [s.upper() for s in sh.sig_name]
            if "ICP" not in upper:
                continue

            icp_idx = upper.index("ICP")
            abp_idx = next((i for i, s in enumerate(upper)
                            if s in {"ABP", "ART", "AP", "ABPM", "ARTM"}), None)

            # Detect if this segment is longer than our cap
            seg_len = getattr(sh, "sig_len", None) or getattr(sh, "n_sig_len", None)
            if seg_len and int(seg_len) > MAX_SAMPLES_PER_SEG:
                was_truncated = True

            channels = [icp_idx] if abp_idx is None else [icp_idx, abp_idx]
            record = wfdb.rdrecord(seg, pn_dir=pn_dir, channels=channels,
                                   sampto=MAX_SAMPLES_PER_SEG)
            fs    = int(record.fs)
            p_sig = record.p_signal  # (N, n_channels)

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
                    feat  = _extract_6(win_icp, win_abp)
                    label = _assign_label(float(np.nanmean(win_icp)))
                    feats.append(feat)
                    labels.append(label)
                except (ValueError, Exception):
                    continue

        except Exception:
            continue

    return feats, labels, was_truncated


# ── Multiprocessing worker (module-level — Windows spawn requires picklable targets) ──

def _mp_extract_worker_q(rec_dir: str, result_q) -> None:
    """Runs in a subprocess. Puts (rec_dir, pid, feats, labels, truncated, err) into queue."""
    try:
        pid = _subject_id_from_rec_dir(rec_dir)
        feats, labels, truncated = extract_from_record(rec_dir, pid)
        result_q.put((rec_dir, pid, feats, labels, truncated, None))
    except Exception as exc:
        result_q.put((rec_dir, None, [], [], False, str(exc)))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

import json

SCAN_CP_FILE       = "mimic_scan_checkpoint.json"
EXTRACT_CP_FILE    = "mimic_extraction_checkpoint.json"
MIN_WINDOWS        = 100   # patient must have >= this many windows to count as "suitable"
CHECKPOINT_EVERY   = 10    # save to disk every N suitable patients
MAX_SAMPLES_PER_SEG = 450_000   # 1 hour at 125 Hz — cap per segment to bound download time


def _save_scan_checkpoint(out_dir: Path, checked: list[str],
                           icp_found: list[tuple[str, int]]) -> None:
    data = {
        "checked":   checked,
        "icp_found": [{"rec": r, "est_wins": w} for r, w in icp_found],
    }
    (out_dir / SCAN_CP_FILE).write_text(json.dumps(data))


def _load_scan_checkpoint(out_dir: Path) -> tuple[set[str], list[tuple[str, int]]]:
    cp = out_dir / SCAN_CP_FILE
    if not cp.exists():
        return set(), []
    data = json.loads(cp.read_text())
    checked  = set(data.get("checked", []))
    icp_found = [(d["rec"], d["est_wins"]) for d in data.get("icp_found", [])]
    print(f"  [scan checkpoint] {len(checked):,} records already checked, "
          f"{len(icp_found)} ICP found — resuming.")
    return checked, icp_found


def _save_extraction_checkpoint(out_dir: Path, features: list, labels: list,
                                 pids: list, completed: list[str],
                                 failed: list[str], large_recs: list[str]) -> None:
    if features:
        np.save(out_dir / "mimic_features.npy",
                np.vstack(features).astype(np.float32))
        np.save(out_dir / "mimic_labels.npy",
                np.array(labels, dtype=np.int64))
        np.save(out_dir / "mimic_patient_ids.npy",
                np.array(pids, dtype=np.int32))
    data = {"completed": completed, "failed": failed,
            "large_recs": large_recs, "n_windows": len(labels)}
    (out_dir / EXTRACT_CP_FILE).write_text(json.dumps(data, indent=2))
    print(f"  [checkpoint] {len(completed)} patients | {len(labels):,} windows | "
          f"{len(large_recs)} large (truncated) saved.", flush=True)


def _load_extraction_checkpoint(out_dir: Path):
    cp = out_dir / EXTRACT_CP_FILE
    if not cp.exists():
        return [], [], [], [], [], []
    data = json.loads(cp.read_text())
    completed  = data.get("completed",  [])
    failed     = data.get("failed",     [])
    large_recs = data.get("large_recs", [])
    if not completed:
        return [], [], [], completed, failed, large_recs
    try:
        features = list(np.load(out_dir / "mimic_features.npy"))
        labels   = list(np.load(out_dir / "mimic_labels.npy"))
        pids     = list(np.load(out_dir / "mimic_patient_ids.npy"))
        print(f"  [extraction checkpoint] {len(completed)} patients done, "
              f"{len(labels):,} windows | {len(large_recs)} large pending — resuming.")
        return features, labels, pids, completed, failed, large_recs
    except Exception as e:
        print(f"  [extraction checkpoint] Could not load arrays ({e}) — starting fresh.")
        return [], [], [], [], [], []


# ── Checkpointed full scan ────────────────────────────────────────────────────

def scan_all_icp_ranked_checkpointed(
    all_records: list[str],
    top_n: int,
    workers: int,
    step: int,
    out_dir: Path,
) -> list[str]:
    """
    Scan all records for ICP, checkpointing every 2000 records.
    Resumes from last checkpoint automatically on restart.
    """
    already_checked, results = _load_scan_checkpoint(out_dir)

    candidates = [r for r in all_records[::step] if r not in already_checked]
    total_to_check = len(already_checked) + len(candidates)
    print(f"  Scanning {len(candidates)} remaining records "
          f"({len(already_checked)} already done, {total_to_check} total) ...")

    CHUNK = 2000

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
        try:
            first    = text.strip().split("\n")[0].split()
            n_samp   = int(first[3]) if len(first) >= 4 else 0
            fs       = int(float(first[2])) if len(first) >= 3 else TARGET_FS
            est_wins = n_samp // (WINDOW_SAMPLES * max(1, fs // TARGET_FS))
        except Exception:
            est_wins = 1
        return rec_dir, max(est_wins, 1)

    for chunk_start in range(0, len(candidates), CHUNK):
        chunk = candidates[chunk_start:chunk_start + CHUNK]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(check_record, rec): rec for rec in chunk}
            for fut in as_completed(futures):
                rec_dir, est_wins = fut.result()
                already_checked.add(rec_dir)
                if est_wins > 0:
                    results.append((rec_dir, est_wins))

        done = len(already_checked)
        print(f"  ... {done:,}/{total_to_check:,} checked, "
              f"{len(results)} ICP found so far", flush=True)
        _save_scan_checkpoint(out_dir, list(already_checked), results)

    # Deduplicate by subject_id — keep longest recording per patient
    subject_best: dict[int, tuple[str, int]] = {}
    for rec_dir, est_wins in results:
        subj_id = _subject_id_from_rec_dir(rec_dir)
        if subj_id not in subject_best or est_wins > subject_best[subj_id][1]:
            subject_best[subj_id] = (rec_dir, est_wins)

    deduped  = sorted(subject_best.values(), key=lambda x: x[1], reverse=True)
    n_dupes  = len(results) - len(deduped)
    print(f"\n  Found {len(results)} ICP records -> {len(deduped)} unique patients "
          f"({n_dupes} duplicate stays removed).")
    print(f"  Top 5 by estimated windows:")
    for rd, ew in deduped[:5]:
        print(f"    {rd}  (~{ew} windows)")

    return [rd for rd, _ in deduped[:top_n]]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(target_patients: int, scan_step: int, out_dir: Path, best: bool = False) -> None:
    import wfdb

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  MIMIC-III ICP Feature Extraction Pipeline")
    mode = "full scan + checkpointed" if best else f"first-found (scan_step={scan_step})"
    print(f"  Target: {target_patients} patients | mode: {mode}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*62}\n")

    # ---- 1. Get full RECORDS list ----
    print("[1/4] Fetching RECORDS list from mimic3wdb ...")
    resp = requests.get(f"{BASE_URL}/RECORDS", auth=AUTH, timeout=30)
    if resp.status_code != 200:
        print(f"  FATAL: HTTP {resp.status_code} - cannot access MIMIC-III")
        sys.exit(1)
    all_records = [l.rstrip("/") for l in resp.text.strip().split("\n")]
    print(f"  Total records: {len(all_records)}")

    # ---- 2. Scan for ICP (checkpointed) ----
    if best:
        print(f"\n[2/4] Full checkpointed scan for all ICP patients ...")
        icp_records = scan_all_icp_ranked_checkpointed(
            all_records, top_n=target_patients, workers=4, step=scan_step, out_dir=out_dir
        )
    else:
        print(f"\n[2/4] Scanning for ICP signal ...")
        icp_records = scan_for_icp_records(
            all_records, scan_step=scan_step, target=target_patients
        )
    print(f"\n  Will extract from {len(icp_records)} ICP records.")
    if not icp_records:
        print("  ERROR: No ICP records found!")
        sys.exit(1)

    # ---- 3. Extract features (parallel subprocesses, hard kill-timeout) ----
    import multiprocessing as mp
    import time as _time

    EXTRACT_WORKERS = 6
    PATIENT_TIMEOUT = 300   # seconds — SIGKILL worker if it hangs this long

    print(f"\n[3/4] Extracting features (parallel={EXTRACT_WORKERS} workers, "
          f"{PATIENT_TIMEOUT}s kill-timeout, checkpointed every {CHECKPOINT_EVERY}) ...")

    all_features, all_labels, all_patient_ids, completed_recs, failed_recs, large_recs = \
        _load_extraction_checkpoint(out_dir)

    done_set  = set(completed_recs) | set(failed_recs) | set(large_recs)
    remaining = [r for r in icp_records if r not in done_set]

    print(f"  {len(completed_recs)} done, {len(failed_recs)} failed, "
          f"{len(large_recs)} large/truncated, {len(remaining)} remaining.")

    suitable_since_last_cp = 0
    total  = len(icp_records)
    offset = len(completed_recs) + len(failed_recs) + len(large_recs)

    def _handle_result(rec_dir, pid, feats, labels, truncated, err, global_idx, timed_out):
        nonlocal suitable_since_last_cp
        tag = " [TIMEOUT->large]" if timed_out else (" [TRUNCATED->large]" if truncated else "")

        if err and not timed_out:
            print(f"  [{global_idx}/{total}] {rec_dir} ... ERR: {err}", flush=True)
            failed_recs.append(rec_dir)
            return

        if timed_out:
            print(f"  [{global_idx}/{total}] {rec_dir} ...{tag}", flush=True)
            large_recs.append(rec_dir)
        elif not feats or len(feats) < MIN_WINDOWS:
            status = f"skip ({len(feats)} windows < {MIN_WINDOWS})" if feats else "no valid windows"
            print(f"  [{global_idx}/{total}] {rec_dir} ... {status}{tag}", flush=True)
            large_recs.append(rec_dir) if truncated else failed_recs.append(rec_dir)
        else:
            counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
            print(f"  [{global_idx}/{total}] {rec_dir} ... "
                  f"{len(feats)} windows  N={counts[0]} E={counts[1]} C={counts[2]}{tag}",
                  flush=True)
            all_features.extend(feats)
            all_labels.extend(labels)
            all_patient_ids.extend([pid] * len(feats))
            completed_recs.append(rec_dir)
            if truncated:
                large_recs.append(rec_dir)
            suitable_since_last_cp += 1

        if suitable_since_last_cp >= CHECKPOINT_EVERY:
            _save_extraction_checkpoint(
                out_dir, all_features, all_labels, all_patient_ids,
                completed_recs, failed_recs, large_recs
            )
            suitable_since_last_cp = 0

    # Process in batches of EXTRACT_WORKERS.
    # Each batch starts all workers simultaneously then joins each with the
    # remaining wall-clock time — any process still alive after PATIENT_TIMEOUT
    # is hard-killed (terminate -> kill) and logged as large/pending.
    for batch_start in range(0, len(remaining), EXTRACT_WORKERS):
        batch = remaining[batch_start:batch_start + EXTRACT_WORKERS]

        procs = []   # (rec_dir, Process, Queue, global_idx)
        for i, rec_dir in enumerate(batch):
            q = mp.Queue()
            p = mp.Process(target=_mp_extract_worker_q, args=(rec_dir, q), daemon=True)
            p.start()
            procs.append((rec_dir, p, q, offset + batch_start + i + 1))

        batch_t0 = _time.monotonic()
        for rec_dir, p, q, global_idx in procs:
            remaining_t = max(0.5, PATIENT_TIMEOUT - (_time.monotonic() - batch_t0))
            p.join(timeout=remaining_t)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                _handle_result(rec_dir, None, [], [], False, None, global_idx, timed_out=True)
            else:
                try:
                    rd, pid, feats, labels, truncated, err = q.get_nowait()
                except Exception as exc:
                    rd, pid, feats, labels, truncated, err = rec_dir, None, [], [], False, str(exc)
                _handle_result(rd, pid, feats, labels, truncated, err, global_idx, timed_out=False)

    # Final save
    if not all_features:
        print("\nFATAL: No features extracted!")
        sys.exit(1)

    _save_extraction_checkpoint(
        out_dir, all_features, all_labels, all_patient_ids,
        completed_recs, failed_recs, large_recs
    )

    print(f"\n{'='*62}")
    print(f"  EXTRACTION COMPLETE")
    lab_arr = np.array(all_labels, dtype=np.int64)
    print(f"  Patients  : {len(set(all_patient_ids))}")
    print(f"  Windows   : {len(all_features):,}")
    lc = np.bincount(lab_arr, minlength=3)
    for cls, name in enumerate(["Normal", "Elevated", "Critical"]):
        print(f"  {name:<10}: {lc[cls]:>7,}  ({100*lc[cls]/len(lab_arr):.1f}%)")
    print(f"  Large/pending: {len(large_recs)} patients truncated at 1hr — re-run without --best to extract remaining data")
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
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)   # required on Windows before any Process.start()
    main(args.target_patients, args.scan_step, args.out_dir, best=args.best)
