"""
download_physionet.py
=====================
Downloads CHARIS and MIMIC-III waveform data from PhysioNet.

Usage:
    python download_physionet.py

Environment variables required:
    PHYSIONET_USERNAME
    PHYSIONET_PASSWORD
"""

import os
import time
import logging
import wfdb
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHARIS_DB      = "charis"
CHARIS_VERSION = "1.0.0"
MIMIC_DB       = "mimic3wdb-matched"
MIMIC_VERSION  = "1.0"
MIMIC_MAX_PATIENTS = 50          # download at most this many MIMIC patients
RETRY_ATTEMPTS     = 3
RETRY_BASE_DELAY   = 5           # seconds; doubles on each retry


def _credentials() -> tuple[str, str]:
    """Read PhysioNet credentials from environment variables.

    Returns
    -------
    tuple[str, str]
        (username, password)

    Raises
    ------
    EnvironmentError
        If either variable is missing.
    """
    user = os.environ.get("PHYSIONET_USERNAME")
    pwd  = os.environ.get("PHYSIONET_PASSWORD")
    if not user or not pwd:
        raise EnvironmentError(
            "Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables "
            "before running the pipeline."
        )
    return user, pwd


def _download_with_retry(
    record_name: str,
    db_dir: str,
    pn_dir: str,
    dest: Path,
    attempt: int = 1,
) -> bool:
    """Download a single wfdb record with exponential-backoff retry.

    Parameters
    ----------
    record_name : str
        Record identifier within the database.
    db_dir : str
        Local destination directory.
    pn_dir : str
        PhysioNet directory string (db/version/subdir).
    dest : Path
        Full local path where the record lands.
    attempt : int
        Current attempt number (1-indexed).

    Returns
    -------
    bool
        True on success, False after all retries exhausted.
    """
    try:
        wfdb.dl_database(pn_dir, dl_dir=str(db_dir), records=[record_name])
        return True
    except Exception as exc:
        if attempt >= RETRY_ATTEMPTS:
            logger.warning("Failed to download %s after %d attempts: %s",
                           record_name, RETRY_ATTEMPTS, exc)
            return False
        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
        logger.info("Retry %d/%d for %s in %ds …", attempt, RETRY_ATTEMPTS,
                    record_name, delay)
        time.sleep(delay)
        return _download_with_retry(record_name, db_dir, pn_dir, dest, attempt + 1)


def download_charis(out_dir: str = "data/raw/charis") -> list[str]:
    """Download all CHARIS ICP waveform records.

    Parameters
    ----------
    out_dir : str
        Local directory to store raw CHARIS files.

    Returns
    -------
    list[str]
        List of successfully downloaded record names.
    """
    user, pwd = _credentials()
    os.environ["WFDB_PHYSIONET_USER"] = user
    os.environ["WFDB_PHYSIONET_PASS"] = pwd

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pn_dir = f"{CHARIS_DB}/{CHARIS_VERSION}"

    logger.info("Fetching CHARIS record list …")
    try:
        records = wfdb.get_record_list(pn_dir)
    except Exception as exc:
        logger.error("Cannot list CHARIS records: %s", exc)
        return []

    logger.info("Found %d CHARIS records.", len(records))
    downloaded: list[str] = []

    for rec in records:
        dest = Path(out_dir) / rec
        if dest.with_suffix(".hea").exists():
            logger.debug("Already present, skipping: %s", rec)
            downloaded.append(rec)
            continue
        ok = _download_with_retry(rec, out_dir, pn_dir, dest)
        if ok:
            downloaded.append(rec)
            logger.info("✓ CHARIS %s", rec)

    logger.info("CHARIS download complete: %d/%d records.", len(downloaded), len(records))
    return downloaded


def download_mimic(
    out_dir: str = "data/raw/mimic",
    max_patients: int = MIMIC_MAX_PATIENTS,
) -> list[str]:
    """Download a subset of MIMIC-III waveform records that contain ABP.

    Parameters
    ----------
    out_dir : str
        Local directory for raw MIMIC files.
    max_patients : int
        Upper bound on how many patients to download.

    Returns
    -------
    list[str]
        List of successfully downloaded record names.
    """
    user, pwd = _credentials()
    os.environ["WFDB_PHYSIONET_USER"] = user
    os.environ["WFDB_PHYSIONET_PASS"] = pwd

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pn_dir = f"{MIMIC_DB}/{MIMIC_VERSION}"

    logger.info("Fetching MIMIC-III record list …")
    try:
        all_records = wfdb.get_record_list(pn_dir)
    except Exception as exc:
        logger.error("Cannot list MIMIC-III records: %s", exc)
        return []

    logger.info("Total MIMIC-III records available: %d", len(all_records))

    # Filter to records likely to have ABP (channel name "ABP" or "ART")
    abp_records: list[str] = []
    for rec in all_records:
        if len(abp_records) >= max_patients:
            break
        try:
            header = wfdb.rdheader(rec, pn_dir=pn_dir)
            sig_names = [s.upper() for s in header.sig_name]
            if "ABP" in sig_names or "ART" in sig_names:
                abp_records.append(rec)
        except Exception:
            continue

    logger.info("Found %d MIMIC records with ABP signal.", len(abp_records))
    downloaded: list[str] = []

    for rec in abp_records:
        dest = Path(out_dir) / rec
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.with_suffix(".hea").exists():
            downloaded.append(rec)
            continue
        ok = _download_with_retry(rec, str(dest.parent), f"{pn_dir}/{Path(rec).parent}", dest)
        if ok:
            downloaded.append(rec)
            logger.info("✓ MIMIC %s", rec)

    logger.info("MIMIC download complete: %d/%d records.", len(downloaded), len(abp_records))
    return downloaded


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    charis_records = download_charis()
    mimic_records  = download_mimic()
    print(f"\nDownloaded CHARIS: {len(charis_records)} records")
    print(f"Downloaded MIMIC : {len(mimic_records)} records")
