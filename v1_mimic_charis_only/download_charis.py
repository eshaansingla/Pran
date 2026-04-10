"""
download_charis.py  --  Download the CHARIS ICP database from PhysioNet.

Open-access dataset, no credentials required.
URL: https://physionet.org/files/charisdb/1.0.0/

Records: charis1 ... charis13  (13 TBI patients with ICP + ABP waveforms)
Size:    ~1.76 GB total

Usage:
    python download_charis.py
    python download_charis.py --out_dir data/raw/charis
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL  = "https://physionet.org/files/charisdb/1.0.0/"
N_RECORDS = 13   # charis1 .. charis13

EXTRA_FILES = ["RECORDS", "SHA256SUMS.txt"]


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> bool:
    """
    Download a single file with resume support and a tqdm progress bar.

    Returns True on success.
    """
    # Resume: check existing size
    existing = dest.stat().st_size if dest.exists() else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as exc:
        print(f"\n  ERROR connecting to {url}: {exc}", file=sys.stderr)
        return False

    if resp.status_code == 416:          # range not satisfiable = file complete
        return True
    if resp.status_code not in (200, 206):
        print(f"\n  HTTP {resp.status_code} for {url}", file=sys.stderr)
        return False

    total = existing + int(resp.headers.get("Content-Length", 0))
    mode  = "ab" if existing else "wb"

    with open(dest, mode) as fh, tqdm(
        total=total,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        leave=False,
        ncols=80,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    return True


def download_charis(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = EXTRA_FILES[:]
    for i in range(1, N_RECORDS + 1):
        files.append(f"charis{i}.hea")
        files.append(f"charis{i}.dat")

    print(f"\nDownloading CHARIS database ({len(files)} files) to {out_dir}/")
    print(f"Source: {BASE_URL}")
    print(f"Estimated total size: ~1.76 GB\n")

    failed = []
    for fname in files:
        dest = out_dir / fname
        url  = BASE_URL + fname

        if dest.exists() and fname.endswith(".dat") and dest.stat().st_size > 1_000_000:
            tqdm.write(f"  [skip] {fname}  (already exists, {dest.stat().st_size//1024//1024} MB)")
            continue

        tqdm.write(f"  Downloading {fname} ...")
        ok = False
        for attempt in range(1, 4):
            ok = _download_file(url, dest)
            if ok:
                break
            wait = 5 * attempt
            tqdm.write(f"    Retry {attempt}/3 in {wait}s ...")
            time.sleep(wait)

        if ok:
            size_mb = dest.stat().st_size / 1024 / 1024
            tqdm.write(f"  [done] {fname}  ({size_mb:.1f} MB)")
        else:
            tqdm.write(f"  [FAIL] {fname}")
            failed.append(fname)

    print(f"\n{'='*55}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        print("Re-run the script to retry — downloads resume automatically.")
    else:
        print(f"All {len(files)} files downloaded to {out_dir}/")
        # List .hea files so the next step can find them
        hea_files = sorted(out_dir.glob("*.hea"))
        print(f"Records available: {len(hea_files)}")
        for h in hea_files:
            print(f"  {h.stem}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download CHARIS ICP database")
    p.add_argument("--out_dir", type=Path, default=Path("data/raw/charis"))
    args = p.parse_args()
    download_charis(args.out_dir)
