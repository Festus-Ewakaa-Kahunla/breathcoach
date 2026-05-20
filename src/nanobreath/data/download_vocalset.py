#!/usr/bin/env python3
"""Download and verify VocalSet from Zenodo.

`VocalSet <https://zenodo.org/records/1193957>`_ is a singing voice dataset of
20 professional singers (~10.1 hours, English vowels and exercises) released
under CC BY 4.0. It ships with 17 clip-level technique labels (vibrato, belt,
breathy, lip trill, inhaled, etc.).

Notes for this project:

- The "inhaled" technique label is *singing on the inhale*, NOT a breath-event
  label.
- We hand-label our own breath events on top of the VocalSet audio.
- We ignore VocalSet's clip-level technique labels for the breath task.

Usage::

    python -m nanobreath.data.download_vocalset [--data-dir <dir>]

By default ``--data-dir`` resolves to ``<repo-root>/data/vocalset``, overridable
via the ``NANOBREATH_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from nanobreath.config import vocalset_dir


VOCALSET_URL = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"
VOCALSET_FILENAME = "VocalSet.zip"
VOCALSET_SIZE_MB_APPROX = 1980


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and verify VocalSet")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=vocalset_dir(),
        help=f"target directory (default: {vocalset_dir()})",
    )
    p.add_argument("--keep-zip", action="store_true",
                   help="keep the zip file after extraction (default: delete)")
    p.add_argument("--no-extract", action="store_true",
                   help="download only, do not extract")
    return p.parse_args()


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def download(url: str, target: Path) -> None:
    """Download with curl so we get a progress bar and resume support."""
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "-C", "-", "-o", str(target), url]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def verify_size(zip_path: Path) -> bool:
    actual = _file_size_mb(zip_path)
    delta = abs(actual - VOCALSET_SIZE_MB_APPROX)
    if delta > 50:
        print(f"WARNING: file size {actual:.1f} MB differs from expected "
              f"~{VOCALSET_SIZE_MB_APPROX} MB by {delta:.1f} MB")
        return False
    print(f"Size OK: {actual:.1f} MB (expected ~{VOCALSET_SIZE_MB_APPROX} MB)")
    return True


def extract(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        print(f"  {len(members)} entries in zip")
        z.extractall(dest_dir)
    print("Extraction complete.")


def summarize(dest_dir: Path) -> None:
    """Print a quick summary of what was extracted."""
    wavs = list(dest_dir.rglob("*.wav"))
    if not wavs:
        print(f"NOTE: no WAV files found under {dest_dir}")
        return
    total_size_mb = sum(w.stat().st_size for w in wavs) / (1024 * 1024)
    singers = sorted({w.parts[-2] if len(w.parts) >= 2 else "?" for w in wavs})
    print()
    print(f"Found {len(wavs)} WAV files, total {total_size_mb:.1f} MB")
    print(f"Distinct singer dirs: {len(singers)}")
    print("Sample paths:")
    for w in wavs[:5]:
        print(f"  {w.relative_to(dest_dir)}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_dir / VOCALSET_FILENAME

    if zip_path.exists():
        print(f"Zip already exists at {zip_path} ({_file_size_mb(zip_path):.1f} MB)")
    else:
        print(f"Downloading VocalSet to {zip_path}...")
        download(VOCALSET_URL, zip_path)

    if not verify_size(zip_path):
        print("Aborting because download may be incomplete.")
        sys.exit(1)

    if not args.no_extract:
        extract(zip_path, args.data_dir)
        summarize(args.data_dir)

    if not args.keep_zip and not args.no_extract:
        print(f"Removing zip {zip_path} (use --keep-zip to retain)")
        zip_path.unlink()


if __name__ == "__main__":
    main()
