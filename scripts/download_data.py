"""
Automated dataset download script for NIDS ML project.

Supported datasets:
  - NSL-KDD   (from the University of New Brunswick mirror)
  - UNSW-NB15 (from the UNSW Research Data portal)

Usage:
    python scripts/download_data.py --dataset nsl-kdd
    python scripts/download_data.py --dataset unsw-nb15
    python scripts/download_data.py --dataset all
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests

# ─── Dataset registry ────────────────────────────────────────────────────────
# Each entry: { url, filename, dest_dir, extract }
DATASETS = {
    "nsl-kdd": [
        {
            "name": "KDDTrain+.txt",
            "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "dest": "data/raw",
            "extract": False,
        },
        {
            "name": "KDDTest+.txt",
            "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
            "dest": "data/raw",
            "extract": False,
        },
    ],
    "unsw-nb15": [
        {
            "name": "UNSW_NB15_training-set.csv",
            "url": (
                "https://research.unsw.edu.au/sites/default/files/documents/"
                "UNSW_NB15_training-set.csv"
            ),
            "dest": "data/raw/UNSW-NB15",
            "extract": False,
        },
        {
            "name": "UNSW_NB15_testing-set.csv",
            "url": (
                "https://research.unsw.edu.au/sites/default/files/documents/"
                "UNSW_NB15_testing-set.csv"
            ),
            "dest": "data/raw/UNSW-NB15",
            "extract": False,
        },
    ],
}

# Root of the project (one level above scripts/)
PROJECT_ROOT = Path(__file__).parent.parent


def _download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Stream-download a file with a progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  → {dest_path}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    print(f"\r  ✓ Saved {downloaded / 1024 / 1024:.2f} MB")


def download_dataset(name: str) -> None:
    """Download all files for the given dataset key."""
    if name not in DATASETS:
        print(f"[ERROR] Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'='*60}")

    for entry in DATASETS[name]:
        dest_path = PROJECT_ROOT / entry["dest"] / entry["name"]

        if dest_path.exists():
            print(f"  [SKIP] Already exists: {dest_path.name}")
            continue

        try:
            _download_file(entry["url"], dest_path)
        except requests.exceptions.HTTPError as e:
            print(f"\n  [WARNING] Download failed ({e}). ")
            print(
                f"  For UNSW-NB15, please visit https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
                f"  and download the CSV files manually to: {dest_path.parent}"
            )
        except requests.exceptions.ConnectionError:
            print(f"\n  [ERROR] Could not connect. Check your internet connection.")

    print(f"\n  Dataset '{name}' ready.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download NIDS benchmark datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the root data directory (default: <project_root>/data/raw)",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        download_dataset(ds)

    print("All downloads complete.")


if __name__ == "__main__":
    main()
