from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "nsl_kdd_sample": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
}


def download_dataset(name: str, output_dir: Path) -> Path:
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{name}.csv"
    urlretrieve(DATASETS[name], destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    path = download_dataset(args.name, Path(args.output_dir))
    print(f"Downloaded to {path}")
