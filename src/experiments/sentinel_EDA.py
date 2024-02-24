import argparse
from pathlib import Path

import numpy as np
import rasterio

from configs.paths import DATA_DIR
from src.configs.label_mappings import MAPS
from src.data.download_data import AOIs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="multiclass", help=f"one of {list(MAPS)}. Default: multiclass")
    parser.add_argument("--aoi", type=str, default="at", help=f"one of {list(AOIs)}")
    args = parser.parse_args()

    sentinel_dir: Path = DATA_DIR / args.aoi / args.labels / "sentinel"
    print("Data path: ", sentinel_dir)
    sentinel_files = list(sentinel_dir.glob("*.tif"))
    print("Number of files: ", len(sentinel_files))
    nan_counts = {}
    zero_counts = {}
    for file in sentinel_files:
        with rasterio.open(file) as src:
            sentinel_data = src.read(1)
            nan_counts[file.name] = np.isnan(sentinel_data).sum()
            zero_counts[file.name] = (sentinel_data == 0).sum()
    print("Files with NaN counts: ", {k: v for k, v in nan_counts.items() if v > 0})
    print("Files with zero counts: ", {k: v for k, v in zero_counts.items() if v > 0})


if __name__ == '__main__':
    main()
