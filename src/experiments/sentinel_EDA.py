import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

from configs.paths import DATA_DIR
from src.configs.label_mappings import MAPS
from src.data.download_s2_osm_data import AOIs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="multiclass", help=f"one of {list(MAPS)}. Default: multiclass")
    parser.add_argument("--aoi", type=str, default="at", help=f"one of {list(AOIs)}")
    args = parser.parse_args()

    sentinel_dir: Path = DATA_DIR / args.aoi / args.labels / "sentinel"
    print("Data path: ", sentinel_dir)
    sentinel_files = list(sentinel_dir.glob("*.tif"))
    print("Number of files: ", len(sentinel_files))

    zero_percentages = []
    for file in sentinel_files:
        with rasterio.open(file) as src:
            sentinel_data = src.read(1)
            total_pixels = sentinel_data.size  # Total number of pixels in the image
            zero_count = (sentinel_data == 0).sum()
            zero_percentage = (zero_count / total_pixels) * 100  # Calculate percentage of zero pixels
            zero_percentages.append(zero_percentage)

    zero_percentages_info = {file.name: percent for file, percent in zip(sentinel_files, zero_percentages) if
                             percent > 0}
    print("Files with percentage of zero counts: ", zero_percentages_info)

    plt.figure(figsize=(10, 6))
    plt.hist(zero_percentages, bins=30, color='skyblue', edgecolor='black')  # Plot percentages instead of counts
    plt.title('Distribution of Zero Pixel Percentages Across Sentinel Files')
    plt.xlabel('Percentage of Zero Pixels')
    plt.ylabel('Number of Files')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
