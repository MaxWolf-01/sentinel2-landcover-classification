import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import rasterio

from src.configs.label_mappings import LabelMap, MAPS
from src.configs.paths import DATA_DIR
from src.data.download_data import AOIs
import argparse


def calculate_osm_label_distribution_and_percentage(osm_dir: Path) -> tuple[dict[int, int], dict[int, float]]:
    """
    Calculate the distribution of labels and their percentages in OSM data stored as GeoTIFF files in a given directory.

    Args:
    - osm_dir (str): The directory where the OSM GeoTIFF files are stored.

    Returns:
    - Tuple[Dict[int, int], Dict[int, float]]: A tuple containing two dictionaries:
        - The first dictionary with label values as keys and their counts as values.
        - The second dictionary with label values as keys and their percentage of the total counts as values.
    """
    label_distribution = {}
    total_count = 0
    for file in osm_dir.glob("*.tif"):
        with rasterio.open(file) as src:
            osm_data = src.read(1)
            unique, counts = np.unique(osm_data, return_counts=True)
            for label, count in zip(unique, counts):
                label_distribution[label] = label_distribution.get(label, 0) + count
                total_count += count  # Add to total count

    label_percentages = {label: (count / total_count) * 100 for label, count in label_distribution.items()}

    return label_distribution, label_percentages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="multiclass", help=f"one of {list(MAPS)}. Default: multiclass")
    parser.add_argument("--aoi", type=str, default="at", help=f"one of {list(AOIs)}")
    args = parser.parse_args()
    osm_dir = DATA_DIR / args.aoi / args.labels / "osm"
    print("Data path: ", osm_dir)
    label_counts, label_percentages = calculate_osm_label_distribution_and_percentage(osm_dir)
    label_map: LabelMap = MAPS[args.labels]
    label_counts: dict[str, float] = {name: value for name, value in zip(label_map, label_counts.values())}
    label_percentages = {name: value for name, value in zip(label_map, label_percentages.values())}

    print("Label counts: ", label_counts)
    print("Label distribution in %: ", label_percentages)
