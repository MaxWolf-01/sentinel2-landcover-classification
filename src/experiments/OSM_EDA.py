import argparse
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import pyplot as plt

from src.configs.label_mappings import LabelMap, MAPS
from src.configs.paths import DATA_DIR
from src.data.download_data import AOIs


def calculate_osm_label_distribution_and_percentage(osm_dir: Path) -> tuple[
    dict[int, int], dict[int, float], list[float]]:
    label_distribution = {}
    total_count = 0
    other_percentages = []  # List to store the percentage of 'other' for each file
    for file in list(osm_dir.glob("*.tif")):
        with rasterio.open(file) as src:
            osm_data = src.read(1)
            unique, counts = np.unique(osm_data, return_counts=True)
            file_total_count = np.sum(counts)
            file_other_count = counts[unique == 0][0] if 0 in unique else 0  # Get count of 'other' label
            other_percentage = (file_other_count / file_total_count) * 100  # Calculate percentage of 'other'
            if other_percentage > 5:
                print(file, other_percentage, file_total_count, file_other_count)
            other_percentages.append(other_percentage)  # Add to list
            for label, count in zip(unique, counts):
                label_distribution[label] = label_distribution.get(label, 0) + count
                total_count += count

    label_percentages = {label: (count / total_count) * 100 for label, count in label_distribution.items()}

    return label_distribution, label_percentages, other_percentages


def plot_other_distribution(other_percentages: list[float], step: float = 5.0):
    """
    Plots the distribution of files by the percentage of 'other' class.

    Args:
    - other_percentages (list[float]): List of percentages of 'other' in each file.
    - step (float): Step size for percentage bins.
    """
    bins = np.arange(0, 100 + step, step)  # Create bins from 0 to 100 with the specified step
    plt.hist(other_percentages, bins=bins, edgecolor='black')
    plt.xlabel('Percentage of "Other"')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Files by Percentage of "Other"')
    plt.xticks(bins)
    plt.grid(axis='y')
    plt.show()


def plot_detailed_other_distribution(other_percentages: list[float], step: float = 0.1):
    """
    Plots the distribution of files by the percentage of 'other' class in a more detailed range.

    Args:
    - other_percentages (list[float]): List of percentages of 'other' in each file.
    - step (float): Step size for percentage bins in the range 0 to 5.
    """
    # Focus on the range from 0 to 5% with a smaller step for higher precision
    bins = np.arange(0, 5 + step, step)  # Create bins from 0 to 5 with the specified step
    plt.hist([p for p in other_percentages if p <= 5], bins=bins, edgecolor='black')  # Only consider percentages ≤ 5%
    plt.xlabel('Percentage of "Other" (0 to 5%)')
    plt.ylabel('Number of Files')
    plt.title('Detailed Distribution of Files by Percentage of "Other" (0 to 5%)')
    plt.xticks(bins)
    plt.grid(axis='y')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="multiclass", help=f"one of {list(MAPS)}. Default: multiclass")
    parser.add_argument("--aoi", type=str, default="at", help=f"one of {list(AOIs)}")
    args = parser.parse_args()
    osm_dir = DATA_DIR / args.aoi / args.labels / "osm"
    print("Data path: ", osm_dir)
    label_counts, label_percentages, other_percentages = calculate_osm_label_distribution_and_percentage(osm_dir)
    label_map: LabelMap = MAPS[args.labels]
    label_counts: dict[str, float] = {name: value for name, value in zip(label_map, label_counts.values())}
    label_percentages = {name: value for name, value in zip(label_map, label_percentages.values())}
    plot_other_distribution(other_percentages)
    plot_detailed_other_distribution(other_percentages)  # New function call

    print("Label counts: ", label_counts)
    print("Label distribution in %: ", label_percentages)

    # Calculate and print the percentage and counts
    files_with_less_than_5_percent_other = [p for p in other_percentages if p < 5]
    percent_with_less_than_5_percent_other = (len(files_with_less_than_5_percent_other) / len(
        list(osm_dir.glob("*.tif"))
    )) * 100
    print(f"Percentage of data with less than 5% 'Other': {percent_with_less_than_5_percent_other:.2f}%")
    print(f"Count of files (before): {len(other_percentages)}")
    print(f"Count of files (after): {len(files_with_less_than_5_percent_other)}")
