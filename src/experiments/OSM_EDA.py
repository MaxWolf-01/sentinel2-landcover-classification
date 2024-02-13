import os
from typing import Dict, Tuple
import numpy as np
import rasterio


def calculate_osm_label_distribution_and_percentage(osm_dir: str) -> Tuple[Dict[int, int], Dict[int, float]]:
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

    for filename in os.listdir(osm_dir):
        if filename.endswith(".tif"):
            file_path = os.path.join(osm_dir, filename)
            with rasterio.open(file_path) as src:
                osm_data = src.read(1)

                unique, counts = np.unique(osm_data, return_counts=True)
                for label, count in zip(unique, counts):
                    if label in label_distribution:
                        label_distribution[label] += count
                    else:
                        label_distribution[label] = count
                    total_count += count  # Add to total count

    label_percentages = {label: (count / total_count) * 100 for label, count in label_distribution.items()}

    return label_distribution, label_percentages


osm_dir = '../../data/test/multiclass/osm'
label_distribution, label_percentages = calculate_osm_label_distribution_and_percentage(osm_dir)
print("Label distribution: ", label_distribution)
print("Label distribution in %: ", label_percentages)
