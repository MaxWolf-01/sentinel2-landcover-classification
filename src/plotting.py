from pathlib import Path

import numpy as np
import rasterio
from einops import einops
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from src.configs.label_mappings import LabelMap, GENERAL_MAP
import numpy.typing as npt

from src.configs.paths import SENTINEL_DIR, OSM_DIR


def plot_sentinel_and_mask(sentinel: Path, mask: Path, label_map: LabelMap) -> None:
    sentinel_img = load_senintel_tiff_for_plotting(sentinel)
    mask_img = load_mask_tiff_for_plotting(mask)
    cmap = get_color_map(label_map)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mask_img, cmap=cmap)
    ax[0].set_title("Mask")
    ax[0].axis("off")

    ax[1].imshow(sentinel_img)
    ax[1].set_title("Sentinel Image")
    ax[1].axis("off")


def load_senintel_tiff_for_plotting(file: Path, scale_percentile_threshold: float = 99.95) -> npt.NDArray:
    """Loads and scales a sentinel tiff image for plotting
    Args:
        file (Path): Path to the sentinel tiff image
        scale_percentile_threshold (float): Percentile threshold for scaling the image

    Returns:
        npt.NDArray: Scaled sentinel tiff image. Shape: (height, width, 3)
    """
    with rasterio.open(file) as f:
        image = f.read([3, 2, 1])  # B3=B, B2=G, B3=R
    image = image.astype(float)
    max_val = np.percentile(image, q=scale_percentile_threshold)
    image = (image / max_val) * 255
    image[image > 255] = 255
    image = image.astype(np.uint8)
    image = einops.rearrange(image, "c h w -> h w c")
    return image


def load_mask_tiff_for_plotting(file: Path) -> npt.NDArray:
    """Loads and scales a mask tiff image for plotting
    Args:
        file (Path): Path to the mask tiff image

    Returns:
        npt.NDArray: Mask tiff image. Shape: (height, width)
    """
    with rasterio.open(file) as f:
        image = f.read(1)
    return image


def get_color_map(label_map: LabelMap) -> ListedColormap:
    """Creates a color map for the given label map
    Args:
        label_map (LabelMap): Label map

    Returns:
        ListedColormap: Color map
    """
    num_classes = len(label_map)
    colors = ["" for _ in range(num_classes)]
    for label in label_map.values():
        colors[label["idx"]] = label["color"]
    cmap = ListedColormap(colors)
    return cmap


def _example() -> None:
    n = 0
    s = SENTINEL_DIR / f"{n}.tif"
    o = OSM_DIR / f"{n}.tif"
    plot_sentinel_and_mask(sentinel=s, mask=o, label_map=GENERAL_MAP)
    plt.show()


if __name__ == "__main__":
    _example()
