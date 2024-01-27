from pathlib import Path

import numpy as np
import rasterio
from einops import einops
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from data.download_data import BBox
from src.configs.label_mappings import LabelMap, GENERAL_MAP
import numpy.typing as npt

from src.configs.paths import SENTINEL_DIR, OSM_DIR


def plot_sentinel_and_mask(sentinel: Path, mask: Path, label_map: LabelMap, p: int) -> None:
    sentinel_img, sentinel_bbox = load_senintel_tiff_for_plotting(sentinel, return_bbox=True)
    mask_img = load_mask_tiff_for_plotting(mask)
    cmap = get_color_map(label_map)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mask_img, cmap=cmap)
    ax[0].set_title("Mask")
    ax[0].axis("off")

    ax[1].imshow(sentinel_img)
    ax[1].set_title("Sentinel Image")
    ax[1].axis("off")

    fig.suptitle(f"BBOX: {sentinel_bbox.__str__(p=p)}", fontsize=14, y=0.95)


def load_senintel_tiff_for_plotting(
    file: Path, scale_percentile_threshold: float = 98, return_bbox: bool = False
) -> npt.NDArray | tuple[npt.NDArray, BBox]:
    """Loads and scales a sentinel tiff image for plotting
    Args:
        file (Path): Path to the sentinel tiff image
        scale_percentile_threshold (float): Percentile threshold for scaling the image
        return_bbox (bool): Whether to return the bounding box information

    Returns:
        npt.NDArray: Scaled sentinel tiff image. Shape: (height, width, 3)
    """
    with rasterio.open(file) as f:
        image = f.read([3, 2, 1])  # B3=B, B2=G, B3=R
        bbox = f.bounds
    p2, p98 = np.percentile(image, q=(100 - scale_percentile_threshold, scale_percentile_threshold))
    image = np.clip((image - p2) / (p98 - p2) * 255, a_min=0, a_max=255).astype(np.uint8)
    image = einops.rearrange(image, "rgb h w -> h w rgb")

    if return_bbox:
        return image, BBox(west=bbox.left, south=bbox.bottom, east=bbox.right, north=bbox.top)
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


def _example(n: int, p: int) -> None:
    s = SENTINEL_DIR / f"{n}.tif"
    o = OSM_DIR / f"{n}.tif"
    plot_sentinel_and_mask(sentinel=s, mask=o, label_map=GENERAL_MAP, p=p)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="sentinel image index of the downloaded data")
    parser.add_argument("--p", type=int, default=6, help="precision for displaying bbox coordinates")
    args = parser.parse_args()

    _example(n=args.n, p=args.p)
