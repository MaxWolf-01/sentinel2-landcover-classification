import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import torch
from einops import einops
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from data.download_s2_osm_data import AOIs, BBox, DataDirs
from data.s2osm_dataset import get_mask_file_idx
from src.configs.label_mappings import LabelMap, MAPS


def plot_sentinel_and_mask(sentinel: Path, mask: Path, label_map: LabelMap, p: int | None = None) -> None:
    """Plots Sentinel image and mask side by side."""
    sentinel_img, sentinel_bbox = load_sentinel_tiff_for_plotting(sentinel, return_bbox=True)
    mask_img = load_mask_tiff_for_plotting(mask)
    plot_images([mask_img, sentinel_img], ["Mask", "Sentinel Image"], sentinel_bbox, label_map, p)


def plot_sentinel_mask_and_pred(
    sentinel: Path, mask: Path, pred_img: npt.NDArray, label_map: LabelMap, p: int | None = None
) -> None:
    """Plots Sentinel image, mask, and prediction side by side."""
    sentinel_img, sentinel_bbox = load_sentinel_tiff_for_plotting(sentinel, return_bbox=True)
    mask_img = load_mask_tiff_for_plotting(mask)
    plot_images(
        [mask_img, pred_img, sentinel_img], ["Mask", "Prediction", "Sentinel Image"], sentinel_bbox, label_map, p
    )


def plot_images(
    images: list[npt.NDArray],
    titles: list[str],
    bbox: BBox | None = None,
    label_map: LabelMap | None = None,
    p: int | None = None,
) -> None:
    """Plots a list of images with their respective titles.
    Args:
        images (List[npt.NDArray]): List of images to be plotted.
        titles (List[str]): Titles for the images.
        bbox (Optional[BBox]): Bounding box information.
        label_map (Optional[ListedColormap]): Color map for images.
        p (Optional[int]): Precision for displaying bbox coordinates.
    """
    cmap = get_color_map(label_map) if label_map else None
    num_images = len(images)
    fig, ax = plt.subplots(1, num_images, figsize=(6 * num_images, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(img, cmap=cmap)
        ax[i].set_title(title)
        ax[i].axis("off")
    if label_map:
        legend_elements = [Patch(facecolor=label_map[label]["color"], label=label) for label in label_map]
        plt.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(-0.15, -0.05), ncol=len(label_map))
    if bbox:
        fig.suptitle(f"BBOX: {bbox.__str__(p=p)}", fontsize=14, y=0.95)


def load_sentinel_tiff_for_plotting(
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


def load_pred_batch_for_plotting(file: Path) -> list[npt.NDArray]:
    return [pred for pred in torch.load(file).argmax(dim=1).cpu().numpy()]  # (B, H, W)


def get_color_map(label_map: LabelMap) -> ListedColormap:
    """Creates a color map for the given label map
    Args:
        label_map (LabelMap): Label map

    Returns:
        ListedColormap: Color map
    """
    num_classes = len(label_map)
    colors = ["" for _ in range(num_classes)]
    for i, label in enumerate(label_map.values()):
        colors[i] = label["color"]
    cmap = ListedColormap(colors)
    return cmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="at", help=f"Default: VIE. Available: {list(AOIs)}")
    parser.add_argument("--labels", type=str, default="multiclass", help=f"Default: Multiclass. Available:{list(MAPS)}")
    parser.add_argument("--n", type=int, default=0, help="sentinel image index of the downloaded data")
    parser.add_argument("--p", type=int, default=6, help="precision for displaying bbox coordinates")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="n is the index of the mask file instead of the sentinel file."
        "Uses the first sentinel file found for the mask file.",
    )
    args = parser.parse_args()
    data_dirs = DataDirs(aoi=args.aoi, map_type=args.labels)
    sentinel_files: dict[int, Path] = data_dirs.sentinel_files
    mask_files: dict[int, Path] = data_dirs.osm_files
    get_sentinel_file_from_mask = (  # noqa: E731
        lambda osm_i: next(f for f in sentinel_files.values() if get_mask_file_idx(f) == int(mask_files[osm_i].stem))
    )

    index = args.n
    max_index = len(sentinel_files)
    print("Press...\nn: next\nb: back\nN: next and close\nB: back and close\n{index} to jump\nq: quit")
    while True:
        index = max(0, min(index, max_index - 1))
        if not args.reverse:
            sentinel_file = sentinel_files[index]
            osm_file = mask_files[get_mask_file_idx(sentinel_file)]
        else:
            sentinel_file = get_sentinel_file_from_mask(index)
            osm_file = mask_files[get_mask_file_idx(sentinel_file)]
            assert int(osm_file.stem) == index, f"{osm_file.stem} != {index}"
        assert sentinel_file.stem.split("_")[0] == osm_file.stem, f"{sentinel_file.stem} != {osm_file.stem}"
        print("Showing:", "/".join(sentinel_file.parts[-4:]), "/".join(osm_file.parts[-4:]))
        plot_sentinel_and_mask(sentinel=sentinel_file, mask=osm_file, label_map=MAPS[args.labels], p=args.p)
        plt.show(block=False)
        user_input = input("Input:")
        if user_input in "nN":
            index += 1
        elif user_input in "bB":
            index -= 1
        elif user_input == "q":
            plt.close("all")
            break
        elif user_input.isnumeric():
            index = int(user_input)
        else:
            continue
        if user_input.isupper():
            plt.close("all")
