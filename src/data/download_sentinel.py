"""
Multiple sentinel images are fetched as segments of a larger AOI over some time-span.
How many images are downloaded is determined by the frequency argument. E.g. "W" means, that for the given time-interval
of one week, a request for one image within that week is sent. For maxcc=0 (cloud-free), it makes sense to use a
lower frequency, as the chance of getting a cloud-free image is higher (and we are wasting less requests).
2MS/QS should be reasonable frequencies (every two months, quarterly).
The sentinel images are saved as data/<aoi>/sentinel/<segment-id>_<time-idx>.tif
"""
import argparse
import concurrent
import json
import math
import os
import shutil
import time
import warnings
from functools import partial
from pathlib import Path

import einops
import geopandas as gpd
import numpy.typing as npt
import pandas as pd
import rasterio
import sentinelhub as sh
from dotenv import load_dotenv
from geopy.distance import geodesic
from tqdm import tqdm

from src.configs.data_config import (
    AOIs,
    BANDS,
    BBox,
    CRS,
    DATA_COLLECTION,
    DataDirs,
    MAX_CLOUD_COVER,
    SEGMENT_LENGTH_KM,
    SEGMENT_SIZE,
    SENTINEL2_EVALSCRIPT,
    TIME_INTERVAL,
)
from src.configs.paths import LOG_DIR

warnings.filterwarnings("error", category=sh.download.sentinelhub_client.SHRateLimitWarning)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aoi", type=str, default="vie", help=f"Specify an AOI. Default: VIE. Available:{list(AOIs)}")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to use for downloading.")
    parser.add_argument(
        "--frequency",
        type=str,
        default="QS",
        help="Pandas time frequency string. Determines how many images to fetch for a given segment. Default: QS.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing sentinel data.")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded segments. Don't overwrite.")
    args = parser.parse_args()

    time_intervals: list[tuple[str, str]] = _split_up_time_interval(TIME_INTERVAL, frequency=args.frequency)
    assert len(time_intervals) > 0, "No time intervals found. Check your time interval and frequency settings."

    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    segments: list[BBox] = calculate_segments(bbox=AOIs[args.aoi], segment_size_km=SEGMENT_LENGTH_KM)
    print(f"Number of segments: {len(segments)}")

    data_dirs = DataDirs(aoi=args.aoi, map_type="")
    if args.overwrite and data_dirs.sentinel.exists() and not args.resume:
        warnings.warn(f"Deleting existing sentinel data: {data_dirs.sentinel}")
        input("Press Enter to continue...")
        shutil.rmtree(data_dirs.sentinel)
    data_dirs.sentinel.mkdir(parents=True, exist_ok=True)

    resume_file = data_dirs.base_path / "resume.json"
    resume_metadata_file = data_dirs.base_path / "metadata.tmp.json"
    skip_indices: list[int] = (
        load_resume_data(
            resume_file=resume_file,
            metadata_file=resume_metadata_file,
            current_metadata=_metadata_dict(args, segments),
            segments=segments,
        )
        if args.resume
        else []
    )

    process_segment = partial(
        _process_segment, time_intervals=time_intervals, sh_config=config, path=data_dirs.sentinel
    )

    log_file = LOG_DIR / "download.log"
    log_file.unlink(missing_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_segment, i, segment): (i, segment)
            for i, segment in enumerate(segments)
            if i not in skip_indices
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, segment = futures[future]
            try:
                future.result()
                skip_indices.append(idx)
                save_resume_data(resume_file, skip_indices)
            except Exception as e:
                msg = f"Error in segment {idx}: {e}\n"
                print(msg)
                with log_file.open("a") as f:
                    f.write(msg)
                raise e
    print(f"Collected {len(data_dirs.sentinel_files)} sentinel images.")
    with (data_dirs.base_path / "metadata.json").open("w") as f:  # write settings to file for reproducibility/info
        json.dump(_metadata_dict(args, segments), f, indent=4)
    resume_file.unlink(missing_ok=True)
    resume_metadata_file.unlink(missing_ok=True)


def load_resume_data(resume_file: Path, metadata_file: Path, current_metadata: dict, segments: list) -> list[int]:
    """Loads the resume file and checks if the metadata matches the current settings. If it does, it returns the
    indices of segments that have already been downloaded."""
    skip_indices: list[int] = []
    if resume_file.exists():
        with resume_file.open("r") as f:
            skip_indices = json.load(f).get("skip_indices", [])
        with metadata_file.open("r") as f:
            resume_metadata: dict = json.load(f)
        assert resume_metadata == current_metadata, (
            f"Metadata mismatch. Please check your settings.\n"
            f"Tried to resume script with: {current_metadata}\nPrevious script was run with: {resume_metadata}"
        )
        print(
            f"Skipping {len(skip_indices)} already downloaded segments. Remaining: {len(segments) - len(skip_indices)}"
        )
    with metadata_file.open("w") as f:
        json.dump(current_metadata, f, indent=4)
    return skip_indices


def save_resume_data(resume_file: Path, skip_indices: list[int]) -> None:
    with resume_file.open("w") as f:
        json.dump({"skip_indices": skip_indices}, f, indent=4)


def _metadata_dict(args: argparse.Namespace, segments: list[BBox]) -> dict:
    return {
        "aoi": args.aoi,
        "bands": BANDS,
        "collection": DATA_COLLECTION.name,
        "crs": CRS.value,
        "frequency": args.frequency,
        "interval": list(TIME_INTERVAL),
        "mapping": args.labels,
        "max_cloud_cover": MAX_CLOUD_COVER,
        "num_segments": len(segments),
        "resolution": list(SEGMENT_SIZE),
        "segment_length_km": SEGMENT_LENGTH_KM,
    }


def _process_segment(
    idx: int,
    segment: BBox,
    time_intervals: list[tuple[str, str]],
    sh_config: sh.SHConfig,
    path: Path,
) -> None:
    sentinel_data: list[npt.NDArray] = []
    for time_interval in time_intervals:
        data = fetch_sentinel_data(segment=segment, sh_config=sh_config, time_interval=time_interval)
        if (data == 0).sum() > 0.5 * data.size:  # if more than 50% is 0, it is usually cut off
            continue
        sentinel_data.append(data)
        time.sleep(2)  # to avoid rate limiting
    save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=segment, sentinel_dir=path)


def _split_up_time_interval(time_interval: tuple[str, str], frequency: str) -> list[tuple[str, str]]:
    """Returns a list of time intervals, split up into smaller time-intervals with the given frequency.
    Args:
        time_interval (tuple[str, str]): Start and end date of the overarching time interval.
        frequency (str): e.g. "W" for weekly, "MS" for month-start frequency, "D" for daily. See pandas docs for more:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    """
    split_up_intervals = []
    date_ranges = pd.date_range(start=time_interval[0], end=time_interval[1], freq=frequency)
    for start, end in zip(date_ranges, date_ranges[1:]):
        split_up_intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return split_up_intervals


def calculate_segments(bbox: BBox, segment_size_km: float) -> list[BBox]:
    """Splits the provided bounding box (AOI) into smaller segments.
    Each segment is approximately `segment_size_km` kilometers wide and tall, although the actual
    segment size in degrees will vary depending on geographic location due to Earth's curvature.
    Args:
        bbox (BBox): Bounding box to split.
        segment_size_km (float): The desired width and height of each segment in kilometers.
    Returns:
        list[BBox]: List of smaller bounding boxes.
    """
    km_width: float = geodesic((bbox.north, bbox.west), (bbox.north, bbox.east)).kilometers
    km_height: float = geodesic((bbox.north, bbox.west), (bbox.south, bbox.west)).kilometers
    print(f"AOI: {bbox}\nWidth: {km_width:.2f}km, Height: {km_height:.2f}km")

    num_lon_segments: int = int(math.ceil(km_width / segment_size_km))
    num_lat_segments: int = int(math.ceil(km_height / segment_size_km))

    lon_increment: float = (bbox.east - bbox.west) / num_lon_segments
    lat_increment: float = (bbox.north - bbox.south) / num_lat_segments

    segments: list[BBox] = []
    for i in range(num_lon_segments):
        for j in range(num_lat_segments):
            segment_west = bbox.west + i * lon_increment
            segment_east = segment_west + lon_increment
            segment_south = bbox.south + j * lat_increment
            segment_north = segment_south + lat_increment
            segments.append(BBox(north=segment_north, south=segment_south, east=segment_east, west=segment_west))
    return segments


def fetch_sentinel_data(segment: BBox, sh_config: sh.SHConfig, time_interval: tuple[str, str]) -> npt.NDArray:
    request = sh.SentinelHubRequest(
        evalscript=SENTINEL2_EVALSCRIPT,
        input_data=[
            sh.SentinelHubRequest.input_data(
                data_collection=DATA_COLLECTION,
                time_interval=time_interval,
                maxcc=MAX_CLOUD_COVER,
                mosaicking_order=sh.MosaickingOrder.LEAST_CC,
                upsampling=sh.ResamplingType.BICUBIC,
            )
        ],
        responses=[sh.SentinelHubRequest.output_response("default", sh.MimeType.TIFF)],
        bbox=sh.BBox((segment.west, segment.south, segment.east, segment.north), crs=CRS),
        size=SEGMENT_SIZE,
        config=sh_config,
        data_folder="../data/sentinel/",
    )
    return request.get_data(save_data=False)[0]  # always returns 1-element list


def save_sentinel_data_as_geotiff(data: list[npt.NDArray], idx: int, aoi: BBox, sentinel_dir: Path) -> None:
    pixel_size_x, pixel_size_y = calculate_pixel_size(aoi, SEGMENT_SIZE)
    for i, image in enumerate(data):
        image = einops.rearrange(image, "h w c -> c h w")
        with rasterio.open(
            fp=sentinel_dir / f"{idx}_{i}.tif",
            mode="w",
            driver="GTiff",
            height=SEGMENT_SIZE[0],
            width=SEGMENT_SIZE[1],
            count=len(BANDS),
            dtype=image.dtype,
            crs=rasterio.crs.CRS().from_epsg(code=CRS.value),
            transform=rasterio.transform.from_origin(aoi.west, aoi.north, pixel_size_x, pixel_size_y),
        ) as f:
            f.write(image)


def calculate_pixel_size(aoi: BBox, resolution: tuple[int, int]) -> tuple[float, float]:
    pixel_size_x = (aoi.east - aoi.west) / resolution[0]
    pixel_size_y = (aoi.north - aoi.south) / resolution[1]
    return pixel_size_x, pixel_size_y


def _visualize_segment_bbox() -> None:
    import shapely
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    aoi = AOIs["at"]
    segments = calculate_segments(aoi, SEGMENT_LENGTH_KM)
    print(f"num segments: {len(segments)}")
    boxes = [shapely.geometry.box(minx=seg.west, maxx=seg.east, miny=seg.south, maxy=seg.north) for seg in segments]
    segments_gdf = gpd.GeoDataFrame(geometry=boxes)
    segments_gdf.set_crs(str(CRS), inplace=True)
    original_gdf = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(minx=aoi.west, maxx=aoi.east, miny=aoi.south, maxy=aoi.north)]
    )
    original_gdf.set_crs(str(CRS), inplace=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    original_gdf.boundary.plot(ax=ax, color="red", linewidth=2, label="Original AOI")
    segments_gdf.boundary.plot(ax=ax, edgecolor="blue", linewidth=1, alpha=0.5)
    segments_gdf.plot(ax=ax, color="blue", alpha=0.1)  # Fill color for segments
    red_patch = Patch(color="red", label="Original AOI")
    blue_patch = Patch(color="blue", alpha=0.5, label="Segments")
    ax.legend(handles=[red_patch, blue_patch])
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # _visualize_segment_bbox()
    main()
