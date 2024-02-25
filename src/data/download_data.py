"""
Multiple sentinel images are fetched over some time-span, with one corresponding OSM-label mask.
How many images are downloaded is determined by the frequency argument. E.g. "W" means, that for the given time-interval
of one week, a request for one image within that week is sent. For maxcc=0 (cloud-free), it makes sense to use a
higher frequency, as the chance of getting a cloud-free image is higher (and we are wasting less requests).
MS/2MS/QS seem to be the most useful frequencies (monthly, every two months, quarterly).
The sentinel images are saved as `./data/{parent_aoi: str}/{label_mapping: str}/{segment_aoi: int}_{time: int}.tif`
The corresponding osm masks as `./data/{parent_aoi: str}/{label_mapping: str}/{segment_aoi: int}.tif`
"""
import argparse
import concurrent
import json
import math
import os
import shutil
import time
import typing
import warnings
from pathlib import Path

import einops
import geopandas as gpd
import numpy.typing as npt
import osmnx as ox
import pandas as pd
import rasterio
import sentinelhub as sh
from dotenv import load_dotenv
from geopy.distance import geodesic
from rasterio.features import rasterize
from rasterio.transform import from_origin
from tqdm import tqdm

from src.configs.label_mappings import BINARY_MAP, LabelMap, MAPS, MULTICLASS_MAP, OSMTagMap
from src.configs.paths import DATA_DIR, ROOT_DIR

from src.configs.download_config import *



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="vie", help=f"Specify an AOI. Default: VIE. Available:{list(AOIs)}")
    parser.add_argument("--labels", type=str, default="multiclass", help="Specify a label mapping to use.")
    parser.add_argument("--workers", type=int, default=1, help="Specify the number of workers.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing.")
    parser.add_argument(
        "--frequency",
        type=str,
        default="QS",
        help="Pandas time frequency string. Determines how many images to fetch for a given segment. Default: QS.",
    )
    parser.add_argument("--overwrite-sentinel", action="store_true", help="re-download existing sentinel data.")
    parser.add_argument("--overwrite-osm", action="store_true", help="re-download existing OSM data.")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded segments. Don't overwrite.")
    args = parser.parse_args()
    assert args.overwrite_sentinel or args.overwrite_osm, "You need to set at least one overwrite flag to True."
    label_map = MAPS[args.labels]

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT_DIR / "osmnx_cache")

    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    data_dirs = DataDirs(aoi=args.aoi, map_type=args.labels)
    if args.overwrite_sentinel and data_dirs.sentinel.exists() and not args.resume:
        warnings.warn(f"Deleting existing sentinel data: {data_dirs.sentinel}")
        input("Press Enter to continue...")
        shutil.rmtree(data_dirs.sentinel)
    if args.overwrite_osm and data_dirs.osm.exists() and not args.resume:
        warnings.warn(f"Deleting existing osm data: {data_dirs.osm}")
        input("Press Enter to continue...")
        shutil.rmtree(data_dirs.osm, ignore_errors=True)
    data_dirs.osm.mkdir(parents=True, exist_ok=True)
    data_dirs.sentinel.mkdir(parents=True, exist_ok=True)

    segments: list[BBox] = calculate_segments(bbox=AOIs[args.aoi], segment_size_km=SEGMENT_LENGTH_KM)
    print(f"Number of segments: {len(segments)}")

    skip_indices: list[int] = []
    resume_indicies_file = data_dirs.base_path / "resume.json"
    resume_metadata_file = data_dirs.base_path / "metadata.tmp.json"
    if args.resume and resume_indicies_file.exists():
        with resume_indicies_file.open("r") as f:
            skip_indices = json.load(f).get("skip_indices", [])
        with resume_metadata_file.open("r") as f:
            resume_metadata: dict = json.load(f)
        current_metadata = get_metadata_dict(args, segments)
        assert resume_metadata == current_metadata, (
            f"Metadata mismatch. Please check your settings.\n"
            f"Tried to resume script with: {current_metadata}\nPrevious script was run with: {resume_metadata}"
        )
        print(
            f"Skipping {len(skip_indices)} already downloaded segments. Remaining: {len(segments) - len(skip_indices)}"
        )
    with resume_metadata_file.open("w") as f:
        json.dump(get_metadata_dict(args, segments), f, indent=4)

    executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        if args.parallel
        else concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    )
    with executor as pool:
        future_to_segment = {
            pool.submit(
                _process_segment,
                segment,
                idx,
                label_map,
                config,
                data_dirs,
                args.frequency,
                args.overwrite_sentinel,
                args.overwrite_osm,
            ): (idx, segment)
            for idx, segment in enumerate(segments)
            if idx not in skip_indices
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_segment), total=len(future_to_segment)):
            idx, segment = future_to_segment[future]
            try:
                future.result()
                skip_indices.append(idx)
                with resume_indicies_file.open("w") as f:
                    json.dump({"skip_indices": skip_indices}, f, indent=4)
            except Exception as e:
                print(f"Segment {segment} idx:{idx} generated an exception.")
                if "terminated abruptly" in str(e):
                    print(
                        "You might have run out of RAM. "
                        "Try lowering the number of workers via the --workers argument."
                        "Or don't use the --parallel flag to use threads instead of processes."
                    )
                raise e
    print(f"Collected {len(list(data_dirs.sentinel.glob('*.tif')))} sentinel images for {len(segments)} osm masks.")
    with (data_dirs.base_path / "metadata.json").open("w") as f:  # write settings to file for reproducibility/info
        json.dump(get_metadata_dict(args, segments), f, indent=4)
    resume_indicies_file.unlink(missing_ok=True)
    resume_metadata_file.unlink(missing_ok=True)


def get_metadata_dict(args: argparse.Namespace, segments: list[BBox]) -> dict:
    return {
        "aoi": args.aoi,
        "mapping": args.labels,
        "interval": list(TIME_INTERVAL),
        "frequency": args.frequency,
        "bands": BANDS,
        "resolution": list(SEGMENT_SIZE),
        "segment_size_km": SEGMENT_LENGTH_KM,
        "crs": CRS.value,
        "collection": DATA_COLLECTION.name,
        "num_segments": len(segments),
    }


def _process_segment(
    segment: BBox,
    idx: int,
    label_map: LabelMap,
    sh_config: sh.SHConfig,
    data_dirs: DataDirs,
    frequency: str,
    get_sentinel: bool,
    get_osm: bool,
) -> None:
    time_intervals: list[tuple[str, str]] = _split_up_time_interval(TIME_INTERVAL, frequency=frequency)
    assert len(time_intervals) > 0, "No time intervals found. Check your time interval and frequency settings."

    if get_osm:
        osm_data: gpd.GeoDataFrame = _fetch_osm_data(segment, tag_mapping=label_map)
        valid_mask = _save_rasterized_osm_data(osm_data, aoi=segment, idx=idx, other_cls_label=0, osm_dir=data_dirs.osm)
        if not valid_mask:
            return

    if not get_sentinel:
        return

    sentinel_data: list[npt.NDArray] = []
    for time_interval in time_intervals:
        data = fetch_sentinel_data(segment=segment, sh_config=sh_config, time_interval=time_interval)
        if (data == 0).sum() > 0.5 * data.size:  # if more than 50% is 0, it is usually cut off
            continue
        sentinel_data.append(data)
        time.sleep(2)  # to avoid rate limiting
    save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=segment, sentinel_dir=data_dirs.sentinel)


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
        evalscript=EVALSCRIPT,
        input_data=[
            sh.SentinelHubRequest.input_data(
                data_collection=DATA_COLLECTION,
                time_interval=time_interval,
                maxcc=0.05,
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
    pixel_size_x, pixel_size_y = _calculate_pixel_size(aoi, SEGMENT_SIZE)
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


def _calculate_pixel_size(aoi: BBox, resolution: tuple[int, int]) -> tuple[float, float]:
    pixel_size_x = (aoi.east - aoi.west) / resolution[0]
    pixel_size_y = (aoi.north - aoi.south) / resolution[1]
    return pixel_size_x, pixel_size_y


def _fetch_osm_data(segment: BBox, tag_mapping: LabelMap) -> gpd.GeoDataFrame:
    """Fetch individual labels -> concant to one -> put on top of aoi segment filled with 'other' label
    NOTE: If there are overlapping features, the last one will be used. TODO make sure sealing is highest prio!
    """
    gdf_list = [
        _fetch_osm_data_by_tags(segment, tags=label["osm_tags"], class_label_idx=i)
        for i, label in enumerate(list(tag_mapping.values())[1:], start=1)  # skip "other" class
    ]
    osm_gdf: gpd.GeoDataFrame = pd.concat(gdf_list, ignore_index=True)  # type: ignore
    osm_gdf.dropna(subset=["geometry"], inplace=True)
    osm_gdf.set_crs(str(CRS), inplace=True)
    osm_gdf.to_crs(str(CRS), inplace=True)
    return osm_gdf


def _fetch_osm_data_by_tags(segment: BBox, tags: OSMTagMap, class_label_idx: int) -> gpd.GeoDataFrame:
    """
    Fetch OSM data within the set bounds and with given feature tags, and assign class labels.
    If no data is found, returns an empty GeoDataFrame.

    Args:
        class_label_idx (int): Class label to assign to the features.
        segment (BBox): Segment to fetch data from.
        tags (dict): Dictionary of feature tags to fetch.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometry and class label.
    """
    try:
        osm_data = ox.features_from_bbox(
            bbox=segment
            , tags=tags
        )
        osm_data["class"] = class_label_idx
        return osm_data[["geometry", "class"]]
    except ox._errors.InsufficientResponseError:
        print(f"No OSM data found for {tags} in segment {segment}")
        return gpd.GeoDataFrame(columns=["geometry", "class"])


def _save_rasterized_osm_data(gdf: gpd.GeoDataFrame, aoi: BBox, osm_dir: Path, idx: int, other_cls_label: int) -> bool:
    pixel_size_x, pixel_size_y = _calculate_pixel_size(aoi, SEGMENT_SIZE)
    transform = from_origin(west=aoi.west, north=aoi.north, xsize=pixel_size_x, ysize=pixel_size_y)

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
    burned = rasterize(shapes=shapes, out_shape=SEGMENT_SIZE, transform=transform, fill=other_cls_label, dtype="uint8")

    if (zero_sum := (burned == 0).sum()) > MAX_UNLABELED * burned.size:
        print(f"Segment {idx} has too many unlabeled pixels: {(zero_sum / burned.size):.2f}")
        return False

    with rasterio.open(
        fp=osm_dir / f"{idx}.tif",
        mode="w",
        driver="GTiff",
        height=SEGMENT_SIZE[0],
        width=SEGMENT_SIZE[1],
        count=1,
        dtype="uint8",
        crs=gdf.crs,
        transform=transform,
    ) as f:
        f.write_band(1, burned)
    return True


def _visualize_segment_bbox() -> None:
    import shapely
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    aoi = AOIs["at"]
    segments = calculate_segments(aoi, SEGMENT_LENGTH_KM)
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
