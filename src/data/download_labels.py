"""
This script downloads either OSM or CNES land cover labels for a given AOI and saves them as GeoTIFFs.
The labels are saved as data/<aoi>/label/<label-type>/<segment-id>.tif,
where <label-type> is either "osm-*" or "cnes-full".
"""
import argparse
import concurrent
import json
import os
import shutil
import warnings
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import osmnx as ox
import pandas as pd
import rasterio
import sentinelhub as sh
from dotenv import load_dotenv
from rasterio.features import rasterize
from rasterio.transform import from_origin
from tqdm import tqdm

from src.configs.data_config import (
    AOIs,
    BBox,
    CNES_LABEL_EVALSCRIPT,
    CRS,
    DataDirs,
    LABEL_MAPS,
    LabelMap,
    MAX_UNLABELED,
    SEGMENT_LENGTH_KM,
    SEGMENT_SIZE,
    TIME_INTERVAL,
)
from src.configs.osm_label_mapping import OSMTagMap
from src.configs.paths import LOG_DIR, ROOT_DIR
from src.data.download_sentinel import (
    calculate_pixel_size,
    calculate_segments,
    load_resume_data,
    save_resume_data,
)


class LabelQualityWarning(Warning):
    """Warning raised when label quality is below a certain threshold."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OSM or CNES land cover labels for a given AOI.")
    parser.add_argument("aoi", type=str, help="Area of interest to download labels for.")
    parser.add_argument("labels", type=str, help="Type of labels to download.")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to use for downloading.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing label data.")
    parser.add_argument("--resume", action="store_true", help="Resume downloading from last segment.")
    args = parser.parse_args()
    assert (
        not args.labels.startswith("cnes") or args.labels == "cnes-full"
    ), "CNES download only available for cnes-full, for simplified CNES mappings, run corresponding script."
    cnes: bool = args.labels == "cnes-full"
    label_map: LabelMap = LABEL_MAPS[args.labels]

    segments: list[BBox] = calculate_segments(bbox=AOIs[args.aoi], segment_size_km=SEGMENT_LENGTH_KM)
    print(f"Number of segments: {len(segments)}")

    data_dirs = DataDirs(aoi=args.aoi, map_type=args.labels)
    if args.overwrite and data_dirs.label.exists() and not args.resume:
        warnings.warn(f"Deleting existing label data at: {data_dirs.label}")
        input("Press Enter to continue...")
        shutil.rmtree(data_dirs.label, ignore_errors=True)
    data_dirs.label.mkdir(parents=True, exist_ok=True)

    resume_file = data_dirs.base_path / "resume.json"
    resume_metadata_file = data_dirs.base_path / "metadata.tmp.json"
    skip_indices: list[int] = (
        load_resume_data(
            resume_file=resume_file,
            metadata_file=resume_metadata_file,
            current_metadata=_metadata_dict(args=args, segments=segments),
            segments=segments,
        )
        if args.resume
        else []
    )

    if cnes:
        load_dotenv()
        config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))
        process_segment = partial(_process_cnes_segment, sh_config=config, path=data_dirs.label)
    else:
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(ROOT_DIR / "osmnx_cache")
        binary: bool = "binary" in args.labels
        process_segment = partial(_process_osm_segment, path=data_dirs.label, label_map=label_map, binary=binary)

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
            except LabelQualityWarning as e:
                with log_file.open("a") as f:
                    f.write(str(e) + "\n")
            except Exception as e:
                msg = f"Error in segment {idx}: {e}\n"
                print(msg)
                with log_file.open("a") as f:
                    f.write(msg)
                raise e
    print(f"Collected {len(skip_indices)} label segments.")
    with (data_dirs.label / "metadata.json").open("w") as f:  # write settings to file for reproducibility/info
        json.dump(_metadata_dict(args=args, segments=segments), f, indent=4)
    resume_file.unlink(missing_ok=True)
    resume_metadata_file.unlink(missing_ok=True)


def _metadata_dict(args: argparse.Namespace, segments: list[BBox]) -> dict:
    return {
        "aoi": args.aoi,
        "crs": CRS.value,
        "labels": args.labels,
        "num_segments": len(segments),
        "resolution": list(SEGMENT_SIZE),
        "time_interval": list(TIME_INTERVAL),
        "type": "label",
        "segment_length_km": SEGMENT_LENGTH_KM,
        "max_unlabeled": MAX_UNLABELED,
    }


def _process_cnes_segment(idx: int, segment: BBox, path: Path, sh_config: sh.SHConfig) -> None:
    # TODO validate data with CNES confidence band... in request/evalscript?
    data: np.ndarray = _fetch_cnes_land_cover_labels(segment, time_interval=TIME_INTERVAL, sh_config=sh_config)
    _save_cnes_labels(data, aoi=segment, idx=idx, cnes_dir=path)


def _process_osm_segment(idx: int, segment: BBox, path: Path, label_map: LabelMap, binary: bool) -> None:
    unlabeled_ratio_info: int | None = _save_rasterized_osm_data(
        gdf=_fetch_osm_data(segment=segment, tag_mapping=label_map),
        aoi=segment,
        idx=idx,
        other_cls_label=0,
        osm_dir=path,
        binary=binary,
    )
    if unlabeled_ratio_info is not None:
        raise LabelQualityWarning(f"Segment {idx} has too many unlabeled pixels: {unlabeled_ratio_info:.2f}")


def _fetch_osm_data(segment: BBox, tag_mapping: LabelMap) -> gpd.GeoDataFrame:
    """Fetch individual labels -> concant to one -> put on top of aoi segment filled with 'other' label
    NOTE: If there are overlapping features, the last one will be used.
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
            north=segment.north, south=segment.south, east=segment.east, west=segment.west, tags=tags
        )
        osm_data["class"] = class_label_idx
        return osm_data[["geometry", "class"]]
    except ox._errors.InsufficientResponseError:
        print(f"No OSM data found for {tags} in segment {segment}")
        return gpd.GeoDataFrame(columns=["geometry", "class"])


def _save_rasterized_osm_data(
    gdf: gpd.GeoDataFrame, aoi: BBox, osm_dir: Path, idx: int, other_cls_label: int, binary: bool
) -> int | None:
    pixel_size_x, pixel_size_y = calculate_pixel_size(aoi, SEGMENT_SIZE)
    transform = from_origin(west=aoi.west, north=aoi.north, xsize=pixel_size_x, ysize=pixel_size_y)

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
    burned = rasterize(shapes=shapes, out_shape=SEGMENT_SIZE, transform=transform, fill=other_cls_label, dtype="uint8")

    passes, zero_count = _passes_unlabeled_threshold(burned)
    if passes and not binary:
        return zero_count / burned.size

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


def _fetch_cnes_land_cover_labels(segment: BBox, sh_config: sh.SHConfig, time_interval: tuple[str, str]) -> npt.NDArray:
    request = sh.SentinelHubRequest(
        evalscript=CNES_LABEL_EVALSCRIPT,
        input_data=[
            sh.SentinelHubRequest.input_data(
                data_collection=sh.DataCollection.define_byoc(collection_id="9baa2732-6597-49d2-ae3b-68ba0a5386b2"),
                time_interval=time_interval,
            )
        ],
        responses=[sh.SentinelHubRequest.output_response("default", sh.MimeType.TIFF)],
        bbox=sh.BBox((segment.west, segment.south, segment.east, segment.north), crs=CRS),
        size=SEGMENT_SIZE,
        config=sh_config,
    )
    return request.get_data()[0]


def _save_cnes_labels(data: npt.NDArray, aoi: BBox, idx: int, cnes_dir: Path) -> None:
    pixel_size_x, pixel_size_y = calculate_pixel_size(aoi, SEGMENT_SIZE)
    transform = from_origin(west=aoi.west, north=aoi.north, xsize=pixel_size_x, ysize=pixel_size_y)
    data = data[:, :, 0]  # only use the first band (actual label; drop confidence and validity bands)
    with rasterio.open(
        fp=cnes_dir / f"{idx}.tif",
        mode="w",
        driver="GTiff",
        height=SEGMENT_SIZE[0],
        width=SEGMENT_SIZE[1],
        count=1,
        dtype="uint8",
        crs=str(CRS),
        transform=transform,
    ) as f:
        f.write_band(1, data)


def _passes_unlabeled_threshold(data: npt.NDArray) -> tuple[bool, int]:
    return (zero_count := (data == 0).sum()) > MAX_UNLABELED * data.size, zero_count


if __name__ == "__main__":
    main()
