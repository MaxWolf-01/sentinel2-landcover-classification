import argparse
import concurrent
import os
import typing

import einops
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import sentinelhub as sh
import geopandas as gpd
import numpy.typing as npt
from dotenv import load_dotenv
from tqdm import tqdm

from src.configs.label_mappings import LabelMap, OSMTagMap, GENERAL_MAP
from src.configs.paths import SENTINEL_DIR, OSM_DIR, ROOT_DIR
from src.utils import load_pritvhi_bands


class BBox(typing.NamedTuple):
    north: float
    south: float
    east: float
    west: float


AOIs: dict[str, BBox] = {
    "VIE": BBox(north=48.341646, south=47.739323, east=16.567383, west=15.117188),  # ca. 20 tifs; rough crop
    "test": BBox(north=48.980217, south=46.845164, east=17.116699, west=13.930664),  # ca. 150 tifs;VIE,NÖ,OÖ,NBGLD,Graz
    # "AT": ...,
}

CRS: sh.CRS = sh.CRS.WGS84  # == "4326" (EPSG)
DATA_COLLECTION = sh.DataCollection.SENTINEL2_L1C  # TODO is this the correct one?
TIME_INTERVAL: tuple[str, str] = ("2023-07-01", "2023-07-15")  # TODO find suitable time interval
BANDS: list[str] = load_pritvhi_bands()
RESOLUTION: tuple[int, int] = (512, 512)  # Width and Height in pixels
SEGMENT_SIZE: int = 25  # Size of segments in km

LABEL_MAPPINGS: dict[str, LabelMap] = {
    "general": GENERAL_MAP,
    # "soil_sealing": ...
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="VIE", help=f"Specify an AOI. Default: VIE. Available:{list(AOIs)}")
    parser.add_argument(
        "--labels", type=str, default="general", help="Specify a label mapping to use. Default: general."
    )
    args = parser.parse_args()
    aoi = AOIs[args.aoi or "VIE"]
    label_map = LABEL_MAPPINGS[args.labels or "general"]

    ox.config(use_cache=True, cache_folder=ROOT_DIR / "osmnx_cache")

    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    SENTINEL_DIR.mkdir(exist_ok=True, parents=True)
    OSM_DIR.mkdir(exist_ok=True, parents=True)

    segments: list[BBox] = calculate_segments(aoi, SEGMENT_SIZE)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_segment = {
            executor.submit(process_segment, segment, idx, label_map, config): (idx, segment)
            for idx, segment in enumerate(segments)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_segment), total=len(future_to_segment)):
            idx, segment = future_to_segment[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Segment {idx} generated an exception: {exc}")


def process_segment(segment: BBox, idx: int, label_map: LabelMap, sh_config: sh.SHConfig) -> None:
    sentinel_data: npt.NDArray = fetch_sentinel_data(segment=segment, sh_config=sh_config)
    save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=segment)
    osm_data = fetch_osm_data(segment, tag_mapping=label_map)
    save_rasterized_osm_data(osm_data, aoi=segment, idx=idx, other_cls_label=label_map["other"]["idx"])


def calculate_segments(bbox: BBox, segment_size_km: int) -> list[BBox]:
    lon_diff = bbox.east - bbox.west
    lat_diff = bbox.north - bbox.south
    lon_segments = int(np.ceil(lon_diff / (segment_size_km / 111)))
    lat_segments = int(np.ceil(lat_diff / (segment_size_km / 111)))
    segments = []
    lon_segment_size = lon_diff / lon_segments
    lat_segment_size = lat_diff / lat_segments
    for i in range(lon_segments):
        for j in range(lat_segments):
            segment_min_lon = bbox.west + i * lon_segment_size
            segment_max_lon = segment_min_lon + lon_segment_size
            segment_min_lat = bbox.south + j * lat_segment_size
            segment_max_lat = segment_min_lat + lat_segment_size
            segments.append(
                BBox(north=segment_max_lat, south=segment_min_lat, east=segment_max_lon, west=segment_min_lon)
            )
    return segments


def fetch_sentinel_data(segment: BBox, sh_config: sh.SHConfig) -> npt.NDArray:
    evalscript = _create_evalscript(BANDS)
    bbox: sh.BBox = sh.BBox((segment.west, segment.south, segment.east, segment.north), crs=CRS)
    request = sh.SentinelHubRequest(
        evalscript=evalscript,
        input_data=[sh.SentinelHubRequest.input_data(data_collection=DATA_COLLECTION, time_interval=TIME_INTERVAL)],
        responses=[sh.SentinelHubRequest.output_response("default", sh.MimeType.TIFF)],
        bbox=bbox,
        size=RESOLUTION,
        config=sh_config,
        data_folder="../data/sentinel/",
    )
    return request.get_data(save_data=False)[0]


def _create_evalscript(bands: list[str]) -> str:
    bands_str = ", ".join([f'"{band}"' for band in bands])
    return f"""
    //VERSION=3
    function setup() {{
        return {{
            input: [{{
                bands: [{bands_str}],
                units: "DN"
            }}],
            output: {{
                bands: {len(bands)},
                sampleType: "INT16"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [{', '.join([f'sample.{band}' for band in bands])}];
    }}
    """


def save_sentinel_data_as_geotiff(data: npt.NDArray, idx: int, aoi: BBox) -> None:
    pixel_size_x, pixel_size_y = calculate_pixel_size(aoi, RESOLUTION)
    data = einops.rearrange(data, "h w c -> c h w")
    with rasterio.open(
        fp=SENTINEL_DIR / f"{idx}.tif",
        mode="w",
        driver="GTiff",
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        count=len(BANDS),
        dtype=data.dtype,
        crs=rasterio.crs.CRS().from_epsg(code=CRS.value),
        transform=rasterio.transform.from_origin(aoi.west, aoi.north, pixel_size_x, pixel_size_y),
    ) as f:
        f.write(data)


def calculate_pixel_size(aoi: BBox, resolution: tuple[int, int]) -> tuple[float, float]:
    pixel_size_x = (aoi.east - aoi.west) / resolution[0]
    pixel_size_y = (aoi.north - aoi.south) / resolution[1]
    return pixel_size_x, pixel_size_y


def fetch_osm_data(segment: BBox, tag_mapping: LabelMap) -> gpd.GeoDataFrame:
    """Fetch individual labels -> concant to one -> put on top of aoi segment filled with 'other' label
    NOTE: If there are overlapping features, the last one will be used.
    """
    gdf_list = [
        fetch_osm_data_by_tags(segment, tags=entry["osm_tags"], class_label=entry["idx"])
        for entry in list(tag_mapping.values())[1:]  # skip "other" class
    ]
    osm_gdf: gpd.GeoDataFrame = pd.concat(gdf_list, ignore_index=True)  # type: ignore
    osm_gdf.dropna(subset=["geometry"], inplace=True)
    osm_gdf.set_crs(str(CRS), inplace=True)
    osm_gdf.to_crs(str(CRS), inplace=True)
    return osm_gdf


def fetch_osm_data_by_tags(segment: BBox, tags: OSMTagMap, class_label: int) -> gpd.GeoDataFrame:
    """
    Fetch OSM data within the set bounds and with given feature tags, and assign class labels.
    If no data is found, returns an empty GeoDataFrame.

    Args:
        class_label (int): Class label to assign to the features.
        segment (BBox): Segment to fetch data from.
        tags (dict): Dictionary of feature tags to fetch.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometry and class label.
    """
    try:
        osm_data = ox.features_from_bbox(
            north=segment.north, south=segment.south, east=segment.east, west=segment.west, tags=tags
        )
        osm_data["class"] = class_label
        return osm_data[["geometry", "class"]]
    except ox._errors.InsufficientResponseError:
        return gpd.GeoDataFrame(columns=["geometry", "class"])


def save_rasterized_osm_data(gdf: gpd.GeoDataFrame, aoi: BBox, idx: int, other_cls_label: int) -> None:
    pixel_size_x, pixel_size_y = calculate_pixel_size(aoi, RESOLUTION)
    transform = from_origin(west=aoi.west, north=aoi.north, xsize=pixel_size_x, ysize=pixel_size_y)

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
    burned = rasterize(shapes=shapes, out_shape=RESOLUTION, transform=transform, fill=other_cls_label, dtype="uint8")
    with rasterio.open(
        fp=OSM_DIR / f"{idx}.tif",
        mode="w",
        driver="GTiff",
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        count=1,
        dtype="uint8",
        crs=gdf.crs,
        transform=transform,
    ) as f:
        f.write_band(1, burned)


if __name__ == "__main__":
    main()
