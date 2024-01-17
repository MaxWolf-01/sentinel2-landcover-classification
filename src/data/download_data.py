import argparse
import json
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
from pathlib import Path
import geopandas as gpd
import numpy.typing as npt
from tqdm import tqdm
from dotenv import load_dotenv

from src.configs.paths import CONFIG_DIR, SENTINEL_DIR, OSM_DIR, ROOT_DIR
from src.utils import load_pritvhi_bands


class BBox(typing.NamedTuple):
    north: float
    south: float
    east: float
    west: float


AOIs: dict[str, BBox] = {
    "VIE": BBox(north=48.341646, south=47.739323, east=16.567383, west=15.117188),  # ca. 20 tifs; rough crop
    "test": BBox(north=48.980217, south=46.845164, east=17.116699, west=13.930664),  # ca. 150 tifs;VIE,NÖ,OÖ,NBGLD,Graz
    "AT": ...,
}

DATA_COLLECTION = sh.DataCollection.SENTINEL2_L1C  # TODO is this the correct one?
TIME_INTERVAL: tuple[str, str] = ("2023-07-01", "2023-07-15")  # TODO find suitable time interval
BANDS: list[str] = load_pritvhi_bands()
RESOLUTION: tuple[int, int] = (512, 512)  # Width and Height in pixels
SEGMENT_SIZE: int = 25  # Size of segments in km

TAG_MAPPING_PATH: Path = CONFIG_DIR / "tag_mapping.json"

# todo what is the difference to tag mapping? (I think FEATURE_TAGS should be deleted and built from tag_mapping.json!)
# TODO set proper tags
FEATURE_TAGS = {  # class labels are indices in this dict (starting at 1)
    "building": ["yes", "residential", "commercial", "industrial"],
    "highway": ["primary", "secondary", "tertiary", "residential"],
    "landuse": ["residential", "commercial", "industrial", "park"],
    "natural": ["water", "wood", "grassland"],
    "amenity": ["school", "hospital", "parking", "restaurant"],
}


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


def fetch_osm_data_by_tags(segment: BBox, tags: dict, class_label: int) -> gpd.GeoDataFrame:
    """
    Fetch OSM data within the set bounds and with given feature tags, and assign class labels.

    Parameters:
        class_label (int): Class label to assign to the features.
        segment (BBox): Segment to fetch data from.
        tags (dict): Dictionary of feature tags to fetch.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with geometry and class label.
    """
    osm_data = ox.features_from_bbox(
        north=segment.north, south=segment.south, east=segment.east, west=segment.west, tags=tags
    )
    osm_data["class"] = class_label
    return osm_data[["geometry", "class"]]


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


def standardize_osm_tags(gdf: gpd.GeoDataFrame, tag_mapping: dict[str, str]) -> gpd.GeoDataFrame:
    #   for variant, standard in tag_mapping.items():
    #      gdf.loc[gdf['key'] == variant, 'key'] = standard
    # TODO: Handle tag-mapping
    return gdf


def fetch_sentinel_data(segment: BBox, sh_config: sh.SHConfig) -> npt.NDArray:
    evalscript = _create_evalscript(BANDS)
    bbox: sh.BBox = sh.BBox((segment.west, segment.south, segment.east, segment.north), crs=sh.CRS.WGS84)
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


def save_sentinel_data_as_geotiff(data: npt.NDArray, idx: int, aoi: BBox) -> None:
    data = einops.rearrange(data, "h w c -> c h w")
    output_path = SENTINEL_DIR / f"{idx}.tif"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        count=len(BANDS),
        dtype=data.dtype,
        crs="+proj=latlong",  # TODO is this correct?
        transform=rasterio.transform.from_origin(*aoi[:2], RESOLUTION[0], RESOLUTION[1]),  # TODO is this correct?
    ) as dst:
        for i in range(len(BANDS)):
            # Write each band separately | TODO why?
            dst.write(data[i], i + 1)


def fetch_osm_data(segment: BBox, tag_mapping: dict[str, str]) -> gpd.GeoDataFrame:
    gdf_list = [
        fetch_osm_data_by_tags(segment, tags={feature: tags}, class_label=i)
        for i, (feature, tags) in enumerate(FEATURE_TAGS.items(), start=1)
    ]

    gdf: gpd.GeoDataFrame = pd.concat(gdf_list, ignore_index=True)  # type: ignore
    gdf = standardize_osm_tags(gdf, tag_mapping=tag_mapping)
    gdf = gdf.dropna(subset=["geometry"])
    return gdf


def save_rasterized_osm_data(gdf: gpd.GeoDataFrame, idx: int) -> None:
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    pixel_size_x = (maxx - minx) / RESOLUTION[0]
    pixel_size_y = (maxy - miny) / RESOLUTION[1]
    transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    output_path = OSM_DIR / f"{idx}.tif"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        count=1,
        dtype="uint8",
        crs=gdf.crs,
        transform=transform,
    ) as dst:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf["class"]))
        burned = rasterize(shapes=shapes, out_shape=RESOLUTION, transform=transform, fill=0)
        dst.write_band(1, burned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="VIE", help=f"Supply an AOI. Default: VIE. Available:{list(AOIs)}")
    args = parser.parse_args()
    aoi = args.aoi or "VIE"

    ox.config(use_cache=True, cache_folder=ROOT_DIR / "osmnx_cache")
    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))
    with TAG_MAPPING_PATH.open("r") as file:
        tag_mapping: dict[str, str] = json.load(file)

    SENTINEL_DIR.mkdir(exist_ok=True, parents=True)
    OSM_DIR.mkdir(exist_ok=True, parents=True)

    segments: list[BBox] = calculate_segments(AOIs[aoi], SEGMENT_SIZE)
    for idx, segment in enumerate(tqdm(segments)):
        sentinel_data: npt.NDArray = fetch_sentinel_data(segment=segment, sh_config=config)
        save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=aoi)
        osm_data = fetch_osm_data(segment, tag_mapping=tag_mapping)
        save_rasterized_osm_data(osm_data, idx=idx)


if __name__ == "__main__":
    main()
