import argparse
import concurrent
import os
import traceback
import typing
from pathlib import Path

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
from shapely import Polygon, MultiPolygon, Point
from tqdm import tqdm

from src.configs.label_mappings import LabelMap, OSMTagMap, MULTICLASS_MAP, BINARY_MAP, MAPS
from src.configs.paths import ROOT_DIR, DATA_DIR


class BBox(typing.NamedTuple):
    north: float
    south: float
    east: float
    west: float

    def __str__(self, p: int | None = None) -> str:
        f = f"{{:.{p}f}}" if p is not None else "{}"
        return (
            f"(N: {f.format(self.north)}, S: {f.format(self.south)},"
            f" E: {f.format(self.east)}, W: {f.format(self.west)})"
        )


AOIs: dict[str, BBox] = {
    "vie": BBox(north=48.341646, south=47.739323, east=16.567383, west=15.117188),  # ca. 20 tifs; rough crop
    "test": BBox(north=48.980217, south=46.845164, east=17.116699, west=13.930664),  # 151 tifs;VIE,NÖ,OÖ,NBGLD,Graz
    "at": BBox(north=49.009121, south=46.439861, east=17.523438, west=9.008164),  # 456 tifs; AT + bits of neighbours
}

CRS: sh.CRS = sh.CRS.WGS84  # == "4326" (EPSG)
DATA_COLLECTION = sh.DataCollection.SENTINEL2_L2A  # use atmosphericlly corrected data
TIME_INTERVAL: tuple[str, str] = ("2023-07-01", "2023-07-15")  # TODO find suitable time interval
# https://github.com/NASA-IMPACT/hls-foundation-os/issues/15#issuecomment-1667699259
BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12"]
RESOLUTION: tuple[int, int] = (512, 512)  # Width and Height in pixels
SEGMENT_SIZE: int = 25  # Size of segments in km

LABEL_MAPPINGS: dict[str, LabelMap] = {
    "multi": MULTICLASS_MAP,
    "binary": BINARY_MAP,
}


class S2OSMDataDirs:
    def __init__(self, aoi: str, map_type: str) -> None:
        self.base_path: Path = DATA_DIR / aoi / map_type
        self.sentinel: Path = self.base_path / "sentinel"
        self.osm: Path = self.base_path / "osm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="vie", help=f"Specify an AOI. Default: VIE. Available:{list(AOIs)}")
    parser.add_argument("--labels", type=str, default="multiclass", help="Specify a label mapping to use.")
    parser.add_argument("--workers", type=int, default=1, help="Specify the number of workers.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing.")
    args = parser.parse_args()
    aoi = AOIs[args.aoi]
    label_map = MAPS[args.labels]
    max_workers = args.workers

    ox.config(use_cache=True, cache_folder=ROOT_DIR / "osmnx_cache")

    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    data_dirs = S2OSMDataDirs(aoi=args.aoi, map_type=args.labels)
    data_dirs.osm.mkdir(parents=True, exist_ok=True)
    data_dirs.sentinel.mkdir(parents=True, exist_ok=True)

    segments: list[BBox] = calculate_segments(aoi, SEGMENT_SIZE)

    pool_executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        if args.parallel
        else concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    )
    with pool_executor as executor:
        future_to_segment = {
            executor.submit(_process_segment, segment, idx, label_map, config, data_dirs): (idx, segment)
            for idx, segment in enumerate(segments)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_segment), total=len(future_to_segment)):
            idx, segment = future_to_segment[future]
            try:
                future.result()
            except Exception as e:
                # cleanup, so we don't have imbalance between sentinel and osm data
                sentinel_file = data_dirs.sentinel / f"{idx}.tif"
                sentinel_file.unlink(missing_ok=True)
                osm_file = data_dirs.osm / f"{idx}.tif"
                osm_file.unlink(missing_ok=True)

                print(f"Segment {segment} idx:{idx} generated an exception:\n{traceback.format_exc()}")
                if "terminated abruptly" in str(e):
                    print(
                        "You might have run out of RAM. "
                        "Try lowering the number of workers via the --workers argument."
                        "Or don't use the --parallel flag to use threads instead of processes."
                    )


def _process_segment(
    segment: BBox, idx: int, label_map: LabelMap, sh_config: sh.SHConfig, data_dirs: S2OSMDataDirs
) -> None:
    sentinel_data: npt.NDArray = fetch_sentinel_data(segment=segment, sh_config=sh_config)
    save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=segment, sentinel_dir=data_dirs.sentinel)
    osm_data = _fetch_osm_data(segment, tag_mapping=label_map)
    _save_rasterized_osm_data(osm_data, aoi=segment, idx=idx, other_cls_label=0, osm_dir=data_dirs.osm)  # other=0


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


def save_sentinel_data_as_geotiff(data: npt.NDArray, idx: int, aoi: BBox, sentinel_dir: Path) -> None:
    pixel_size_x, pixel_size_y = _calculate_pixel_size(aoi, RESOLUTION)
    data = einops.rearrange(data, "h w c -> c h w")
    with rasterio.open(
            fp=sentinel_dir / f"{idx}.tif",
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
            north=segment.north, south=segment.south, east=segment.east, west=segment.west, tags=tags
        )
        osm_data["class"] = class_label_idx
        return osm_data[["geometry", "class"]]
    except ox._errors.InsufficientResponseError:
        print(f"No OSM data found for {tags} in segment {segment}")
        return gpd.GeoDataFrame(columns=["geometry", "class"])


def generate_points_in_polygon(polygon, density=100):
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.linspace(minx, maxx, num=density)
    y_coords = np.linspace(miny, maxy, num=density)
    points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if polygon.contains(point):
                points.append(point)
    return points

def listPoints(someGeometry):
    '''List the points in a Polygon in a geometry entry - some polygons are more complex than others, so accommodating for that'''
    pointList = []
    try:
        #Note: might miss parts within parts with this
        for part in someGeometry:
            x, y = part.exterior.coords.xy
            pointList.append(list(zip(x,y)))
    except:
        try:
            x,y = someGeometry.exterior.coords.xy
            pointList.append(list(zip(x,y)))
        except:
            #this will return the geometry as is, enabling you to see if special handling is required - then modify the function as need be
            pointList.append(someGeometry)
    return pointList
def _save_rasterized_osm_data(gdf: gpd.GeoDataFrame, aoi, osm_dir: Path, idx: int, other_cls_label: int, density=10) -> None:
    pixel_size_x, pixel_size_y = _calculate_pixel_size(aoi, RESOLUTION)
    transform = from_origin(west=aoi.west, north=aoi.north, xsize=pixel_size_x, ysize=pixel_size_y)

    shapes = []
    for _, row in gdf.iterrows():
        geom = row['geometry']
        if geom.is_valid and not geom.is_empty:
            if geom.geom_type == 'Polygon':
                points = geom.apply(lambda x: listPoints(x)).values.tolist()
                shapes.extend(((point, row['class']) for point in points))
            elif geom.geom_type == 'MultiPolygon':
                print('polygon detected, opinion rejected')
                for poly in geom.geoms:
                    points = geom.apply(lambda x: listPoints(x)).values.tolist()
                    shapes.extend(((point, row['class']) for point in points))
            else:
                # Handle other geometry types (e.g., Point, LineString) as-is
                shapes.append((geom, row['class']))

    if shapes:
        burned = rasterize(shapes=shapes, out_shape=RESOLUTION, transform=transform, fill=other_cls_label,
                           dtype='uint8', all_touched=True)
        with rasterio.open(fp=osm_dir / f"{idx}.tif", mode='w', driver='GTiff', height=RESOLUTION[0],
                           width=RESOLUTION[1], count=1, dtype='uint8', crs=gdf.crs, transform=transform) as f:
            f.write_band(1, burned)


if __name__ == "__main__":
    main()
