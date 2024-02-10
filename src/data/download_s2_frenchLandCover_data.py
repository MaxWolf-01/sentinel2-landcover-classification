import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import rasterio
import sentinelhub as sh
from dotenv import load_dotenv
from tqdm import tqdm

from src.configs.download_config import BBox, CRS, RESOLUTION, AOIs, DataDirs, SEGMENT_SIZE
from src.data.download_s2_osm_data import _calculate_segments, _fetch_sentinel_data


def fetch_cnes_land_cover_data(segment: BBox, sh_config: sh.SHConfig, year: int) -> np.ndarray:
    collection_id = "9baa2732-6597-49d2-ae3b-68ba0a5386b2"
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{"bands": ["OCS", "OCS_Confidence", "OCS_Validity"], "units": "DN"}],
            output: {bands: 3, sampleType: "UINT8"}
        };
    }

    function evaluatePixel(sample) {
        return [sample.OCS, sample.OCS_Confidence, sample.OCS_Validity];
    }
    """
    bbox_sh = sh.BBox([segment.west, segment.south, segment.east, segment.north], crs=CRS)
    request = sh.SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            sh.SentinelHubRequest.input_data(
                data_collection=sh.DataCollection.define_byoc(collection_id),
                time_interval=(f"{year}-01-01", f"{year}-12-31"),
            )
        ],
        responses=[sh.SentinelHubRequest.output_response("default", sh.MimeType.TIFF)],
        bbox=bbox_sh,
        size=RESOLUTION,
        config=sh_config,
    )
    data = request.get_data()[0]
    return data


def save_data_as_geotiff(data: np.ndarray, output_path: Path, bbox: BBox, is_sentinel=True) -> None:
    bands_count = data.shape[2]  # data shape is (height, width, bands)

    pixel_size_x = (bbox.east - bbox.west) / RESOLUTION[0]
    pixel_size_y = (bbox.north - bbox.south) / RESOLUTION[1]

    transform = rasterio.transform.from_origin(west=bbox.west, north=bbox.north, xsize=pixel_size_x, ysize=pixel_size_y)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=bands_count,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        if is_sentinel:
            # For Sentinel data, ensure bands are in the first dimension
            dst.write(data.transpose((2, 0, 1)))
        else:
            # For CNES Land Cover data (or any multiclass-band data), write each band separately
            for i in range(1, bands_count + 1):
                dst.write(data[:, :, i - 1], i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="fr", help="Specify an AOI. Available: {}".format(list(AOIs.keys())))
    parser.add_argument("--year", type=int, default=2020, help="Specify the year for the CNES Land Cover Map.")
    parser.add_argument("--labels", type=str, default="multiclass", help="Specify a label mapping to use.")
    parser.add_argument("--workers", type=int, default=4, help="Specify the number of workers.")
    args = parser.parse_args()

    load_dotenv()
    sh_config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    aoi = AOIs[args.aoi]
    data_dirs = DataDirs(aoi=args.aoi, map_type=args.labels)
    data_dirs.land_cover.mkdir(parents=True, exist_ok=True)
    data_dirs.sentinel.mkdir(parents=True, exist_ok=True)

    segments = _calculate_segments(aoi, SEGMENT_SIZE)

    for idx, segment in enumerate(tqdm(segments)):
        try:
            sentinel_data = _fetch_sentinel_data(segment, sh_config)
            save_data_as_geotiff(sentinel_data, data_dirs.sentinel / f"sentinel_{idx}.tif", segment, is_sentinel=True)

            cnes_data = fetch_cnes_land_cover_data(segment, sh_config, args.year)
            save_data_as_geotiff(cnes_data, data_dirs.land_cover / f"cnes_{idx}.tif", segment, is_sentinel=False)
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
