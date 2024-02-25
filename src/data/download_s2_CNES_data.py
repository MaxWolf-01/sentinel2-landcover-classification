import argparse
import concurrent.futures
import json
import os
import shutil
import traceback
import warnings
from pathlib import Path
import numpy as np
import rasterio
import sentinelhub as sh
from dotenv import load_dotenv
from tqdm import tqdm
from src.configs.download_config import BBox, CRS, RESOLUTION, AOIs, DataDirs, SEGMENT_SIZE, TIME_INTERVAL, \
    SEGMENT_LENGTH_KM
from src.data.download_s2_osm_data import fetch_sentinel_data, calculate_segments, _split_up_time_interval


def fetch_cnes_land_cover_data(segment: BBox, sh_config: sh.SHConfig, time_interval: tuple[str, str]) -> np.ndarray:
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
                time_interval=time_interval,
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
            dst.write(data.transpose((2, 0, 1)))
        else:
            for i in range(1, bands_count + 1):
                dst.write(data[:, :, i - 1], i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="fr", help="Specify an AOI. Available: {}".format(list(AOIs.keys())))
    parser.add_argument("--labels", type=str, default="multiclass", help="Specify a label mapping to use.")
    parser.add_argument("--workers", type=int, default=4, help="Specify the number of workers.")
    parser.add_argument("--frequency", type=str, default="QS", help="Pandas time frequency string. Determines how many images to fetch for a given segment. Default: QS.")
    parser.add_argument("--overwrite-sentinel", action="store_true", help="Re-download existing Sentinel data.")
    parser.add_argument("--overwrite-cnes", action="store_true", help="Re-download existing CNES data.")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded segments. Don't overwrite.")

    args = parser.parse_args()
    assert args.overwrite_sentinel or args.overwrite_cnes, "You need to set at least one overwrite flag to True."

    load_dotenv()
    sh_config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    aoi = AOIs[args.aoi]
    data_dirs = DataDirs(aoi=args.aoi, map_type=args.labels)
    data_dirs.land_cover.mkdir(parents=True, exist_ok=True)
    data_dirs.sentinel.mkdir(parents=True, exist_ok=True)

    segments = calculate_segments(aoi, SEGMENT_LENGTH_KM)
    time_intervals = _split_up_time_interval(TIME_INTERVAL, frequency=args.frequency)

    # Load or initialize resume data
    resume_file = Path("resume_data.json")
    if args.resume and resume_file.exists():
        with open(resume_file, "r") as file:
            resume_data = json.load(file)
    else:
        resume_data = {"downloaded": []}

    for idx, segment in enumerate(tqdm(segments)):
        if args.resume and str(idx) in resume_data["downloaded"]:
            continue  # Skip this segment if resuming and it's already downloaded

        for time_interval in time_intervals:
            try:
                sentinel_file_path = data_dirs.sentinel / f"sentinel_{idx}.tif"
                cnes_file_path = data_dirs.land_cover / f"cnes_{idx}.tif"

                if args.overwrite_sentinel:
                    sentinel_data = fetch_sentinel_data(segment, sh_config, time_interval)
                    save_data_as_geotiff(sentinel_data, sentinel_file_path, segment, is_sentinel=True)

                if args.overwrite_cnes:
                    cnes_data = fetch_cnes_land_cover_data(segment, sh_config, time_interval)
                    save_data_as_geotiff(cnes_data, cnes_file_path, segment, is_sentinel=False)

                resume_data["downloaded"].append(str(idx))  # Mark this segment as downloaded

            except Exception as e:
                print(f"Error processing segment {idx} during {time_interval}: {e}")
                traceback.print_exc()

        # Save progress for resuming
        with open(resume_file, "w") as file:
            json.dump(resume_data, file, indent=4)


if __name__ == "__main__":
    main()
