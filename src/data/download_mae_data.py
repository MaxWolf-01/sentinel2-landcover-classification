"""Theoretically, we could also simply use the download_s2_osm_data.py script, but let's keep it separate for now, so we could
e.g. fine-tune the mae on a larger dataset"""

import argparse
import concurrent
import os
import traceback
from pathlib import Path

import numpy.typing as npt
import sentinelhub as sh
from dotenv import load_dotenv
from tqdm import tqdm

from src.configs.paths import DATA_DIR
from src.data.download_s2_osm_data import AOIs, BBox, calculate_segments, fetch_sentinel_data, save_sentinel_data_as_geotiff

TIME_INTERVAL: tuple[str, str] = ("2023-07-01", "2023-07-15")  # TODO find suitable time interval
# https://github.com/NASA-IMPACT/hls-foundation-os/issues/15#issuecomment-1667699259
BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12"]
RESOLUTION: tuple[int, int] = (512, 512)  # Width and Height in pixels
SEGMENT_SIZE: int = 25  # Size of segments in km


def get_mae_data_dir(aoi: str) -> Path:
    return DATA_DIR / "mae" / aoi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=str, default="vie", help=f"Specify an AOI. Default: VIE. Available:{list(AOIs)}")
    parser.add_argument("--workers", type=int, default=1, help="Specify the number of workers.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing.")
    args = parser.parse_args()

    load_dotenv()
    config = sh.SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    data_dir: Path = get_mae_data_dir(aoi=args.aoi)
    data_dir.mkdir(parents=True, exist_ok=True)

    segments: list[BBox] = calculate_segments(bbox=AOIs[args.aoi], segment_size_km=SEGMENT_SIZE)

    pool_executor = (
        concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        if args.parallel
        else concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    )
    with pool_executor as executor:
        future_to_segment = {
            executor.submit(_process_segment, segment, idx, config, data_dir): (idx, segment)
            for idx, segment in enumerate(segments)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_segment), total=len(future_to_segment)):
            idx, segment = future_to_segment[future]
            try:
                future.result()
            except Exception as e:
                print(f"Segment {segment} idx:{idx} generated an exception:\n{traceback.format_exc()}")
                if "terminated abruptly" in str(e):
                    print(
                        "You might have run out of RAM. "
                        "Try lowering the number of workers via the --workers argument."
                        "Or don't use the --parallel flag to use threads instead of processes."
                    )


def _process_segment(segment: BBox, idx: int, sh_config: sh.SHConfig, data_dir: Path) -> None:
    sentinel_data: list[npt.NDArray] = fetch_sentinel_data(segment=segment, sh_config=sh_config)  # TODO update
    save_sentinel_data_as_geotiff(sentinel_data, idx=idx, aoi=segment, sentinel_dir=data_dir)


if __name__ == "__main__":
    main()
