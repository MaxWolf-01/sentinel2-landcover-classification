import json
import typing
from pathlib import Path
from typing import NamedTuple, Dict
import sentinelhub as sh

from src.configs.label_mappings import LabelMap, BINARY_MAP, MULTICLASS_MAP


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


class DataDirs:
    def __init__(self, aoi: str, map_type: str) -> None:
        self.base_path: Path = DATA_DIR / aoi / map_type
        self.sentinel: Path = self.base_path / "sentinel"
        self.osm: Path = self.base_path / "osm"
        self.land_cover: Path = self.base_path / "land_cover"
        print(self.base_path / "land_cover")

    @property
    def sentinel_files(self) -> dict[int, Path]:
        files = sorted(list(self.sentinel.glob("*.tif")), key=lambda path: tuple(map(int, path.stem.split("_"))))
        return {i: path for i, path in enumerate(files)}

    @property
    def osm_files(self) -> dict[int, Path]:
        return {int(path.stem): path for path in sorted(list(self.osm.glob("*.tif")), key=lambda path: int(path.stem))}

    @property
    def french_lc_files(self) -> dict[int, Path]:
        return {int(path.stem): path for path in
                sorted(list(self.land_cover.glob("*.tif")), key=lambda path: int(path.stem))}


CRS: sh.CRS = sh.CRS.WGS84  # Coordinate Reference System
DATA_COLLECTION = sh.DataCollection.SENTINEL2_L1C  # Sentinel-2 Level 2A data collection
RESOLUTION: tuple[int, int] = (512, 512)  # Resolution in pixels (width, height)
SEGMENT_SIZE: int = 25  # Segment size in kilometers for data processing

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

AOIs: dict[str, BBox] = {
    "vie": BBox(north=48.341646, south=47.739323, east=16.567383, west=15.117188),  # rough crop
    "test": BBox(north=48.980217, south=46.845164, east=17.116699, west=13.930664),  # VIE,NÖ,OÖ,NBGLD,Graz
    "at": BBox(north=49.009121, south=46.439861, east=17.523438, west=9.008164),  # AT + bits of neighbours
    "small": BBox(north=48.286391, south=48.195845, east=16.463699, west=16.311951),  # small area in VIE; 6 segments
    "fr": BBox(west=4.508514, south=45.477466, east=5.284424, north=45.897655),  # Lyon in france
}

DATA_COLLECTION = sh.DataCollection.SENTINEL2_L2A  # use atmosphericlly corrected data
# https://github.com/NASA-IMPACT/hls-foundation-os/issues/15#issuecomment-1667699259
BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12"]
EVALSCRIPT: str = f"""
    //VERSION=3
    function setup() {{
        return {{
            input: [{{
                bands: {json.dumps(BANDS)},
                units: "DN"
            }}],
            output: {{
                bands: {len(BANDS)},
                sampleType: "INT16"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [{', '.join([f'sample.{band}' for band in BANDS])}];
    }}
"""
CRS: sh.CRS = sh.CRS.WGS84  # == "4326" (EPSG)
TIME_INTERVAL: tuple[str, str] = ("2023-01-01", "2023-12-31")
SEGMENT_SIZE: tuple[int, int] = (512, 512)  # width and height of a single segment in pixels
SEGMENT_LENGTH_KM: float = 5.12  # 512*10m = 5.12km; 10m = lowest sentinel resolution
LABEL_MAPPINGS: dict[str, LabelMap] = {
    "multi": MULTICLASS_MAP,
    "binary": BINARY_MAP,
}
MAX_UNLABELED: float = 0.3  # Maximum percentage of unlabeled pixels in a segment

# Label mappings, can be extended with actual mappings
LABEL_MAPPINGS: dict[str, dict] = {
    "multiclass": {},
    "binary": {},
}
