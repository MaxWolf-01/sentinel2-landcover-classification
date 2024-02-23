from pathlib import Path
from typing import NamedTuple, Dict
import sentinelhub as sh


class BBox(NamedTuple):
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
        if aoi is None or map_type is None:
            raise ValueError("AOI and map_type must not be None.")

        base_path: Path = DATA_DIR / aoi / map_type
        self.sentinel: Path = base_path / "sentinel"
        self.osm: Path = base_path / "osm"
        self.land_cover: Path = base_path / "land_cover"
        print(base_path / "land_cover")


CRS: sh.CRS = sh.CRS.WGS84  # Coordinate Reference System
DATA_COLLECTION = sh.DataCollection.SENTINEL2_L1C  # Sentinel-2 Level 2A data collection
RESOLUTION: tuple[int, int] = (512, 512)  # Resolution in pixels (width, height)
SEGMENT_SIZE: int = 25  # Segment size in kilometers for data processing

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Area of Interests (AOIs)
AOIs: Dict[str, BBox] = {
    "fr": BBox(west=4.508514, south=45.477466, east=5.284424, north=45.897655),  # Lyon in france
    "vie": BBox(north=48.341646, south=47.739323, east=16.567383, west=15.117188),  # rough crop
    "test": BBox(north=48.980217, south=46.845164, east=17.116699, west=13.930664),  # VIE,NÖ,OÖ,NBGLD,Graz
    "at": BBox(north=49.009121, south=46.439861, east=17.523438, west=9.008164),  # AT + bits of neighbours
    "small": BBox(north=48.286391, south=48.195845, east=16.463699, west=16.311951),  # small area in VIE; 6 segments
}

TIME_INTERVAL: tuple[str, str] = (
    "2021-01-01",
    "2021-12-31",
)  # If using french lc data, check for availability: https://collections.sentinel-hub.com/cnes-land-cover-map/readme.html

# Sentinel-2 Bands
BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Label mappings, can be extended with actual mappings
LABEL_MAPPINGS: dict[str, dict] = {
    "multiclass": {},
    "binary": {},
}
