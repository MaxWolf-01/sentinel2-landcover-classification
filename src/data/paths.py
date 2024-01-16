from pathlib import Path

BASE_DIR: Path = Path(__file__).parent  # src/data
DATA_DIR: Path = BASE_DIR.parent.parent / "data"  # data
SENTINEL_DIR: Path = DATA_DIR / "sentinel"
OSM_DIR: Path = DATA_DIR / "osm"
