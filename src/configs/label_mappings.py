import typing

OSMTagMap = dict[str, str | bool]  # collection of osm tags, see parameter tags of osmnx.features.features_from_bbox


class LabelEntry(typing.TypedDict):
    idx: int
    color: str
    osm_tags: OSMTagMap


LabelMap = dict[str, LabelEntry]

MULTICLASS_MAP: LabelMap = {
    "other": {
        "color": "#000000",
        "osm_tags": {},
    },
    "nature": {
        "idx": 1,
        "color": "#00ff00",
        "osm_tags": {
            "landuse": [
                "forest",
                "grass",
                "recreation_ground",
                "village_green",
                "meadow",
                "forestry",
                "mountain_ridge",
                "mountain_pass",
            ],
            "natural": True,
            "leisure": ["park", "garden"],
        },
    },
    "impervious_surface": {
        "color": "#646464",
        "osm_tags": {
            "highway": True,
            "building": True,
            "railway": True,
            "landuse": ["industrial", "commercial", "residential", "retail"],
            "aeroway": ["aerodrome"],  # TODO could we set true?
            "amenity": ["parking"],
        },
    },
    "agriculture": {
        "color": "#f5a142",
        "osm_tags": {"landuse": ["farmland", "farmyard", "vineyard", "orchard", "agricultural"]},
    },
}

BINARY_MAP: LabelMap = {
    "other": MULTICLASS_MAP["other"],
    "impervious_surface": MULTICLASS_MAP["impervious_surface"],
}

MAPS = {
    "binary": BINARY_MAP,
    "multiclass": MULTICLASS_MAP,
}
