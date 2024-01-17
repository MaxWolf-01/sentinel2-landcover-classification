import typing

OSMTagMap = dict[str, str | bool]  # collection of osm tags, see parameter tags of osmnx.features.features_from_bbox


class LabelEntry(typing.TypedDict):
    idx: int
    color: str
    osm_tags: OSMTagMap


LabelMap = dict[str, LabelEntry]

# class labels as in Multiclass Semantic Segmentation with Very High-Resolution Satellite Images | doi:10.2760/46796
# OSM mapping might differ from the one used in the paper (todo maybe authors can share this detail actually?)
GENERAL_MAP: LabelMap = {
    "building": {
        "idx": 1,
        "color": "#ff0000",
        "osm_tags": {
            "building": True,
        },
    },
    "low_vegetation": {
        "idx": 2,
        "color": "#00ff00",
        "osm_tags": {
            "landuse": "grass",
            "leisure": "garden",
            "natural": "grassland",
        },
    },
    "high_vegetation": {
        "idx": 3,
        "color": "#00ff00",
        "osm_tags": {
            "natural": "wood",
            "landuse": "forest",
        },
    },
    "water": {
        "idx": 4,
        "color": "#006432",
        "osm_tags": {
            "natural": "water",
            "waterway": True,
        },
    },
    "impervious_surface": {
        "idx": 5,
        "color": "#ffff00",
        "osm_tags": {
            "highway": True,
            "surface": "paved",
        },
    },
    "railway": {
        "idx": 5,
        "color": "#ff00ff",
        "osm_tags": {
            "railway": True,
        },
    },
}
OTHER_LABEL_ENTRY: LabelEntry = {
    "idx": 0,
    "color": "#000000",
    "osm_tags": {},
}

# todo add other, more specific configs, e.g. specialized for soil sealing, ...
