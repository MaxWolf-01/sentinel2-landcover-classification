import typing

OSMTagMap = dict[str, str | bool]  # collection of osm tags, see parameter tags of osmnx.features.features_from_bbox


class LabelEntry(typing.TypedDict):
    idx: int
    color: str
    osm_tags: OSMTagMap


LabelMap = dict[str, LabelEntry]


def get_idx_to_label_map(label_map: LabelMap) -> dict[int, str]:
    return {entry["idx"]: label for label, entry in label_map.items()}


# class labels as in Multiclass Semantic Segmentation with Very High-Resolution Satellite Images | doi:10.2760/46796
# OSM mapping might differ from the one used in the paper (todo maybe authors can share this detail actually?)
GENERAL_MAP: LabelMap = {
    "other": {
        "idx": 0,
        "color": "#000000",
        "osm_tags": {},
    },
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
        "color": "#006432",
        "osm_tags": {
            "natural": "wood",
            "landuse": "forest",
        },
    },
    "water": {
        "idx": 4,
        "color": "#0032fa",
        "osm_tags": {
            "natural": "water",
            "waterway": True,
        },
    },
    "impervious_surface": {
        "idx": 5,
        "color": "#646464",
        "osm_tags": {
            "highway": True,
            "surface": "paved",
        },
    },
    "railway": {
        "idx": 6,
        "color": "#c864c8",
        "osm_tags": {
            "railway": True,  # todo this seems to braod for 10m sentinel resuolution?
        },
    },
}

# todo add other, more specific configs, e.g. specialized for soil sealing, ...
