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
        "idx": 2,
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
        "idx": 3,
        "color": "#f5a142",
        "osm_tags": {"landuse": ["farmland", "farmyard", "vineyard", "orchard", "agricultural"]},
    },
}

# todo add other, more specific configs, e.g. specialized for soil sealing, ...
