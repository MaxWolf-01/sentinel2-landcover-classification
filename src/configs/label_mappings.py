import typing

OSMTagMap = dict[str, str | bool]  # collection of osm tags, see parameter tags of osmnx.features.features_from_bbox


class LabelEntry(typing.TypedDict):
    color: str
    osm_tags: OSMTagMap


LabelMap = dict[str, LabelEntry]

MULTICLASS_MAP: LabelMap = {
    "other": {
        "color": "#000000",
        "osm_tags": {},
    },
    "nature": {
        "color": "#00ff00",
        "osm_tags": {
            "boundary": [
                "national_park",
                "protected_area",
            ],
            "landuse": [
                "forest",
                "forestry",
                "grass",
                "greenfield",
                "meadow",
                "mountain_pass",
                "mountain_ridge",
                "recreation_ground",
                "village_green",
            ],
            "leisure": [
                "dog_park",
                "garden",
                "nature_reserve",
                "park",
                "protected_area",
            ],
            "natural": True,
            "region": [
                "mountain_range",
                "natural_area",
            ],
            "surface": [
                "earth",
                "grass",
                "mud",
                "rock",
                "sand",
            ],
            "waterway": [
                "brook",
                "canal",
                "ditch",
                "drain",
                "river",
                "riverbank",
                "stream",
                "waterfall",
            ],
            "wetland": [
                "bog",
                "fen",
                "marsh",
                "reedbed",
                "swamp",
            ],
        },
    },
    "impervious_surface": {
        "color": "#646464",
        "osm_tags": {
            "aeroway": True,
            "amenity": [
                "parking",
                "parking_space",
            ],
            "barrier": [
                "city_wall",
                "fence",
                "guard_rail",
                "kerb",
                "retaining_wall",
                "wall",
            ],
            "building": True,
            "highway": True,
            "landuse": [
                "airport",
                "brownfield",
                "commercial",
                "construction",
                "depot",
                "garages",
                "impervious_surface",
                "industrial",
                "institutional",
                "landfill",
                "military",
                "port",
                "quarry",
                "residential",
                "retail",
            ],
            "leisure": [
                "pitch",
                "sports_centre",
                "swimming_pool",
                "track",
            ],
            "man_made": [
                "bridge",
                "pier",
                "reservoir_covered",
                "tower",
                "wastewater_plant",
                "water_works",
            ],
            "power": [
                "substation",
                "transformer",
            ],
            "public_transport": [
                "platform",
            ],
            "railway": True,
            "surface": [
                "asphalt",
                "cobblestone",
                "concrete",
                "metal",
                "paving_stones",
                "sett",
                "unhewn_cobblestone",
            ],
            "tourism": [
                "aquarium",
                "attraction",
                "theme_park",
                "zoo",
            ],
            "waterway": [
                "dock",
                "lock_gate",
            ],
        },
    },
    "agriculture": {
        "color": "#f5a142",
        "osm_tags": {
            "crop": True,
            "landuse": [
                "agricultural",
                "agriculture",
                "allotments",
                "animal_keeping",
                "farmland",
                "farmyard",
                "flowerbed",
                "orchard",
                "paddy",
                "salt_pond",
                "vineyard",
            ],
            "produce": [
                "cocoa",
                "coffee",
                "fiber",
                "flowers",
                "fruit",
                "grain",
                "herbs",
                "hop",
                "nuts",
                "oil",
                "rubber",
                "spices",
                "sugar",
                "tea",
                "tobacco",
                "vegetables",
                "vine",
            ],
        },
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
