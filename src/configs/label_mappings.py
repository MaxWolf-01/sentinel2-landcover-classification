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
            "natural": True,
            "leisure": [
                "dog_park",
                "garden",
                "nature_reserve",
                "park",
                "protected_area",
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
            "boundary": [
                "national_park",
                "protected_area",
            ],
            "wetland": [
                "bog",
                "fen",
                "marsh",
                "reedbed",
                "swamp",
            ],
            "surface": [
                "earth",
                "grass",
                "mud",
                "rock",
                "sand",
            ],
            "region": [
                "mountain_range",
                "natural_area",
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
            "building": True,
            "highway": True,
            "railway": True,
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
            "surface": [
                "asphalt",
                "cobblestone",
                "concrete",
                "metal",
                "paving_stones",
                "sett",
                "unhewn_cobblestone",
            ],
            "man_made": [
                "bridge",
                "pier",
                "reservoir_covered",
                "tower",
                "wastewater_plant",
                "water_works",
            ],
            "public_transport": [
                "platform",
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
            "power": [
                "substation",
                "transformer",
            ],
            "barrier": [
                "city_wall",
                "fence",
                "guard_rail",
                "kerb",
                "retaining_wall",
                "wall",
            ],
            "leisure": [
                "pitch",
                "sports_centre",
                "swimming_pool",
                "track",
            ],
        },
    },
    "agriculture": {
        "color": "#f5a142",
        "osm_tags": {
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
            "crop": True,
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
