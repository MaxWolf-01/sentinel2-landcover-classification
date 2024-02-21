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
                "forestry",
                "mountain_ridge",
                "mountain_pass",
                "greenfield",
            ],
            "natural": True,
            "leisure": ["park", "garden", "nature_reserve", "dog_park", "protected_area"],
            "waterway": ["river", "stream", "brook", "canal", "drain", "ditch", "riverbank", "waterfall"],
            "boundary": ["national_park", "protected_area"],
            "wetland": ["bog", "swamp", "marsh", "reedbed", "fen"],
            "surface": ["grass", "earth", "mud", "sand", "rock"],
            "region": ["natural_area", "mountain_range"],
        },
    },
    "impervious_surface": {
        "color": "#646464",
        "osm_tags": {
            "highway": True,
            "building": True,
            "railway": True,
            "landuse": [
                "industrial",
                "commercial",
                "residential",
                "retail",
                "construction",
                "impervious_surface",
                "garages",
                "institutional",
                "brownfield",
                "depot",
                "landfill",
                "quarry",
                "military",
                "port",
                "airport",
            ],
            "aeroway": True,
            "amenity": ["parking", "parking_space"],
            "surface": ["asphalt", "concrete", "paving_stones", "sett", "unhewn_cobblestone", "cobblestone", "metal"],
            "man_made": ["pier", "wastewater_plant", "water_works", "bridge", "tower", "reservoir_covered"],
            "public_transport": ["platform"],
            "tourism": ["attraction", "theme_park", "zoo", "aquarium"],
            "waterway": ["lock_gate", "dock"],
            "power": ["substation", "transformer"],
            "barrier": ["wall", "fence", "retaining_wall", "kerb", "guard_rail", "city_wall"],
            "leisure": ["pitch", "track", "sports_centre", "swimming_pool"],
        },
    },
    "agriculture": {
        "color": "#f5a142",
        "osm_tags": {
            "landuse": [
                "farmland",
                "farmyard",
                "vineyard",
                "orchard",
                "agricultural",
                "agriculture",
                "allotments",
                "paddy",
                "meadow",
                "animal_keeping",
                "flowerbed",
                "salt_pond",
            ],
            "leisure": ["garden"],
            "produce": [
                "grain",
                "vegetables",
                "fruit",
                "nuts",
                "oil",
                "hop",
                "vine",
                "tea",
                "tobacco",
                "cocoa",
                "coffee",
                "sugar",
                "rubber",
                "fiber",
                "flowers",
                "herbs",
                "spices",
            ],
# TODO simply set `crop: True?`
            "crop": [
                "wheat",
                "corn",
                "rice",
                "soybean",
                "barley",
                "oat",
                "rye",
                "potato",
                "cassava",
                "sugarcane",
                "sunflower",
                "cotton",
                "tea",
                "coffee",
                "cocoa",
                "tobacco",
                "hemp",
                "linen",
                "olives",
                "grapes",
                "apples",
                "oranges",
                "lemons",
                "bananas",
                "mangoes",
                "tomatoes",
                "carrots",
                "onions",
                "lettuce",
                "pumpkins",
                "nuts",
                "beans",
                "peas",
                "spices",
                "herbs",
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
