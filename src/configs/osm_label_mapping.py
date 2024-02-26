import typing

OSMTagMap = dict[str, str | bool]  # collection of osm tags, see parameter tags of osmnx.features.features_from_bbox


class OsmLabelEntry(typing.TypedDict):
    color: str
    osm_tags: OSMTagMap


# NOTE: The ordering of the top-level keys determines the priority of the classes (in case of label overlap)
#             -> last label in the dictionary has the highest priority (as it overwrites the previous ones)
OsmLabelMap = dict[str, OsmLabelEntry]

_OTHER: OsmLabelMap = {
    "other": {
        "color": "#000000",
        "osm_tags": {},
    }
}
_AGRICULTURE: OsmLabelMap = {
    "agriculture": {
        "color": "#f5a142",
        "osm_tags": {
            "crop": True,
            "landuse": [
                "agricultural",
                "agriculture",
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
    }
}
_NATURE: OsmLabelMap = {
    "nature": {
        "color": "#00ff00",
        "osm_tags": {
            "boundary": [
                "national_park",
                "protected_area",
            ],
            "landuse": [
                "allotments",
                "forest",
                "forestry",
                "grass",
                "greenfield",
                "meadow",
                "mountain_pass",
                "mountain_ridge",
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
    }
}

_IMPERVIOUS_SURFACE: OsmLabelMap = {
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
                "landfill",
                "military",
                "port",
                "quarry",
                "residential",
                "retail",
            ],
            "leisure": [
                "pitch",
                "swimming_pool",
                "track",
            ],
            "man_made": [
                "bridge",
                "pier",
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
            "waterway": [
                "dock",
                "lock_gate",
            ],
        },
    },
}
OSM_MULTICLASS: OsmLabelMap = _OTHER | _AGRICULTURE | _NATURE | _IMPERVIOUS_SURFACE
OSM_BINARY_IMPERVIOUS: OsmLabelMap = _OTHER | _IMPERVIOUS_SURFACE
OSM_BINARY_NATURE: OsmLabelMap = _OTHER | _NATURE
OSM_BINARY_AGRICULTURE: OsmLabelMap = _OTHER | _AGRICULTURE
