import typing

import numpy as np


class CnesLabelEntry(typing.TypedDict):
    color: str


CnesLabelMap = dict[str, CnesLabelEntry]

# CNES Land Cover nomenclature and colouring scheme
# https://collections.sentinel-hub.com/cnes-land-cover-map/readme.html
CNES_LABEL_MAP: CnesLabelMap = {
    "Dense built-up area": {"color": "#ff00ff"},
    "Diffuse built-up area": {"color": "#ff55ff"},
    "Industrial and commercial areas": {"color": "#ffaaff"},
    "Roads": {"color": "#00ffff"},
    "Oilseeds (Rapeseed)": {"color": "#ffff00"},
    "Straw cereals (Wheat, Triticale, Barley)": {"color": "#d0ff00"},
    "Protein crops (Beans / Peas)": {"color": "#a1d600"},
    "Soy": {"color": "#ffab44"},
    "Sunflower": {"color": "#d6d600"},
    "Corn": {"color": "#ff5500"},
    "Rice": {"color": "#c5ffff"},
    "Tubers/roots": {"color": "#aaaa61"},
    "Grasslands": {"color": "#aaaa00"},
    "Orchards and fruit growing": {"color": "#aaaaff"},
    "Vineyards": {"color": "#550000"},
    "Hardwood forest": {"color": "#009c00"},
    "Softwood forest": {"color": "#003200"},
    "Natural grasslands and pastures": {"color": "#aaff00"},
    "Woody moorlands": {"color": "#55aa7f"},
    "Natural mineral surfaces": {"color": "#ff0000"},
    "Beaches and dunes": {"color": "#ffb802"},
    "Glaciers and eternal snows": {"color": "#bebebe"},
    "Water": {"color": "#0000ff"},
}

_AGRICULTURE_KEY, _NATURE_KEY, _IMPERVIOUS_SURFAE_KEY = "agriculture", "nature", "impervious_surface"
_OTHER: CnesLabelMap = {"other": {"color": "#000000"}}
_AGRICULTURE: CnesLabelMap = {_AGRICULTURE_KEY: {"color": "#f5a142"}}
_NATURE: CnesLabelMap = {_NATURE_KEY: {"color": "#00ff00"}}
_IMPERVIOUS_SURFACE: CnesLabelMap = {_IMPERVIOUS_SURFAE_KEY: {"color": "#646464"}}
CNES_SIMPLIFIED_MULTICLASS: CnesLabelMap = _OTHER | _AGRICULTURE | _NATURE | _IMPERVIOUS_SURFACE
CNES_SIMPLIFIED_BINARY_IMPERVIOUS: CnesLabelMap = _OTHER | _IMPERVIOUS_SURFACE
CNES_SIMPLIFIED_BINARY_NATURE: CnesLabelMap = _OTHER | _NATURE
CNES_SIMPLIFIED_BINARY_AGRICULTURE: CnesLabelMap = _OTHER | _AGRICULTURE
CNES_TO_SIMPLIFIED: dict[int, str] = {
    1: _IMPERVIOUS_SURFAE_KEY,  # Dense built-up area
    2: _IMPERVIOUS_SURFAE_KEY,  # Diffuse built-up area
    3: _IMPERVIOUS_SURFAE_KEY,  # Industrial and commercial areas
    4: _IMPERVIOUS_SURFAE_KEY,  # Roads
    5: _AGRICULTURE_KEY,  # Oilseeds (Rapeseed)
    6: _AGRICULTURE_KEY,  # Straw cereals (Wheat, Triticale, Barley)
    7: _AGRICULTURE_KEY,  # Protein crops (Beans / Peas)
    8: _AGRICULTURE_KEY,  # Soy
    9: _AGRICULTURE_KEY,  # Sunflower
    10: _AGRICULTURE_KEY,  # Corn
    11: _AGRICULTURE_KEY,  # Rice
    12: _AGRICULTURE_KEY,  # Tubers/roots
    13: _NATURE_KEY,  # Grasslands
    14: _NATURE_KEY,  # Orchards and fruit growing
    15: _NATURE_KEY,  # Vineyards
    16: _NATURE_KEY,  # Hardwood forest
    17: _NATURE_KEY,  # Softwood forest
    18: _NATURE_KEY,  # Natural grasslands and pastures
    19: _NATURE_KEY,  # Woody moorlands
    20: _NATURE_KEY,  # Natural mineral surfaces
    21: _NATURE_KEY,  # Beaches and dunes
    22: _NATURE_KEY,  # Glaciers and eternal snows
    23: _NATURE_KEY,  # Water
}


def map_cnes_label_to_simplified_category(labels: np.ndarray, label_map: CnesLabelMap) -> np.ndarray:
    first_dimension_labels = labels[:, :, 0]  #TODO: Change if using multiple dimensions of CNES

    label_keys: list[str] = list(label_map.keys())

    def map_func(x: int) -> int:
        return label_keys.index(CNES_TO_SIMPLIFIED[x])

    mapped_labels = np.vectorize(map_func)(first_dimension_labels)
    return mapped_labels.astype(int)

