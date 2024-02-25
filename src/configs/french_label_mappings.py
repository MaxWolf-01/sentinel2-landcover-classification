import numpy as np
from matplotlib.colors import ListedColormap

# CNES Land Cover nomenclature and colouring scheme (https://collections.sentinel-hub.com/cnes-land-cover-map/readme.html)
CNES_LABEL_MAP = {
    1: {"color": "#ff00ff", "label": "Dense built-up area"},
    2: {"color": "#ff55ff", "label": "Diffuse built-up area"},
    3: {"color": "#ffaaff", "label": "Industrial and commercial areas"},
    4: {"color": "#00ffff", "label": "Roads"},
    5: {"color": "#ffff00", "label": "Oilseeds (Rapeseed)"},
    6: {"color": "#d0ff00", "label": "Straw cereals (Wheat, Triticale, Barley)"},
    7: {"color": "#a1d600", "label": "Protein crops (Beans / Peas)"},
    8: {"color": "#ffab44", "label": "Soy"},
    9: {"color": "#d6d600", "label": "Sunflower"},
    10: {"color": "#ff5500", "label": "Corn"},
    11: {"color": "#c5ffff", "label": "Rice"},
    12: {"color": "#aaaa61", "label": "Tubers/roots"},
    13: {"color": "#aaaa00", "label": "Grasslands"},
    14: {"color": "#aaaaff", "label": "Orchards and fruit growing"},
    15: {"color": "#550000", "label": "Vineyards"},
    16: {"color": "#009c00", "label": "Hardwood forest"},
    17: {"color": "#003200", "label": "Softwood forest"},
    18: {"color": "#aaff00", "label": "Natural grasslands and pastures"},
    19: {"color": "#55aa7f", "label": "Woody moorlands"},
    20: {"color": "#ff0000", "label": "Natural mineral surfaces"},
    21: {"color": "#ffb802", "label": "Beaches and dunes"},
    22: {"color": "#bebebe", "label": "Glaciers and eternal snows"},
    23: {"color": "#0000ff", "label": "Water"},
}


def get_cnes_color_map() -> ListedColormap:
    """Generates a color map for CNES Land Cover data."""
    colors = [entry["color"] for entry in CNES_LABEL_MAP.values()]
    return ListedColormap(colors)


def get_label_info() -> list[dict[str, str]]:
    """Returns a list of dictionaries containing label info for visualization and analysis."""
    return [{"id": key, "label": value["label"], "color": value["color"]} for key, value in CNES_LABEL_MAP.items()]


# Mapping of CNES labels to simplified categories
SIMPLIFIED_LABEL_MAP = {  # TODO: Clean up and clarify
    1: 0,  # Dense built-up area
    2: 0,  # Diffuse built-up area
    3: 0,  # Industrial and commercial areas
    4: 0,  # Roads
    5: 1,  # Oilseeds (Rapeseed)
    6: 1,  # Straw cereals (Wheat, Triticale, Barley)
    7: 1,  # Protein crops (Beans / Peas)
    8: 1,  # Soy
    9: 1,  # Sunflower
    10: 1,  # Corn
    11: 1,  # Rice
    12: 1,  # Tubers/roots
    13: 2,  # Grasslands
    14: 2,  # Orchards and fruit growing
    15: 2,  # Vineyards
    16: 2,  # Hardwood forest
    17: 2,  # Softwood forest
    18: 2,  # Natural grasslands and pastures
    19: 2,  # Woody moorlands
    20: 3,  # Natural mineral surfaces
    21: 3,  # Beaches and dunes
    22: 3,  # Glaciers and eternal snows
    23: 3,  # Water
}


# Function to map detailed labels to simplified categories
def map_labels_to_simplified_categories(labels: np.ndarray) -> np.ndarray:
    def map_func(x):
        return SIMPLIFIED_LABEL_MAP.get(x, 0)

    mapped_labels = np.vectorize(map_func)(labels)
    return mapped_labels.astype(int)
