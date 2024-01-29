import osmnx as ox
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin


def convert_tags_for_osmnx(feature_tags):
    """
    Convert a nested dictionary of feature tags into a flat dictionary suitable for OSMnx.

    Args:
    - feature_tags (dict): A nested dictionary of feature tags.

    Returns:
    - dict[str, str]: A flat dictionary with tag keys and string values.
    """
    osmnx_tags = {}
    for feature_category, tags_dict in feature_tags.items():
        for tag_key, tag_values in tags_dict.items():
            for tag_value in tag_values:
                # Each tag value is appended to the key with a semicolon to separate multiple values
                if tag_key in osmnx_tags:
                    osmnx_tags[tag_key] += ";" + tag_value
                else:
                    osmnx_tags[tag_key] = tag_value
    return osmnx_tags


def split_bounding_box(north, south, east, west, num_splits=2):
    """
    Split a bounding box into smaller boxes.
    """
    latitudes = np.linspace(south, north, num_splits + 1)
    longitudes = np.linspace(west, east, num_splits + 1)
    smaller_boxes = []
    for i in range(num_splits):
        for j in range(num_splits):
            s_north = float(latitudes[i + 1])
            s_south = float(latitudes[i])
            s_east = float(longitudes[j + 1])
            s_west = float(longitudes[j])
            smaller_boxes.append((s_north, s_south, s_east, s_west))
    return smaller_boxes

def fetch_osm_data_by_tags(
        i_north: float, i_south: float, i_east: float, i_west: float, tags: dict[str, str], class_label: int
) -> gpd.GeoDataFrame:
    """
    Fetch OSM data within the specified bounding box and with given tags, and assign class labels.

    Parameters:
        i_north (float): Northern latitude of bounding box.
        i_south (float): Southern latitude of bounding box.
        i_east (float): Eastern longitude of bounding box.
        i_west (float): Western longitude of bounding box.
        i_tags (dict[str, str]): Tags to filter OSM features.
        class_label (int): Class label to assign to the features.

    Returns:
        GeoDataFrame: GeoDataFrame with geometry and class label.
    """
    osm_data = ox.features_from_bbox(north=i_north, south=i_south, east=i_east, west=i_west, tags=tags)
    osm_data["class"] = class_label
    # Return the data with only relevant columns
    return osm_data[["geometry", "class"]]


# north, south, east, west = 46.95, 46.85, 9.7, 9.6  # Example coordinates
north, south, east, west = 48.3230, 48.1230, 16.5800, 16.2000

# Tags for different features and their corresponding class labels
feature_tags = {
    "water": {
        "natural": ["water", "wetland", "waterfall"],
        "waterway": ["river", "stream", "canal", "drain", "ditch"]
    },
    "sealed": {
        "highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"],
        "building": ["yes", "commercial", "industrial", "residential"],
        "landuse": ["industrial", "commercial", "residential", "retail"],
        "aeroway": ["aerodrome"],
        "amenity": ["parking"],
    },
    "green_space": {
        "landuse": ["forest", "grass", "recreation_ground", "village_green", "meadow", "forestry"],
        "natural": ["wood", "scrub", "beach", "natural_protection", "tree"],
        "leisure": ["park", "garden"]
    },
    "agriculture": {
        "landuse": ["farmland", "farmyard", "vineyard", "orchard", "agricultural"]
    }
}

# asign indecies to each class
feature_classes = {feature: idx for idx, feature in enumerate(feature_tags.keys(), start=1)}

# List to hold GeoDataFrames for each feature class
gdf_list = []

# Split the bounding box
smaller_bboxes = split_bounding_box(north, south, east, west, num_splits=8)

converted_tags = convert_tags_for_osmnx(feature_tags)

for small_north, small_south, small_east, small_west in smaller_bboxes:
    # Fetch data for each feature class and append to the list
    for feature, tags in feature_tags.items():
        gdf = fetch_osm_data_by_tags(small_north, small_south, small_east, small_west, tags, feature_classes[feature])
        gdf_list.append(gdf)

# Concatenate all the GeoDataFrames into one
combined_gdf: gpd.GeoDataFrame = pd.concat(gdf_list, ignore_index=True)  # type: ignore

# Drop NaN geometries if any
combined_gdf = combined_gdf.dropna(subset=["geometry"])

def rasterize_gdf(gdf: gpd.GeoDataFrame, output_path: str, tile_width: int, tile_height: int) -> None:
    """
    Rasterize the geometries in the GeoDataFrame into a GeoTIFF file.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with geometry and class label.
        output_path (str): Path to the output GeoTIFF file.
        tile_width (int): Width of the output raster in pixels
        tile_height (int): Height of the output raster in pixels.
    """

    # Calculate bounds of the combined GeoDataFrame
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Calculate pixel size
    pixel_size_x = (maxx - minx) / tile_width
    pixel_size_y = (maxy - miny) / tile_height

    # Create a transform
    transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    # Define a scaling factor for the class labels for better visibility
    scale_factor = 255 // gdf['class'].max()  # TODO remove this for production usage!!

    # Rasterize the GeoDataFrame
    with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=tile_height,
            width=tile_width,
            count=1,
            dtype="uint8",
            crs=gdf.crs,
            transform=transform,
    ) as dst:
        # Rasterize each geometry into the raster
        # 'shapes' is an iterable of (geometry, value) pairs
        shapes = ((geom, value * scale_factor) for geom, value in zip(gdf.geometry, gdf["class"]))

        # Rasterize the shapes with values into the raster
        burned = rasterize(shapes=shapes, out_shape=(tile_height, tile_width), transform=transform, fill=0)
        dst.write_band(1, burned)


# Example usage of the rasterization function
rasterize_gdf(combined_gdf, "output_raster.tif", tile_width=512, tile_height=512)