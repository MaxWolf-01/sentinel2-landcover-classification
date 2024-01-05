import osmnx as ox
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


def fetch_osm_data_by_tags(
        north: float, south: float, east: float, west: float, tags: dict[str, str], class_label: int
) -> gpd.GeoDataFrame:
    """
    Fetch OSM data within the specified bounding box and with given tags, and assign class labels.

    Parameters:
        north (float): Northern latitude of bounding box.
        south (float): Southern latitude of bounding box.
        east (float): Eastern longitude of bounding box.
        west (float): Western longitude of bounding box.
        tags (dict[str, str]): Tags to filter OSM features.
        class_label (int): Class label to assign to the features.

    Returns:
        GeoDataFrame: GeoDataFrame with geometry and class label.
    """
    osm_data = ox.features_from_bbox(north=north, south=south, east=east, west=west, tags=tags)
    osm_data["class"] = class_label
    # Return the data with only relevant columns
    return osm_data[["geometry", "class"]]

# north, south, east, west = 46.95, 46.85, 9.7, 9.6  # Example coordinates
north = 48.3230
south = 48.1230
east = 16.5800
west = 16.2000

# Tags for different features and their corresponding class labels
feature_tags = {
    "building": {
        "building": True
    },
    "low_vegetation": {
        "landuse": ["grass", "meadow"]
    },
    "high_vegetation": {
        "natural": ["tree", "wood"],
        "landuse": "forest"
    },
    "water": {
        "natural": "water",
        "waterway": True,
        "landuse": ["reservoir", "basin"],
        "amenity": ["fountain", "swimming_pool"]
    },
    "road": {
        "highway": True
    },
    # "railway": {
    #     "railway": True
    # },
}


# asign indecies to each class
feature_classes = {feature: idx for idx, feature in enumerate(feature_tags.keys(), start=1)}

# List to hold GeoDataFrames for each feature class
gdf_list = []

# Fetch data for each feature class and append to the list
for feature, tags in feature_tags.items():
    gdf = fetch_osm_data_by_tags(north, south, east, west, tags, feature_classes[feature])
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
    scale_factor = 255 // gdf['class'].max() # TODO remove this for production ussage!!

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
