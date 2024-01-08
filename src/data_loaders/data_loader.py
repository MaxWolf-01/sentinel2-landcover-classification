import json

import osmnx as ox
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


class DataProvider:
    def __init__(self):
        self.bounds = None
        self.feature_tags = {}
        self.gdf = None
        self.tag_mapping = {}

    def set_bounds(self, north, south, east, west):
        self.bounds = (north, south, east, west)

    def set_feature_tags(self, feature_tags):
        self.feature_tags = feature_tags

    def fetch_data(self):
        if self.bounds is None:
            raise ValueError("Bounds are not set.")
        if not self.feature_tags:
            raise ValueError("Feature tags are not set.")

        gdf_list = []
        for feature, tags in self.feature_tags.items():
            gdf = self._fetch_osm_data_by_tags(tags)
            gdf["class"] = feature
            gdf_list.append(gdf)

        self.gdf = self.standardize_osm_tags(self.gdf, self.tag_mapping)
        self.gdf = pd.concat(gdf_list, ignore_index=True)
        self.gdf = self.gdf.dropna(subset=["geometry"])

    def load_tag_mapping(self, file_path):
        with open(file_path, "r") as file:
            self.tag_mapping = json.load(file)

    def standardize_osm_tags(self, gdf, tag_mapping):
        for variant, standard in tag_mapping.items():
            gdf.loc[gdf["key"] == variant, "key"] = standard
        return gdf

    def _fetch_osm_data_by_tags(self, tags):
        north, south, east, west = self.bounds
        return ox.features_from_bbox(north, south, east, west, tags=tags)

    def preprocess_data(self):
        # Implement any preprocessing steps here
        pass

    def rasterize_data(self, output_path, tile_width, tile_height):
        if self.gdf is None:
            raise ValueError("GeoDataFrame is empty. Fetch data first.")

        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        pixel_size_x = (maxx - minx) / tile_width
        pixel_size_y = (maxy - miny) / tile_height
        transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=tile_height,
            width=tile_width,
            count=1,
            dtype="uint8",
            crs=self.gdf.crs,
            transform=transform,
        ) as dst:
            shapes = ((geom, value) for geom, value in zip(self.gdf.geometry, self.gdf["class"]))
            burned = rasterize(shapes=shapes, out_shape=(tile_height, tile_width), transform=transform, fill=0)
            dst.write_band(1, burned)

    def get_data_frame(self):
        if self.gdf is None:
            raise ValueError("GeoDataFrame is empty. Fetch data first.")
        return self.gdf
