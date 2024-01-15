import json
import os
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sentinelhub import BBox, CRS, SentinelHubRequest, DataCollection, MimeType, SHConfig
from sklearn.preprocessing import LabelEncoder


def _create_evalscript(bands):
    bands_str = ", ".join([f'"{band}"' for band in bands])
    return f"""
    //VERSION=3
    function setup() {{
        return {{
            input: [{{
                bands: [{bands_str}],
                units: "DN"
            }}],
            output: {{
                bands: {len(bands)},
                sampleType: "INT16"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [{', '.join([f'sample.{band}' for band in bands])}];
    }}
    """


def fetch_osm_data_by_tags(class_label, segment, tags):
    """
    Fetch OSM data within the set bounds and with given feature tags, and assign class labels.

    Parameters:
        class_label (int): Class label to assign to the features.

    Returns:
        GeoDataFrame: GeoDataFrame with geometry and class label.
    """

    west, south, east, north = segment

    osm_data = ox.features_from_bbox(north=north, south=south, east=east, west=west, tags=tags)
    osm_data["class"] = class_label
    return osm_data[["geometry", "class"]]


def calculate_segments(min_lon, min_lat, max_lon, max_lat, segment_size_km):
    lon_diff = max_lon - min_lon
    lat_diff = max_lat - min_lat
    lon_segments = int(np.ceil(lon_diff / (segment_size_km / 111)))
    lat_segments = int(np.ceil(lat_diff / (segment_size_km / 111)))
    segments = []
    lon_segment_size = lon_diff / lon_segments
    lat_segment_size = lat_diff / lat_segments
    for i in range(lon_segments):
        for j in range(lat_segments):
            segment_min_lon = min_lon + i * lon_segment_size
            segment_max_lon = segment_min_lon + lon_segment_size
            segment_min_lat = min_lat + j * lat_segment_size
            segment_max_lat = segment_min_lat + lat_segment_size
            segments.append([segment_min_lon, segment_min_lat, segment_max_lon, segment_max_lat])
    return segments


def standardize_osm_tags(gdf, tag_mapping):
    #   for variant, standard in tag_mapping.items():
    #      gdf.loc[gdf['key'] == variant, 'key'] = standard
    # TODO: Handle tag-mapping
    return gdf


class DataGenerator:
    def __init__(self, config):
        self.bounds = None
        self.feature_tags = {
            "building": ["yes", "residential", "commercial", "industrial"],
            "highway": ["primary", "secondary", "tertiary", "residential"],
            "landuse": ["residential", "commercial", "industrial", "park"],
            "natural": ["water", "wood", "grassland"],
            "amenity": ["school", "hospital", "parking", "restaurant"],
        }

        self.gdf = None
        self.tag_mapping = {}
        self.config = config

    def set_bounds(self, north, south, east, west):
        self.bounds = (north, south, east, west)

    def set_feature_tags(self, feature_tags):
        self.feature_tags = feature_tags

    def load_tag_mapping(self, file_path):
        with open(file_path, "r") as file:
            self.tag_mapping = json.load(file)

    def fetch_and_store_data(self, aoi_bbox, time_interval, bands, resolution, segment_size_km=25):
        segments = calculate_segments(*aoi_bbox, segment_size_km)
        for idx, segment in enumerate(segments):
            self.fetch_and_store_sentinel_data(segment, time_interval, bands, resolution, idx)
            self.fetch_data(segment)
            self.rasterize_data(os.path.join("../data/osm/", f"osm_data_{idx}.tif"), *resolution)

    def fetch_and_store_sentinel_data(self, segment_bbox, time_interval, bands, resolution, idx):
        request = self.create_request(segment_bbox, time_interval, bands, resolution)
        data = request.get_data(save_data=False)

        # Assuming data has shape (1, Height, Width, Channels), reshape to (Channels, 1, Height, Width)
        reshaped_data = data[0].transpose(2, 0, 1)

        output_path = os.path.join("../data/sentinel/", f"sentinel_data_{idx}.tif")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=resolution[0],
            width=resolution[1],
            count=len(bands),
            dtype=reshaped_data.dtype,
            crs="+proj=latlong",
            transform=rasterio.transform.from_origin(*segment_bbox[:2], resolution[0], resolution[1]),
        ) as dst:
            for i in range(len(bands)):
                # Write each band separately
                dst.write(reshaped_data[i], i + 1)

    def create_request(self, aoi_bbox, time_interval, bands, resolution):
        evalscript = _create_evalscript(bands)
        bbox = BBox(bbox=aoi_bbox, crs=CRS.WGS84)
        return SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C, time_interval=time_interval)
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=resolution,
            config=self.config,
            data_folder="../data/sentinel/",
        )

    def fetch_data(self, segment):
        gdf_list = []
        for feature, tags in self.feature_tags.items():
            tags_dict = {feature: tags}  # TODO: Improve code efficiency
            gdf = fetch_osm_data_by_tags(class_label=feature, segment=segment, tags=tags_dict)
            gdf_list.append(gdf)

        self.gdf = pd.concat(gdf_list, ignore_index=True)
        self.gdf = standardize_osm_tags(self.gdf, self.tag_mapping)
        self.gdf = self.gdf.dropna(subset=["geometry"])

    def rasterize_data(self, output_path, tile_width, tile_height):
        if self.gdf is None:
            raise ValueError("GeoDataFrame is empty. Fetch data first.")

        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        pixel_size_x = (maxx - minx) / tile_width
        pixel_size_y = (maxy - miny) / tile_height
        transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

        label_encoder = LabelEncoder()
        self.gdf["class_encoded"] = label_encoder.fit_transform(
            self.gdf["class"]
        )  # FIXME why call this here, repeatedly?

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
            shapes = ((geom, value) for geom, value in zip(self.gdf.geometry, self.gdf["class_encoded"]))
            burned = rasterize(shapes=shapes, out_shape=(tile_height, tile_width), transform=transform, fill=0)
            dst.write_band(1, burned)

    def get_data_frame(self):  # FIXME never used
        if self.gdf is None:
            raise ValueError("GeoDataFrame is empty. Fetch data first.")
        return self.gdf

    def main(self):
        # Initialize Sentinel Hub configuration
        config = SHConfig(
            sh_client_id="6a27779e-becf-49f4-8c28-27f5fdabb0dc", sh_client_secret="dSc8LwCGLEFNfb15dGqheAQM9v2HyMfF"
        )

        # Set the bounds for Vienna (approximate)
        vienna_bounds = (15.117188, 47.739323, 16.567383, 48.341646)

        # Set the time interval for Sentinel-2 data
        time_interval = ("2023-07-01", "2023-07-15")  # Example time interval

        bands = ("B02", "B03", "B04", "B05", "B06", "B07")  # Prithvi Bands

        # Specify the resolution
        resolution = (512, 512)  # Width and Height in pixels

        # Initialize the DataGenerator
        data_generator = DataGenerator(config)

        # Load tag mapping from a JSON file (you need to create this file based on your requirements)
        data_generator.load_tag_mapping("tag_mapping.json")

        # Set feature tags for OSM data (you need to define these based on your requirements)
        self.feature_tags = {"building": "yes", "highway": ["primary", "secondary"]}

        # Fetch and store data for Vienna
        data_generator.fetch_and_store_data(vienna_bounds, time_interval, bands, resolution)


if __name__ == "__main__":
    # Initialize Sentinel Hub configuration
    config = SHConfig(sh_client_id=os.getenv("SH_CLIENT_ID"), sh_client_secret=os.getenv("SH_CLIENT_SECRET"))

    # Create an instance of the DataGenerator class with the configuration
    data_generator = DataGenerator(config)

    # Call the main function of the data generator
    data_generator.main()
