import json
import os

from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, SHConfig
import numpy as np
import overpy

class DataLoader:

    def __init__(self, client_id="6a27779e-becf-49f4-8c28-27f5fdabb0dc", client_secret="dSc8LwCGLEFNfb15dGqheAQM9v2HyMfF"):
        self.config = SHConfig()

        if client_id and client_secret:
            self.config.sh_client_id = client_id
            self.config.sh_client_secret = client_secret
        else:
            raise ValueError("Missing Sentinel Hub client ID and secret.")

    def create_request(self, aoi_bbox, time_interval, bands, resolution):
        evalscript = self._create_evalscript(bands)
        bbox = BBox(bbox=aoi_bbox, crs=CRS.WGS84)
        return SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C,
                                                      time_interval=time_interval)],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=resolution,
            config=self.config
        )

    def _create_evalscript(self, bands):
        bands_str = ', '.join([f'"{band}"' for band in bands])
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
        segments = []
        # Generate bounding box coordinates for each segment
        for i in range(lon_segments):
            for j in range(lat_segments):
                segment_min_lon = min_lon + i * (segment_size_km / 111)
                segment_max_lon = min(segment_min_lon + (segment_size_km / 111), max_lon)
                segment_min_lat = min_lat + j * (segment_size_km / 111)
                segment_max_lat = min(segment_min_lat + (segment_size_km / 111), max_lat)

                segments.append([segment_min_lon, segment_min_lat, segment_max_lon, segment_max_lat])

        return segments

    def fetch_osm_data(self, bbox):
        """
        Fetch OSM data for a given bounding box.

        Args:
        - bbox (list): A list of [min_lon, min_lat, max_lon, max_lat].

        Returns:
        - result: The OSM data.
        """
        api = overpy.Overpass()
        query = f"""
            [out:json];
            (
                way["building"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                way["highway"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                way["landuse"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                // Add more OSM features if needed
            );
            out body;
            >;
            out skel qt;
        """
        result = api.query(query)
        return result #TODO: Not serialized yet.

    def fetch_and_save_all_segments(self, segments, time_interval, bands, resolution, output_dir):
        for i, segment in enumerate(segments):
            print(f"Fetching data for segment {i + 1}/{len(segments)}...")
            # Fetch Sentinel-2 data
            request = self.create_request(segment, time_interval, bands, resolution)
            data = request.get_data()

            # Fetch OSM data
            osm_result = self.fetch_osm_data(segment)

            # Process OSM data
            osm_data = []
            for element in osm_result.ways:
                # Example: Extracting basic information from each way
                osm_data.append({
                    "id": element.id,
                    "tags": element.tags,
                    "nodes": [node.id for node in element.nodes]
                })
            # Add processing for other types (nodes, relations) if needed

            # Process and save both Sentinel-2 and OSM data
            segment_dir = os.path.join(output_dir, f"segment_{i}")
            os.makedirs(segment_dir, exist_ok=True)

            # Save Sentinel-2 data
            np.save(os.path.join(segment_dir, "sentinel_data.npy"), data)

            # Save OSM data
            with open(os.path.join(segment_dir, "osm_data.json"), 'w') as f:
                json.dump(osm_data, f)

            print(f"Data for segment {i + 1} saved.")


def calculate_segments(min_lon, min_lat, max_lon, max_lat, segment_size_km=25):
    """
    Calculate segments (bounding boxes) for an area of interest (AOI) to fit within the resolution limits.

    Args:
    - min_lon (float): Minimum longitude of the AOI.
    - min_lat (float): Minimum latitude of the AOI.
    - max_lon (float): Maximum longitude of the AOI.
    - max_lat (float): Maximum latitude of the AOI.
    - segment_size_km (int): Size of each segment in kilometers. Default is 25 km.

    Returns:
    - List of segments (bounding boxes) where each segment is represented as [min_lon, min_lat, max_lon, max_lat].
    """

    # Calculate the number of segments needed horizontally and vertically
    lon_diff = max_lon - min_lon  # Difference in longitude
    lat_diff = max_lat - min_lat  # Difference in latitude

    # Approximate conversion of longitude and latitude differences to kilometers
    # 111 kilometers is roughly the distance represented by one degree of latitude.
    lon_segments = int(np.ceil(lon_diff / (segment_size_km / 111)))  # Number of longitudinal segments
    lat_segments = int(np.ceil(lat_diff / (segment_size_km / 111)))  # Number of latitudinal segments

    # Initialize an empty list to store segments
    segments = []

    # Calculate the size of each segment in degrees
    lon_segment_size = lon_diff / lon_segments
    lat_segment_size = lat_diff / lat_segments

    # Generate the segments
    for i in range(lon_segments):
        for j in range(lat_segments):
            segment_min_lon = min_lon + i * lon_segment_size
            segment_max_lon = segment_min_lon + lon_segment_size
            segment_min_lat = min_lat + j * lat_segment_size
            segment_max_lat = segment_min_lat + lat_segment_size
            segments.append([segment_min_lon, segment_min_lat, segment_max_lon, segment_max_lat])

    return segments


def main():
    # Initialize DataLoader with Sentinel Hub client ID and secret
    data_loader = DataLoader()
    #AOI
    min_lon, min_lat, max_lon, max_lat = 9.5, 46.4, 17.2, 49.0

    # Calculate segments for the AOI
    segment_size_km = 25  # Size of each segment in kilometers
    segments = calculate_segments(min_lon, min_lat, max_lon, max_lat, segment_size_km)
    print(f"Number of segments: {len(segments)}")

    print(segments[1])
    # Define Sentinel-2 parameters
    time_interval = ('2023-01-01', '2023-01-31')
    bands = ["B01", "B02", "B03", "B04"]
    resolution = (512, 512)

    output_dir = '../data/experiments'  # Replace with your directory

    data_loader.fetch_and_save_all_segments(segments, time_interval, bands, resolution, output_dir)


if __name__ == "__main__":
    main()


