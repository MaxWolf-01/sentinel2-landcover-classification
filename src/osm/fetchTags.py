import argparse
import osmium as o


class TagListHandler(o.SimpleHandler):
    """
    A handler to process tags from OpenStreetMap data. It filters and records
    unique tags based on a predefined set of keys.

    Attributes:
        file_full_path (str): Path to the file where full tag information (key=value) is stored.
        file_key_path (str): Path to the file where only tag keys are stored.
        tags_full (set): A set to hold unique full tags (key=value).
        tags_key (set): A set to hold unique tag keys.
        filter (set): A set containing tag keys to filter and process. Only tags with these keys are considered.
    """

    def __init__(self, file_full_path, file_key_path):
        super().__init__()
        self.file_full_path = file_full_path
        self.file_key_path = file_key_path
        self.tags_full = set()
        self.tags_key = set()
        self.filter = {
            "natural", "crossing", "highway", "bridge", "surface", "train", "covered",
            "overtaking", "waterway", "parking", "station", "amenity",
            "bus_bay", "stop", "agricultural", "subway", "bus_lines", "river", "building", "lanes", "bus_stop",
            "sidewalk", "forestry", "water_source", "lane_markings", "landuse", "landcover", "playground",
            "watermill", "wetland", "atmotorway", "monorail", "cover", "mount", "natural_protection",
            "crop", "water", "area", "backcountry", "watering_place", "raceway",
            "waterfall", "water_characteristic", "farmyard", "landfill", "wood_provided"
        }

    def node(self, n):
        """
        Processes each node in the OSM file, filtering and recording tags
        based on the predefined filter.

        Args:
            n (osmium.osm.Node): The node object from the OSM file.
        """
        for tag in n.tags:
            tag_key = tag.k.split(":")[0]
            if tag_key in self.filter and 'name' not in tag_key:
                tag_full = f"{tag.k}={tag.v}"
                if tag_full not in self.tags_full:
                    self.tags_full.add(tag_full)
                if tag_key not in self.tags_key:
                    self.tags_key.add(tag_key)

    def write_files(self):
        """
        Writes the collected unique full tags and tag keys to their respective files.
        """
        with open(self.file_full_path, "w") as f_full, open(self.file_key_path, "w") as f_key:
            for tag_full in sorted(self.tags_full):
                f_full.write(tag_full + "\n")
            for tag_key in sorted(self.tags_key):
                f_key.write(tag_key + "\n")


def main():
    """
    Main function to parse arguments and run the tag list handler on an OSM file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="austria-latest.osm.pbf",
                        help="Path to the OSM file to be processed. Default: 'austria-latest.osm.pbf'")
    args = parser.parse_args()

    # Initialize the handler with file paths for output
    handler = TagListHandler("../osm/unique_full_tags.txt", "../osm/unique_key_tags.txt")
    handler.apply_file(args.filename)
    handler.write_files()


if __name__ == "__main__":
    main()
