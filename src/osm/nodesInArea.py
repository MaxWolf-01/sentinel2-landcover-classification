import argparse

import osmium
import json


class BoundingBoxHandler(osmium.SimpleHandler):
    """
    An osmium handler to extract nodes within a specified bounding box.

    Attributes:
        bbox (list): The bounding box coordinates as [south, west, north, east].
        nodes (list): Accumulates nodes that fall within the specified bounding box.
    """

    def __init__(self, bounding_box):
        """
        Initializes the BoundingBoxHandler with a bounding box.

        Args:
            bbox (list): The bounding box coordinates as [south, west, north, east].
        """
        super(BoundingBoxHandler, self).__init__()
        self.bbox = bounding_box
        self.nodes = []

    def node(self, n):
        """
        Processes each node in the osmium file. If the node is within the bounding box
        and has tags, it is added to the nodes list.

        Args:
            n (osmium.osm.Node): The node to process.
        """
        if (self.bbox[0] <= n.location.lat <= self.bbox[2] and
                self.bbox[1] <= n.location.lon <= self.bbox[3] and
                len(n.tags) > 0):
            self.nodes.append({
                "id": n.id,
                "lat": n.location.lat,
                "lon": n.location.lon,
                "tags": dict(n.tags)
            })

    def write_to_file(self, filename):
        """
        Writes the accumulated nodes within the bounding box to a JSON file.

        Args:
            filename (str): The name of the file to write the output to.
        """
        with open(filename, 'w') as f:
            json.dump(self.nodes, f, indent=4)


if __name__ == '__main__':
    # Define the bounding box [south, west, north, east]
    bbox = [48.140872, 15.738700, 48.341646, 15.945871]

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="austria-latest.osm.pbf",
                        help="Path to the OSM file to be processed. Default: 'austria-latest.osm.pbf'")
    parser.add_argument("--outfile", type=str, default="nodes_within_bbox.json",
                        help="Path to the output file. Default: 'nodes_within_bbox.json'")
    args = parser.parse_args()
    pbf_file = args.filename
    output_file = args.outfile

    handler = BoundingBoxHandler(bbox)
    handler.apply_file(pbf_file)
    handler.write_to_file(output_file)

    print(f"Nodes written to {output_file} in JSON format.")
