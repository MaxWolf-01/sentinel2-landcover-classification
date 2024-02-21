import argparse

import osmium as o
import json
from src.configs.label_mappings import MULTICLASS_MAP


#
class RelationFilterHandler(o.SimpleHandler):
    def __init__(self, bbox, filter_by_tags):
        super(RelationFilterHandler, self).__init__()
        self.bbox = bbox
        self.filter_by_tags = filter_by_tags
        self.data = []
        self.label_map = MULTICLASS_MAP

    def relation(self, r):
        # Placeholder for your relation_in_bbox logic
        if self.relation_in_bbox(r):
            tags = dict(r.tags)
            for category, details in self.label_map.items():
                if self.matches_tags(tags, details["osm_tags"]):
                    members = [{"type": m.type, "ref": m.ref, "role": m.role} for m in r.members]
                    self.data.append({
                        "type": "relation",
                        "id": r.id,
                        # "members": members,
                        "tags": tags,
                        "category": category  # Optionally classify the relation
                    })
                    break  # Assuming you want to classify into the first matching category

    def matches_tags(self, tags, category_tags):
        for key, values in category_tags.items():
            if key in tags:
                if values is True:  # If the value is True, the presence of the key is enough
                    return True
                elif isinstance(values, list) and tags[key] in values:
                    return True
        return False

    def relation_in_bbox(self, relation):
        # Implement your bounding box logic here
        return True  # Placeholder logic

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract relations from OSM data within a bounding box.")

    parser.add_argument("--filename", type=str, default="austria-latest.osm.pbf",
                        help="Path to the OSM file to be processed.")
    parser.add_argument("--outfile", type=str, default="osm_data.json",
                        help="Path to the output file.")
    parser.add_argument("--filter-tags", action="store_true",
                        help="Filter the output by tags.")

    args = parser.parse_args()

    # Bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox = [15.2207735, 48.0404845, 15.324359, 48.140872]

    handler = RelationFilterHandler(bbox=bbox, filter_by_tags=args.filter_tags)
    handler.apply_file(args.filename, locations=True)
    handler.write_to_file(args.outfile)

    print(f"Relations written to {args.outfile} in JSON format.")
