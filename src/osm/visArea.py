import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Define the bounding box
north, south, east, west = 48.140872, 48.0404845, 15.324359, 15.2207735

# Create a polygon from the bounding box
bbox_polygon = Polygon([(west, north), (east, north), (east, south), (west, south)])

# Download the network data from OSM within the specified bounding box
graph = ox.graph_from_polygon(bbox_polygon, network_type='all')

# Extract nodes and edges
nodes, edges = ox.graph_to_gdfs(graph)

# Plot the borders
fig, ax = plt.subplots(figsize=(12, 8))
edges.plot(ax=ax, linewidth=1, edgecolor='blue')

# Set the axes to equal to avoid distortion
ax.set_aspect('equal')
plt.tight_layout()

# Show the plot
plt.show()
