import json
import networkx as nx
from math import dist

#Define a function to calculate the center of catch points
def hold_centers(data):
    routes = {}
    for _,row in data.iterrows():
        image = row["filename"]
        shape = json.loads(row["region_shape_attributes"])
        x = shape["all_points_x"]
        y = shape["all_points_y"]

        center_x = sum(x) / len(x)
        center_y = sum(y) / len(y)

        if image not in routes:
            routes[image] = []
        
        routes[image].append((center_x,center_y))

    return routes


# Check the visualisation
def build_route_graph(route,threshold):
    G = nx.Graph()
    for i, (x1, y1) in enumerate(route):
        G.add_node(i, pos=(x1, y1))
        for j, (x2, y2) in enumerate(route):
            if i < j:
                dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if dist <= threshold:
                    G.add_edge(i, j)
    return G