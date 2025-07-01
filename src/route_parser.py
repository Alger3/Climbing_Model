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


# TODO: Need to complicated the algorithmn
# assign the label to each athletes
def is_route_feasible(climber,points):
    height = climber["height"]
    ape_index = climber["ape_index"]
    weight = climber["weight"]
    flexibility = climber["flexibility"]
    strength = climber["strength"]

    # Add condition to judge complete or not
    reach = height * ape_index
    flex_bonus = (flexibility/10)*0.15
    strength_bonus = (strength/100)*0.2
    weight_punish = weight*0.05

    max_reach = reach * (1 + strength_bonus + flex_bonus) - weight_punish

    for i in range(len(points)-1):
        if dist(points[i],points[i+1]) > max_reach:
            return False

    return True