import json
import random
import networkx as nx
import numpy as np
import math
from math import dist
import matplotlib.pyplot as plt

# Suppose PIXEL TO CM
PIXEL_TO_CM = 0.45

def pixel_dist(p1,p2):
    dx = (p1[0]-p2[0])*PIXEL_TO_CM
    dy = (p1[1]-p2[1])*PIXEL_TO_CM
    return np.sqrt(dx**2+dy**2)

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
def build_wall_graph(wall,threshold):
    G = nx.Graph()
    for i, (x1, y1) in enumerate(wall):
        G.add_node(i, pos=(x1, y1))
        for j, (x2, y2) in enumerate(wall):
            if i < j:
                dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if dist <= threshold:
                    G.add_edge(i, j)
    return G

# Sort climbing points
def sort_climbing_points(wall):
    return sorted(wall,key=lambda p:(p[1],p[0]))

# Simulate a route from wall (pick from bottom to top)
def simulate_route_from_wall(wall_points,max_reach,num_holds):
    route = []
    used = set()

    # Pick an random point which is lower than 20% height
    y_threshold = sorted([p[1] for p in wall_points], reverse=True)[int(len(wall_points) * 0.2)]
    bottom_candidates = [p for p in wall_points if p[1] >= y_threshold]
    if not bottom_candidates:
        return []

    current = random.choice(bottom_candidates)
    route.append(current)
    used.add(current)

    while len(route) < num_holds:
        candidates = [
            p for p in wall_points
            if p not in used and dist(current, p) <= max_reach and p[1] < current[1]
        ]
        if not candidates:
            break
        next_point = random.choice(candidates)
        route.append(next_point)
        used.add(next_point)
        current = next_point

    return route

# Simulate a route from wall (pick from each height range)
def simulate_route_by_height(wall_points, num_each_area, span_pixel):
    route = []

    # Sorted the points by height, descending
    points_height = sorted([p[1] for p in wall_points], reverse=True)
    # Sorted the points by width
    # points_width = sorted([p[0] for p in wall_points])
    # Calculate how height is the wall
    height = points_height[0] - points_height[-1]
    # Calculate how wide is the wall
    # width = points_width[-1] - points_width[0]
    # split them into different areas by height and width
    height_layers = math.ceil(height/span_pixel)
    # width_layers = math.ceil(width/span_pixel)

    height_areas = [i*span_pixel for i in range(1, height_layers+1)]
    # width_areas = [i*span_pixel for i in range(1, width_layers+1)]

    # areas = [(x,y) for x in width_areas for y in height_areas]
    areas = sorted([y for y in height_areas],reverse=True)

    #TODO: Pick random points in each area
    for value in areas:
        area_points = [p for p in wall_points if p[1] <= value and p[1] > value-500]
        if len(area_points) >= num_each_area:
            route.append(random.sample(area_points,num_each_area))
        else:
            route.append(random.sample(area_points,len(area_points)))

    route = [item for l in route for item in l]

    return route

def plot_simulated_routes(sim_routes_df, routes_per_row=3):
    num_routes = len(sim_routes_df)
    rows = (num_routes + routes_per_row - 1) // routes_per_row
    fig, axs = plt.subplots(rows, routes_per_row, figsize=(5 * routes_per_row, 5 * rows))
    axs = axs.flatten()

    for i, (route_id, points) in enumerate(zip(sim_routes_df['route_id'], sim_routes_df['points'])):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        axs[i].plot(xs, ys, marker='o', linestyle='-', color='royalblue')

        for idx, (x, y) in enumerate(zip(xs, ys)):
            axs[i].text(x, y, str(idx), fontsize=10, color='black', ha='center', va='center', zorder=6)

        axs[i].scatter(xs[0],ys[0],color='green',s=100,zorder=5,label="Start")
        axs[i].scatter(xs[-1],ys[-1],color='red',s=100,zorder=5,label="End")

        axs[i].invert_yaxis()
        axs[i].set_title(f"Route: {route_id}")
        axs[i].set_xlabel("x (px)")
        axs[i].set_ylabel("y (px)")
        axs[i].grid(True)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_reachability(route, hand_pos, foot_pos, reachable_hands, reachable_feet):
    # h: hand, f: foot, rh: reachable hands, fh: reachable feet
    set_rh = set(reachable_hands)
    set_rf = set(reachable_feet)
    set_both = set_rh & set_rf

    # remove the overlapping points
    set_rh = set_rh - set_both
    set_rf = set_rf - set_both

    x_all, y_all = zip(*route)
    x_h, y_h = zip(*hand_pos)
    x_f, y_f = zip(*foot_pos)
    
    x_rh, y_rh = zip(*reachable_hands) if reachable_hands else ([],[])
    x_rf, y_rf = zip(*reachable_feet) if reachable_feet else ([],[])
    x_bh, y_bh = zip(*set_both) if set_both else ([],[])

    fig, axs = plt.subplots(1, 2, figsize=(16,10))

    # Current Pos
    axs[0].scatter(x_all, y_all, c='gray', label='All Holds')
    axs[0].scatter(x_h, y_h, c='blue', label='Current Hands')
    axs[0].scatter(x_f, y_f, c='black', label='Current Feet')
    axs[0].invert_yaxis()
    axs[0].set_title("Current State")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].legend()

    # Reachable Points
    axs[1].scatter(x_all, y_all, c='gray', label='All Holds')
    axs[1].scatter(x_rh, y_rh, c='green', label='Reachable Hands')
    axs[1].scatter(x_rf, y_rf, c='orange', label='Reachable Feet')
    axs[1].scatter(x_bh, y_bh, c='red', label='Both Reachable')
    axs[1].invert_yaxis()
    axs[1].set_title("Reachable Points")
    axs[1].set_xlabel("X")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
