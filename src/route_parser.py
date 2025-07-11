import json
import random
import networkx as nx
from math import dist
import matplotlib.pyplot as plt

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

# Simulate a route from wall
def simulate_route_from_wall(points,max_reach,num_holds):
    route = []
    used = set()

    # 只从底部 20% 高度的点中随机选起点
    y_threshold = sorted([p[1] for p in points], reverse=True)[int(len(points) * 0.2)]
    bottom_candidates = [p for p in points if p[1] >= y_threshold]
    if not bottom_candidates:
        return []

    current = random.choice(bottom_candidates)
    route.append(current)
    used.add(current)

    while len(route) < num_holds:
        candidates = [
            p for p in points
            if p not in used and dist(current, p) <= max_reach and p[1] < current[1]
        ]
        if not candidates:
            break
        next_point = random.choice(candidates)
        route.append(next_point)
        used.add(next_point)
        current = next_point

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
