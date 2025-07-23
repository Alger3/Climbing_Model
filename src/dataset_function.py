from route_parser import pixel_dist_to_cm
import torch
import numpy as np
from torch_geometric.data import Data
from route_parser import PIXEL_TO_CM
import matplotlib.pyplot as plt

# Calculate the reach range of hand and leg
def get_reach_ranges(climber):
    height = climber["height"]
    ape_index = climber["ape_index"]
    flexibility = climber["flexibility"]
    leg_len_factor = climber["leg_len_factor"]

    arm_reach = height * ape_index + flexibility * 3
    leg_reach = height * leg_len_factor + flexibility * 2
    
    return arm_reach, leg_reach

# Calculate which holds are reachable
def get_reachable_points(current_hand, current_foot, climber, route):
    hand_reach, foot_reach = get_reach_ranges(climber)
    reachable_hand = []
    reachable_foot = []

    for p in route:
        if any(pixel_dist_to_cm(h,p) <= hand_reach for h in current_hand):
            reachable_hand.append(p)
        if any(pixel_dist_to_cm(f,p) <= foot_reach for f in current_foot):
            reachable_foot.append(p)
    
    return reachable_hand, reachable_foot

def generate_labeled_route_no_sides(route, hand_points, foot_points, climber):
    hand_reach, foot_reach = get_reach_ranges(climber)

    labels = []
    for p in route:
        hand = any(pixel_dist_to_cm(p, h) <= hand_reach for h in hand_points)
        foot = any(pixel_dist_to_cm(p, f) <= foot_reach for f in foot_points)

        if hand and foot:
            labels.append(3)  # both hand and foot can reach
        elif hand:
            labels.append(1)  # only hand can reach
        elif foot:
            labels.append(2)  # only foot can reach
        else:
            labels.append(0)  # cant reach
    return labels

# Build the Graph
def build_graph_reachability(route, hand_points, foot_points, climber, labels):
    node_features = []

    for p in route:
        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points]

        feature = list(p) + [  # x, y
            np.mean(hand_dists),
            np.min(hand_dists),
            np.mean(foot_dists),
            np.min(foot_dists),
            climber['height'],
            climber['ape_index'],
            climber['flexibility'],
            climber['leg_len_factor']
        ]
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    num_nodes = len(route)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    return Data(x=x, edge_index=edge_index, y=y)

def sample_grap_points_by_radius(route, climber, max_trials=50):
    arm_reach, leg_reach = get_reach_ranges(climber)
    max_reach = max(arm_reach, leg_reach)
    radius = max_reach / 2

    for _ in range(max_trials):
        center = route[np.random.randint(len(route))]

        dists_cm = np.array([pixel_dist_to_cm(p, center) for p in route])
        within_idxs = np.where(dists_cm <= radius)[0]

        if len(within_idxs) >= 4:
            selected = np.random.choice(within_idxs, 4, replace=False)
            points = [route[i] for i in selected]

            sorted_points = sorted(points, key=lambda p: p[1])
            hand_points = sorted_points[:2]  
            foot_points = sorted_points[2:]  

            if max(p[1] for p in hand_points) < min(p[1] for p in foot_points):
                return hand_points, foot_points, center
    return None, None, None

def plot_route_with_grab(route, hand_points, foot_points, center=None):
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys, c='gray', label='Holds')

    if center is not None:
        plt.scatter(center[0], center[1], color='orange', marker='x', s=150, label='Center')

    for p in hand_points:
        plt.scatter(p[0], p[1], color='green', s=120, label='Hand')

    for p in foot_points:
        plt.scatter(p[0], p[1], color='blue', s=120, label='Foot')

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Grab Position (Pixel)")
    plt.show()


