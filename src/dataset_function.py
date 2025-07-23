from route_parser import pixel_dist
import torch
import numpy as np
from torch_geometric.data import Data

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
        if any(pixel_dist(h,p) <= hand_reach for h in current_hand):
            reachable_hand.append(p)
        if any(pixel_dist(f,p) <= foot_reach for f in current_foot):
            reachable_foot.append(p)
    
    return reachable_hand, reachable_foot

def generate_labeled_route_no_sides(route, hand_points, foot_points, climber):
    hand_reach, foot_reach = get_reach_ranges(climber)

    labels = []
    for p in route:
        hand = any(pixel_dist(p, h) <= hand_reach for h in hand_points)
        foot = any(pixel_dist(p, f) <= foot_reach for f in foot_points)

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
        hand_dists = [pixel_dist(p, h) for h in hand_points]
        foot_dists = [pixel_dist(p, f) for f in foot_points]

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