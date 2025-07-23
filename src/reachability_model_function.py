from torch_geometric.data import Data
from route_parser import pixel_dist_to_cm
import numpy as np
import torch

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