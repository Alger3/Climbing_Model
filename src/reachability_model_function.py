from torch_geometric.data import Data
from route_parser import pixel_dist_to_cm
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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

    # 全连接图
    # num_nodes = len(route)
    # edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
    # # Let the edge become undirected edge
    # edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    # KNN
    num_nodes = len(route)
    route_array = np.array(route)

    nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(route_array)
    _, indices = nbrs.kneighbors(route_array)

    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edges.append([i,j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)

    return Data(x=x, edge_index=edge_index, y=y)

class ReachabilityGNN(nn.Module):
    def __init__(self, node_in=10, hidden=32, out=4):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, out)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.norm(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.norm(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.norm(x)
        return self.classifier(x)