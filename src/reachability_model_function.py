from torch_geometric.data import Data
from route_parser import pixel_dist_to_cm
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
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
    route_array = np.array(route)

    nbrs = NearestNeighbors(n_neighbors=min(10,len(route)), algorithm='auto').fit(route_array)
    _, indices = nbrs.kneighbors(route_array)

    edges = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edges.add(tuple(sorted((i, j))))
    
    edge_index = torch.tensor(list(edges), dtype=torch.long).T

    climber_feat = torch.tensor([
        climber['height'],
        climber['ape_index'],
        climber['flexibility'],
        climber['leg_len_factor']
    ], dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, y=y, climber=climber_feat)

class ReachabilityGNN(nn.Module):
    def __init__(self, node_in=6, climber_in=4, hidden=64, out=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(node_in, hidden, heads=2, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=2, concat=False)
        self.climber_embed = nn.Linear(climber_in, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden, out)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        climber_vec = data.climber  # shape: [batch_size, 4]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        climber_embed = self.climber_embed(climber_vec)         # [B, 64]
        climber_embed = F.relu(climber_embed)
        climber_embed = F.dropout(climber_embed, p=self.dropout, training=self.training)

        climber_per_node = climber_embed[batch]                 # [N, 64]

        x = torch.cat([x, climber_per_node], dim=1)             # [N, 128]
        return self.classifier(x)                               # [N, 4]
    
# Used in imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()