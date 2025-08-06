import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from route_parser import pixel_dist_to_cm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

class ReachabilityFeaturesGNN(nn.Module):
    def __init__(self, node_in=10, climber_in=6, hidden=64, out=4, heads=4):
        super().__init__()
        self.conv1 = GATConv(node_in, hidden, heads=heads, concat=True)
        self.norm1 = BatchNorm(hidden*heads)

        self.conv2 = GATConv(hidden*heads, hidden, heads=1, concat=True)
        self.norm2 = BatchNorm(hidden)

        self.climber_embed = nn.Sequential(
            nn.Linear(climber_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        climber_vec = data.climber  # [B, climber_in]

        # Node feature GCN pipeline
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = F.relu(self.norm2(self.conv2(x, edge_index)))

        # Climber vector expansion
        climber_embed = self.climber_embed(climber_vec)  # [B, H]
        climber_per_node = climber_embed[batch]          # [N, H]

        # Combine and classify
        x = torch.cat([x, climber_per_node], dim=1)
        return self.classifier(x)
    
def plot_graph_prediction(graph, model, title):
    label_colors = {
        0: 'gray',  # unreachable
        1: 'blue',  # hand reachable
        2: 'orange',# foot reachable
        3: 'green'  # both reachable
    }

    label_names = {
        0: 'unreachable',
        1: 'hand',
        2: 'foot',
        3: 'both'
    }

    model.eval()
    with torch.no_grad():
        graph = graph.to(next(model.parameters()).device)
        logits = model(graph)
        preds = logits.argmax(dim=1).cpu().numpy()
        coords = graph.x[:, :2].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8,6))

        for i, (x,y) in enumerate(coords):
            label = preds[i]
            ax.scatter(x, y, color=label_colors[label], s=100, edgecolors='black')

        if hasattr(graph, 'hands'):
            hands = graph.hands.cpu().numpy()
            for hx, hy in hands:
                ax.scatter(hx, hy, s=150, facecolors='none', edgecolors='red', linewidths=2, marker='s', label='hand start')
        if hasattr(graph, 'feet'):
            feet = graph.feet.cpu().numpy()
            for fx, fy in feet:
                ax.scatter(fx, fy, s=150, facecolors='none', edgecolors='purple', linewidths=2, marker='s', label='foot start')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label_names[i],
                   markerfacecolor=label_colors[i], markersize=10, markeredgecolor='black')
            for i in label_colors
        ] + [
            Line2D([0], [0], marker='s', color='r', label='Hand Start',
                   markerfacecolor='none', markeredgewidth=2, markersize=10),
            Line2D([0], [0], marker='s', color='purple', label='Foot Start',
                   markerfacecolor='none', markeredgewidth=2, markersize=10)
        ]
        ax.legend(handles=legend_elements, title="Predicted Labels")

        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

def build_graph_reachability_features(route, hand_points, foot_points, climber, holds_features, labels):
    node_features = []

    for i, p in enumerate(route):
        hold_feat = holds_features[i]

        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points]

        feature = list(p) + [  # x, y
            np.mean(hand_dists),
            np.min(hand_dists),
            np.mean(foot_dists),
            np.min(foot_dists),
            hold_feat.get("shape_area", 0),
            hold_feat.get("shape_perimeter", 0),
            hold_feat.get("shape_aspect_ratio", 0),
            hold_feat.get("shape_circularity", 0),
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
        climber['leg_len_factor'],
        climber["weight"],
        climber["strength"]
    ], dtype=torch.float).unsqueeze(0)

    graph = Data(x=x, edge_index=edge_index, y=y, climber=climber_feat)

    graph.hands = torch.tensor(hand_points, dtype=torch.float)
    graph.feet = torch.tensor(foot_points, dtype=torch.float)

    return graph