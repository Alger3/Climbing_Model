import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class ReachabilityFeaturesGNN(nn.Module):
    def __init__(self, node_in=6, climber_in=6, hidden=64, out=4):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.norm1 = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
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