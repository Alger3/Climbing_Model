import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ReachabilityFeaturesGNN(nn.Module):
    def __init__(self, node_in=6, climber_in=5, hidden=64, out=4):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.climber_embed = nn.Linear(climber_in, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        climber_vec = data.climber  # [batch_size, climber_in]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        climber_embed = self.climber_embed(climber_vec)  # [B, H]
        climber_per_node = climber_embed[batch]          # expand to [N, H]
        
        x = torch.cat([x, climber_per_node], dim=1)      # concat node & climber
        return self.classifier(x)