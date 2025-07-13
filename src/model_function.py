from torch_geometric.data import Data
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

def build_graph_from_sample(entry):
    features = entry["features"]
    points = features["points"] if "points" in features else []

    # catch points as node (x,y)
    x = torch.tensor(points,dtype=torch.float)

    # edge: (i -> i+1)
    edge_index = torch.tensor([[i,i+1] for i in range(len(points)-1)],dtype=torch.long).T

    # Climber features
    climber_vec = torch.tensor([
        features["height"],
        features["ape_index"],
        features["strength"],
        features["weight"],
        features["flexibility"]
    ],dtype=torch.float)

    label = torch.tensor([entry["feasible"]],dtype=torch.long)

    data = Data(x=x,edge_index=edge_index,y=label)
    data.climber_features = climber_vec

    return data

class ClimbGNN(nn.Module):
    def __init__(self, node_in=2,hidden=32,global_feat_dim=5):
        super().__init__()
        self.conv1 = GCNConv(node_in,hidden)
        self.conv2 = GCNConv(hidden,hidden)