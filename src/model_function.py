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

# This function is for inference (no feasibility), different structure
def build_graph_for_inference(route,climber_info):
    x = torch.tensor(route,dtype=torch.float)
    edge_index = torch.tensor([[i,i+1] for i in range(len(route)-1)],dtype=torch.long).T

    climber_vec = torch.tensor([
        climber_info["height"],
        climber_info["ape_index"],
        climber_info["strength"],
        climber_info["weight"],
        climber_info["flexibility"]
    ],dtype=torch.float)

    data = Data(x=x,edge_index=edge_index)
    data.climber_features = climber_vec

    return data

# This class is inherited from PyTorch-nn.Module
class ClimbGNN(nn.Module):
    # node_in: the node is 2-dimensionality (x,y)
    # hidden: hidden layer
    # global_feat_dim: climber characteristic dimension
    def __init__(self, node_in=2,hidden=32,global_feat_dim=5):
        super().__init__()
        # Two graph convolutional layer
        self.conv1 = GCNConv(node_in,hidden)
        self.conv2 = GCNConv(hidden,hidden)

        # Small MLP which can map the climber characteristic to 32 dimension
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_dim,32),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden+32,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,data):
        # x: node, edge_index: the relation of edge (i,j)
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # pooling over nodes
        x = torch.mean(x, dim=0)

        # Nonlinear mapping of the player's body feature vectors
        g_feat = self.global_mlp(data.climber_features)

        # combine the route vec with climber characteristic vec
        out = torch.cat([x, g_feat], dim=0)
        return self.classifier(out.unsqueeze(0))  # add batch dimension

def predict_feasibility(model, route, climber_info):
    # route: [(x1,y1),(x2,y2),...]
    # climber_info: (dict), height, weight, ape_index, strength, flexibility
    # output: 0 or 1

    data = build_graph_for_inference(route,climber_info)
    model.eval()
    out = model(data)

    return torch.argmax(out).item()

# 使用不同的algorithm
# 一只脚在某点，根据这个去计算手和脚的可达处
# model：比如我输入route的全部points和climber当前的points（手脚）以及characteristic，然后model可以返回climber的可及点。（不断重复）