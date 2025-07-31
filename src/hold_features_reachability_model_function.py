import torch
from torch_geometric.data import Data

def build_graph_hold_features(df_group):
    x = torch.tensor(df_group[[
        'center_x',
        'center_y',
        'shape_area',
        'shape_perimeter',
        'shape_aspect_ratio',
        'shape_circularity'
    ]].values, dtype=torch.float64)