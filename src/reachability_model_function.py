from torch_geometric.data import Data, Batch
from route_parser import pixel_dist_to_cm
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import copy
from itertools import combinations
from dataset_function import get_reach_ranges

def build_graph_reachability(route, hand_points, foot_points, climber, labels, epsilon_cm=1.0):
    node_features = []

    for p in route:
        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points]

        is_hand_point = int(min(hand_dists) <= epsilon_cm)
        is_foot_point = int(min(foot_dists) <= epsilon_cm)

        feature = list(p) + [  # x, y
            np.mean(hand_dists),
            np.min(hand_dists),
            np.mean(foot_dists),
            np.min(foot_dists),
            float(is_hand_point),
            float(is_foot_point)
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

    graph = Data(x=x, edge_index=edge_index, y=y, climber=climber_feat)

    graph.hands = torch.tensor(hand_points, dtype=torch.float)
    graph.feet = torch.tensor(foot_points, dtype=torch.float)

    return graph


# ------------------------------------------------------------
# Model: GAT + FiLM (主干不吃 flags；flag 旁路加权)
# ------------------------------------------------------------
class ReachabilityGNN(nn.Module):
    def __init__(self, node_in_main=6, climber_in=4, hidden=64, out=4, dropout=0.2, alpha_flag=0.05, heads=2):
        super().__init__()
        self.alpha_flag = alpha_flag
        self.dropout = dropout

        self.lin_in = nn.Linear(node_in_main, hidden)
        self.conv1  = GATConv(hidden, hidden, heads=heads, concat=False)
        self.conv2  = GATConv(hidden, hidden, heads=heads, concat=False)

        # FiLM: 从攀爬者特征生成每层 γ/β
        self.climber_embed = nn.Linear(climber_in, hidden)
        self.film1 = nn.Linear(hidden, 2*hidden)
        self.film2 = nn.Linear(hidden, 2*hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out)
        )

        # 仅用 flags 的旁路
        self.flag_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, out)
        )
        with torch.no_grad():
            for m in self.flag_head:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=1e-3)

    def _apply_film(self, x, climber_h, film_layer, batch):
        gb = film_layer(climber_h)                          # [B, 2H]
        gamma, beta = torch.chunk(gb, 2, dim=-1)            # [B,H], [B,H]
        return x * (1 + gamma[batch]) + beta[batch]         # 残差式调制

    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x_all.size(0), dtype=torch.long, device=x_all.device)

        # 主干只用前 6 维连续特征；flag 只进旁路
        flags = x_all[:, -2:]                # [N,2]
        x     = x_all[:, :-2]                # [N,6]

        c = F.relu(self.climber_embed(data.climber))        # [B,H]
        c = F.dropout(c, p=self.dropout, training=self.training)

        x = self.lin_in(x)
        x = self._apply_film(x, c, self.film1, batch)
        x = F.relu(self.conv1(x, edge_index))
        x = self._apply_film(x, c, self.film2, batch)
        x = F.relu(self.conv2(x, edge_index))

        logits_main = self.classifier(x)
        logits_flag = self.flag_head(flags)
        return logits_main + self.alpha_flag * logits_flag

class ReachabilityGNNv11(nn.Module):
    def __init__(self, node_in=8, climber_in=4, hidden=64, out=4, dropout=0.2):
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

        self.flag_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, out)
        )

        with torch.no_grad():
            for m in self.flag_head:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=1e-3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        climber_vec = data.climber  # shape: [batch_size, 4]

        flags = x[:, -2:]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        climber_embed = self.climber_embed(climber_vec)         # [B, 64]
        climber_embed = F.relu(climber_embed)
        climber_embed = F.dropout(climber_embed, p=self.dropout, training=self.training)

        climber_per_node = climber_embed[batch]                 # [N, 64]

        # 主分支 logits
        feat = torch.cat([x, climber_per_node], dim=1)            # [N, 2H]
        logits_main = self.classifier(feat)                       # [N, out]

        # flag 分支 logits（只看 is_hand_point/is_foot_point）
        logits_flag = self.flag_head(flags) 

        return logits_main + logits_flag                            # [N, 4]

# ------------------------------------------------------------
# Focal Loss（不平衡）
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def plot_graph_prediction(graph, model, title, goal_xy=None):
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

        if goal_xy is not None:
            gx, gy = goal_xy
            ax.scatter(gx, gy, s=200, facecolors='none', edgecolors='red', linewidths=3, marker='o', label='Goal')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label_names[i],
                   markerfacecolor=label_colors[i], markersize=10, markeredgecolor='black')
            for i in label_colors
        ] + [
            Line2D([0], [0], marker='s', color='r', label='Hand Start',
                   markerfacecolor='none', markeredgewidth=2, markersize=10),
            Line2D([0], [0], marker='s', color='purple', label='Foot Start',
                   markerfacecolor='none', markeredgewidth=2, markersize=10),
            Line2D([0], [0], marker='o', color='red', label='Goal',
                   markerfacecolor='none', markeredgewidth=3, markersize=10)
        ]
        ax.legend(handles=legend_elements, title="Predicted Labels")

        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def simulate_climb_to_goal(graph, model, goal_xy, max_steps=30, 
                           foot_below_hands=True, y_margin=0):
    epsilon_cm = 1.0
    device = next(model.parameters()).device
    g = copy.deepcopy(graph).to(device)

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords])
    step = 0

    while step <= max_steps:
        step += 1
        model.eval()
        with torch.no_grad():
            logits = model(g)
            coords = g.x[:, :2].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        plot_graph_prediction(g.cpu(), model, f"Step {step}", goal_xy)

        goal_label = preds[goal_idx]
        if goal_label != 0:
            print(f"Success: Model predict climber can reach the goal (label={goal_label}) in Step {step}")
            break

        # 可及索引
        hand_idx = np.where((preds == 1) | (preds == 3))[0]
        foot_idx = np.where((preds == 2) | (preds == 3))[0]

        if len(hand_idx) == 0 and len(foot_idx) == 0:
            print("No Reachable Holds.")
            break

        # ---------- 选手 ----------
        selected_hand_indices = []
        if len(hand_idx) > 0:
            # 手1：离目标最近
            hand_dists_goal = sorted(
                [(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in hand_idx],
                key=lambda x: x[1]
            )
            first_hand = hand_dists_goal[0][0]
            selected_hand_indices.append(first_hand)

            # 手2：离手1最近
            if len(hand_idx) > 1:
                first_xy = coords[first_hand]
                candidates = [i for i in hand_idx if i != first_hand]
                second_hand = min(candidates, key=lambda j: pixel_dist_to_cm(coords[j], first_xy))
                selected_hand_indices.append(second_hand)

        selected_hands = [coords[i] for i in selected_hand_indices]

        # ---------- 选脚：尽量在手的“下面” ----------
        used_idxs = set(selected_hand_indices)
        # 原始候选（不含已选手）
        base_foot_candidates = [i for i in foot_idx if i not in used_idxs]

        # 若需要脚在手下方，则先筛一遍
        foot_candidates = base_foot_candidates
        if foot_below_hands and len(selected_hands) > 0:
            hand_y_max = max(h[1] for h in selected_hands)  # 像素坐标，y越大越“下”
            filtered = [i for i in base_foot_candidates if coords[i][1] > hand_y_max + y_margin]
            # 如果筛完一个都没有，就回退用原始候选
            if len(filtered) > 0:
                foot_candidates = filtered

        # 在候选里按目标距离排序，取前2
        foot_dists = sorted([(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in foot_candidates],
                            key=lambda x: x[1])
        selected_feet = [coords[i] for i, _ in foot_dists[:2]]

        # 写回选择结果
        if len(selected_hands) > 0:
            g.hands = torch.tensor(selected_hands, dtype=torch.float, device=device)
        if len(selected_feet) > 0:
            g.feet = torch.tensor(selected_feet, dtype=torch.float, device=device)

        # 重算节点特征（保持你原逻辑）
        new_x = g.x.clone().cpu().numpy()
        for i, p in enumerate(coords):
            hand_dists_all = [pixel_dist_to_cm(p, h) for h in g.hands.cpu().numpy()]
            foot_dists_all = [pixel_dist_to_cm(p, f) for f in g.feet.cpu().numpy()]
            is_hand_point = int(min(hand_dists_all) <= epsilon_cm)
            is_foot_point = int(min(foot_dists_all) <= epsilon_cm)

            new_x[i, 2] = np.mean(hand_dists_all)  # mean hand dist
            new_x[i, 3] = np.min(hand_dists_all)   # min hand dist
            new_x[i, 4] = np.mean(foot_dists_all)  # mean foot dist
            new_x[i, 5] = np.min(foot_dists_all)   # min foot dist
            new_x[i, 6] = is_hand_point
            new_x[i, 7] = is_foot_point
        g.x = torch.tensor(new_x, dtype=torch.float, device=device)

    else:
        print(f"Climber cannot reach the goal in {max_steps} steps.")
