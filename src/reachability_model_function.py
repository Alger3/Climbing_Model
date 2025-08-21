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
from route_parser import PIXEL_TO_CM

def build_single_graph_reachability(route, hand_points, foot_points, climber, epsilon_cm=1.0):
    node_features = []

    arm_reach, leg_reach = get_reach_ranges(climber)

    for p in route:
        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points]

        is_hand_point = int(min(hand_dists) <= epsilon_cm)
        is_foot_point = int(min(foot_dists) <= epsilon_cm)

        feature = list(p) + [  # (x,y)
            np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0),
            np.clip(np.min(hand_dists) / arm_reach, 0.0, 3.0),
            np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0),
            np.clip(np.min(foot_dists) / leg_reach, 0.0, 3.0),
            float(is_hand_point),
            float(is_foot_point)
        ]
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)

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
        climber["arm_span"],
        climber["leg_span"]
    ], dtype=torch.float).unsqueeze(0)

    graph = Data(x=x, edge_index=edge_index, climber=climber_feat)

    graph.hands = torch.tensor(hand_points, dtype=torch.float)
    graph.feet = torch.tensor(foot_points, dtype=torch.float)

    return graph

def build_graph_reachability(route, hand_points, foot_points, climber, labels, epsilon_cm=1.0):
    node_features = []

    arm_reach, leg_reach = get_reach_ranges(climber)

    for p in route:
        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points]

        is_hand_point = int(min(hand_dists) <= epsilon_cm)
        is_foot_point = int(min(foot_dists) <= epsilon_cm)

        feature = list(p) + [  # (x,y)
            np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0),
            np.clip(np.min(hand_dists) / arm_reach, 0.0, 3.0),
            np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0),
            np.clip(np.min(foot_dists) / leg_reach, 0.0, 3.0),
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
        climber['leg_len_factor'],
        climber["arm_span"],
        climber["leg_span"]
    ], dtype=torch.float).unsqueeze(0)

    graph = Data(x=x, edge_index=edge_index, y=y, climber=climber_feat)

    graph.hands = torch.tensor(hand_points, dtype=torch.float)
    graph.feet = torch.tensor(foot_points, dtype=torch.float)

    return graph


# Model: GAT + FiLM (主干不吃 flags；flag 旁路加权)
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
    def __init__(self, node_in=8, climber_in=6, hidden=64, out=4, dropout=0.2):
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

class ReachabilityGNNv13(nn.Module):
    def __init__(self, node_in_cont=6, climber_in=6, hidden=64, out=4, dropout=0.2, alpha_flag=0.03):
        super().__init__()
        self.dropout = dropout
        self.alpha_flag = alpha_flag

        self.conv1 = GATConv(hidden, hidden, heads=1, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False)

        self.lin_in = nn.Linear(node_in_cont, hidden)
        self.climber_embed = nn.Sequential(nn.LayerNorm(climber_in), nn.Linear(climber_in, hidden), nn.ReLU())
        self.film1 = nn.Linear(hidden, 2*hidden)
        self.film2 = nn.Linear(hidden, 2*hidden)

        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out))
        self.flag_head = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, out))

    def apply_film(self, h, c, film, batch):
        gamma, beta = torch.chunk(film(c), 2, dim=-1)
        gamma, beta = gamma[batch], beta[batch]
        return h * (1 + gamma) + beta

    def forward(self, data):
        x_all, edge_index, batch = data.x, data.edge_index, data.batch
        x_cont, flags = x_all[:, :6], x_all[:, 6:8]

        h = self.lin_in(x_cont)
        c = self.climber_embed(data.climber)
        h = self.apply_film(h, c, self.film1, batch); h = F.relu(self.conv1(h, edge_index))
        h = self.apply_film(h, c, self.film2, batch); h = F.relu(self.conv2(h, edge_index))

        logits = self.cls(h) + self.alpha_flag * self.flag_head(flags)
        return logits


# Focal Loss（不平衡）
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

def recompute_node_features_with_climber(g, epsilon_cm=1.0, clip_max=3.0):
    import numpy as np, torch

    # 取坐标与起始手脚
    coords = g.x[:, :2].cpu().numpy()
    hands  = g.hands.cpu().numpy()
    feet   = g.feet.cpu().numpy()

    climber = {
        "height": g.climber[0][0].item(),
        "ape_index": g.climber[0][1].item(),
        "flexibility": g.climber[0][2].item(),
        "leg_len_factor": g.climber[0][3].item(),
        "arm_span": g.climber[0][4].item(),
        "leg_span": g.climber[0][5].item()
    }

    arm_reach, leg_reach = get_reach_ranges(climber)



    new = g.x.clone().cpu().numpy()

    for i, p in enumerate(coords):
        hand_dists = [pixel_dist_to_cm(p, h) for h in hands]
        foot_dists = [pixel_dist_to_cm(p, f) for f in feet]

        is_hand_point = int(min(hand_dists) <= epsilon_cm)
        is_foot_point = int(min(foot_dists) <= epsilon_cm)

        new[i, 2] = np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0)
        new[i, 3] = np.clip(np.min(hand_dists) / arm_reach, 0.0, 3.0)
        new[i, 4] = np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0)
        new[i, 5] = np.clip(np.min(foot_dists) / leg_reach, 0.0, 3.0)

        new[i, 6] = float(is_hand_point)
        new[i, 7] = float(is_foot_point)

    g.x = torch.tensor(new, dtype=g.x.dtype, device=g.x.device)
    return g

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

# 一次（一个step）直接更新手和脚
def simulate_climb_to_goal(graph, model, goal_xy, max_steps=40, 
                           foot_below_hands=True, y_margin=0):
    epsilon_cm = 1.0
    device = next(model.parameters()).device
    g = copy.deepcopy(graph).to(device)

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords])
    step = 0

    climber = {
        "height": g.climber[0][0].item(),
        "ape_index": g.climber[0][1].item(),
        "flexibility": g.climber[0][2].item(),
        "leg_len_factor": g.climber[0][3].item(),
        "arm_span": g.climber[0][4].item(),
        "leg_span": g.climber[0][5].item()
    }

    arm_reach, leg_reach = get_reach_ranges(climber)

    while step < max_steps:
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
        hands  = g.hands.cpu().numpy()
        feet   = g.feet.cpu().numpy()
        new_x = g.x.clone().cpu().numpy()
        for i, p in enumerate(coords):
            hand_dists = [pixel_dist_to_cm(p, h) for h in hands]
            foot_dists = [pixel_dist_to_cm(p, f) for f in feet]

            is_hand_point = int(min(hand_dists) <= epsilon_cm)
            is_foot_point = int(min(foot_dists) <= epsilon_cm)

            new_x[i, 2] = np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 3] = np.clip(np.min(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 4] = np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0)
            new_x[i, 5] = np.clip(np.min(foot_dists) / leg_reach, 0.0, 3.0)

            new_x[i, 6] = float(is_hand_point)
            new_x[i, 7] = float(is_foot_point)
        g.x = torch.tensor(new_x, dtype=torch.float, device=device)

    else:
        print(f"Climber cannot reach the goal in {max_steps} steps.")

# 一次（一个step）只更新手或者脚
def simulate_climb_to_goal_v2(
    graph, model, goal_xy, max_steps=40,
    foot_below_hands=True, y_margin=0,
    angle_threshold_deg=30.0,      # 超过该角度先动脚，否则动手
    foot_align_w=1.0,              # 动脚时：脚中心向手中心对齐的惩罚权重（越大越“居中”）
    foot_goal_w=0.1,               # 动脚时：仍给一点“靠近目标”的偏好（0~0.3）
    prefer_hands_above_feet=True,  
    y_above_margin_hand=0         
):
    import copy, numpy as np, torch, math

    epsilon_cm = 1.0
    device = next(model.parameters()).device
    g = copy.deepcopy(graph).to(device)

    def get_np(arr):
        if arr is None: return np.zeros((0,2), dtype=float)
        x = arr.detach().cpu().numpy()
        return x if x.size else np.zeros((0,2), dtype=float)

    def centers_from_state(g):
        hands_np = get_np(getattr(g, "hands", None))
        feet_np  = get_np(getattr(g, "feet",  None))
        hand_c = hands_np.mean(axis=0) if hands_np.shape[0] > 0 else None
        foot_c = feet_np.mean(axis=0)  if feet_np.shape[0]  > 0 else None
        return hand_c, foot_c, hands_np, feet_np

    def angle_to_vertical_deg(hand_c, foot_c):
        # 以脚中心为角点，计算连线与“竖直线”的夹角（0°=完全竖直，90°=完全水平）
        if hand_c is None or foot_c is None:
            return None
        dx = hand_c[0] - foot_c[0]
        dy = hand_c[1] - foot_c[1]   # 像素坐标 y 向下增大
        ang = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
        return ang

    def map_points_to_indices(points, coords):
        # 将任意坐标映射回最近节点索引（用于“排除已占用手点”）
        idxs = []
        for p in points:
            j = int(np.argmin(np.linalg.norm(coords - p, axis=1)))
            idxs.append(j)
        return idxs

    # ---------- [SR-1] 像素→厘米的比例，用于把像素间距换成 cm ----------
    def pixel_to_cm_scale():
        return float(pixel_dist_to_cm(np.array([0., 0.]), np.array([1., 0.])))
    PIXEL_TO_CM = pixel_to_cm_scale()

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords])
    step = 0

    climber = {
        "height": g.climber[0][0].item(),
        "ape_index": g.climber[0][1].item(),
        "flexibility": g.climber[0][2].item(),
        "leg_len_factor": g.climber[0][3].item(),
        "arm_span": g.climber[0][4].item(),
        "leg_span": g.climber[0][5].item()
    }
    arm_reach, leg_reach = get_reach_ranges(climber)

    # ---------- [SR-2] Standing Reach（换算到厘米） ----------
    H = float(climber["height"])
    W = float(climber["arm_span"])
    standing_reach_cm = 55.581 + (0.121 * H) + (0.814 * W)
    reach_guard_cm = 0.75 * standing_reach_cm

    while step < max_steps:
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

        # ---- 可及集合 ----
        hand_idx = np.where((preds == 1) | (preds == 3))[0]
        foot_idx = np.where((preds == 2) | (preds == 3))[0]
        if len(hand_idx) == 0 and len(foot_idx) == 0:
            print("No Reachable Holds.")
            break

        # ---- 根据当前状态计算角度，决定动手还是动脚 ----
        hand_c, foot_c, hands_np, feet_np = centers_from_state(g)
        angle_deg = angle_to_vertical_deg(hand_c, foot_c)

        # 若还没有手或脚，先动手初始化（下一步才有角度可用）
        if angle_deg is None:
            move_what = "hands" if len(hand_idx) > 0 else ("feet" if len(foot_idx) > 0 else None)
        else:
            move_what = "feet" if angle_deg > angle_threshold_deg else "hands"

        # ---------- [SR-3] 姿态保护：若“手脚中心的竖直间距”超过 StandingReach 的 75% ----------
        # 超限时，覆盖 move_what：优先移动“离目标更远”的那一类（手/脚），把人拉回合理范围
        if (hand_c is not None) and (foot_c is not None):
            sep_pix = abs(hand_c[1] - foot_c[1])          # 竖直像素差
            sep_cm  = sep_pix * PIXEL_TO_CM               # 换算到厘米
            if sep_cm > reach_guard_cm:
                # 两类中心到目标的距离（cm）
                hand_goal_cm = pixel_dist_to_cm(hand_c, goal_xy)
                foot_goal_cm = pixel_dist_to_cm(foot_c, goal_xy)
                # 谁更远，谁移动；若某类没有候选，则移动另一类
                if (hand_goal_cm >= foot_goal_cm and len(hand_idx) > 0) or len(foot_idx) == 0:
                    move_what = "hands"
                elif len(foot_idx) > 0:
                    move_what = "feet"
                # （其余选择逻辑仍沿用下面原来的规则）

        # ---- 选 / 动 手（加入“在脚上方优先”的偏好）----
        if move_what == "hands" and len(hand_idx) > 0:
            # 1) 组建候选集
            hand_candidates = list(hand_idx)
            if (prefer_hands_above_feet and feet_np.size > 0):
                # 脚中更靠上的那只脚（y 更小=更靠上）
                feet_y_min = float(feet_np[:, 1].min())
                thresh = feet_y_min - y_above_margin_hand
                above_feet = [i for i in hand_candidates if coords[i][1] < thresh]
                if len(above_feet) > 0:
                    hand_candidates = above_feet  # 有就强制只在脚上方里选

            # 2) 手1：在 hand_candidates 里离目标最近
            hand_dists_goal = sorted(
                [(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in hand_candidates],
                key=lambda x: x[1]
            )
            first_hand = hand_dists_goal[0][0]
            selected_hand_indices = [first_hand]

            # 3) 手2：优先在 hand_candidates 里离手1最近；若只有 1 个，则回退到全体 hand_idx
            if len(hand_candidates) > 1:
                first_xy = coords[first_hand]
                candidates = [i for i in hand_candidates if i != first_hand]
                second_hand = min(candidates, key=lambda j: pixel_dist_to_cm(coords[j], first_xy))
                selected_hand_indices.append(second_hand)
            elif len(hand_idx) > 1:
                first_xy = coords[first_hand]
                candidates = [i for i in hand_idx if i != first_hand]
                second_hand = min(candidates, key=lambda j: pixel_dist_to_cm(coords[j], first_xy))
                selected_hand_indices.append(second_hand)

            selected_hands = [coords[i] for i in selected_hand_indices]
            if len(selected_hands) > 0:
                g.hands = torch.tensor(selected_hands, dtype=torch.float, device=device)

        # ---- 选 / 动 脚（“朝手方向对齐”）----
        elif move_what == "feet" and len(foot_idx) > 0:
            current_hand_indices = map_points_to_indices(hands_np, coords) if hands_np.size else []
            used_idxs = set(current_hand_indices)

            base_foot_candidates = [i for i in foot_idx if i not in used_idxs]
            if len(base_foot_candidates) == 0:
                selected_feet_indices = []
            else:
                foot_candidates = base_foot_candidates
                if foot_below_hands and hands_np.size > 0:
                    hand_y_max = float(np.max(hands_np[:, 1]))  # y 越大越“下”
                    filtered = [i for i in base_foot_candidates if coords[i][1] > hand_y_max + y_margin]
                    if len(filtered) > 0:
                        foot_candidates = filtered

                if len(foot_candidates) <= 2:
                    foot_dists = sorted([(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in foot_candidates],
                                        key=lambda x: x[1])
                    selected_feet_indices = [i for i, _ in foot_dists[:2]]
                else:
                    cx = float(np.mean(hands_np[:, 0])) if hands_np.size > 0 else \
                         float(np.mean([coords[i][0] for i in foot_candidates]))
                    def d_goal(i): return pixel_dist_to_cm(coords[i], goal_xy)
                    best = None  # (score, (i,j))
                    cand = foot_candidates
                    for a in range(len(cand)):
                        for b in range(a+1, len(cand)):
                            i, j = cand[a], cand[b]
                            feet_center_x = 0.5 * (coords[i][0] + coords[j][0])
                            # （这里原来就乘了 PIXEL_TO_CM；保持一致）
                            align_pen = abs(feet_center_x - cx) * PIXEL_TO_CM
                            score = foot_align_w * align_pen + foot_goal_w * (d_goal(i) + d_goal(j))
                            if (best is None) or (score < best[0]):
                                best = (score, (i, j))
                    selected_feet_indices = list(best[1]) if best is not None else []

            selected_feet = [coords[i] for i in selected_feet_indices]
            if len(selected_feet) > 0:
                g.feet = torch.tensor(selected_feet, dtype=torch.float, device=device)

        # ---- 重算节点特征（保持你的原逻辑）----
        hands = g.hands.cpu().numpy()
        feet  = g.feet.cpu().numpy()
        new_x = g.x.clone().cpu().numpy()
        for i, p in enumerate(coords):
            hand_dists = [pixel_dist_to_cm(p, h) for h in hands]
            foot_dists = [pixel_dist_to_cm(p, f) for f in feet]

            is_hand_point = int(min(hand_dists) <= epsilon_cm)
            is_foot_point = int(min(foot_dists) <= epsilon_cm)

            new_x[i, 2] = np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 3] = np.clip(np.min(hand_dists)  / arm_reach, 0.0, 3.0)
            new_x[i, 4] = np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0)
            new_x[i, 5] = np.clip(np.min(foot_dists)  / leg_reach, 0.0, 3.0)
            new_x[i, 6] = float(is_hand_point)
            new_x[i, 7] = float(is_foot_point)
        g.x = torch.tensor(new_x, dtype=torch.float, device=device)

    else:
        print(f"Climber cannot reach the goal in {max_steps} steps.")

def calculate_the_step_to_goal(graph, model, goal_xy, max_steps=40, 
                               foot_below_hands=True, y_margin=0):
    """
    无可视化：若模型在某一步预测目标点可达(非0标签)，返回该步数；若在 max_steps 内都不可达，返回 -1。
    """
    import copy
    import numpy as np
    import torch

    epsilon_cm = 1.0
    device = next(model.parameters()).device
    g = copy.deepcopy(graph).to(device)

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords])
    step = 0

    climber = {
        "height": g.climber[0][0].item(),
        "ape_index": g.climber[0][1].item(),
        "flexibility": g.climber[0][2].item(),
        "leg_len_factor": g.climber[0][3].item(),
        "arm_span": g.climber[0][4].item(),
        "leg_span": g.climber[0][5].item()
    }

    arm_reach, leg_reach = get_reach_ranges(climber)

    while step < max_steps:
        step += 1
        model.eval()
        with torch.no_grad():
            logits = model(g)
            coords = g.x[:, :2].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        # 成功条件：目标点被预测为可达(非0标签)
        if preds[goal_idx] != 0:
            return step

        # 可及索引
        hand_idx = np.where((preds == 1) | (preds == 3))[0]
        foot_idx = np.where((preds == 2) | (preds == 3))[0]

        if len(hand_idx) == 0 and len(foot_idx) == 0:
            return -1  # 无任何可及点，失败

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
        base_foot_candidates = [i for i in foot_idx if i not in used_idxs]

        foot_candidates = base_foot_candidates
        if foot_below_hands and len(selected_hands) > 0:
            hand_y_max = max(h[1] for h in selected_hands)  # 像素坐标，y越大越“下”
            filtered = [i for i in base_foot_candidates if coords[i][1] > hand_y_max + y_margin]
            if len(filtered) > 0:
                foot_candidates = filtered

        foot_dists = sorted([(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in foot_candidates],
                            key=lambda x: x[1])
        selected_feet = [coords[i] for i, _ in foot_dists[:2]]

        # 写回选择结果
        if len(selected_hands) > 0:
            g.hands = torch.tensor(selected_hands, dtype=torch.float, device=device)
        if len(selected_feet) > 0:
            g.feet = torch.tensor(selected_feet, dtype=torch.float, device=device)

        # 重算节点特征
        hands  = g.hands.cpu().numpy()
        feet   = g.feet.cpu().numpy()
        new_x = g.x.clone().cpu().numpy()
        for i, p in enumerate(coords):
            hand_dists = [pixel_dist_to_cm(p, h) for h in hands]
            foot_dists = [pixel_dist_to_cm(p, f) for f in feet]

            is_hand_point = int(min(hand_dists) <= epsilon_cm)
            is_foot_point = int(min(foot_dists) <= epsilon_cm)

            new_x[i, 2] = np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 3] = np.clip(np.min(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 4] = np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0)
            new_x[i, 5] = np.clip(np.min(foot_dists) / leg_reach, 0.0, 3.0)

            new_x[i, 6] = float(is_hand_point)
            new_x[i, 7] = float(is_foot_point)
        g.x = torch.tensor(new_x, dtype=torch.float, device=device)

    # 循环结束仍未到达
    return -1

def calculate_the_step_to_goal_v2(
    graph, model, goal_xy, max_steps=40,
    foot_below_hands=True, y_margin=0,
    angle_threshold_deg=30.0,      # 超过该角度先动脚，否则动手
    foot_align_w=1.0,              # 动脚：脚中心向手中心对齐的惩罚权重
    foot_goal_w=0.1,               # 动脚：靠近目标的权重
    prefer_hands_above_feet=True,  
    y_above_margin_hand=0
):
    import copy, numpy as np, torch, math

    epsilon_cm = 1.0
    device = next(model.parameters()).device
    g = copy.deepcopy(graph).to(device)

    def get_np(arr):
        if arr is None: return np.zeros((0,2), dtype=float)
        x = arr.detach().cpu().numpy()
        return x if x.size else np.zeros((0,2), dtype=float)

    def centers_from_state(g):
        hands_np = get_np(getattr(g, "hands", None))
        feet_np  = get_np(getattr(g, "feet",  None))
        hand_c = hands_np.mean(axis=0) if hands_np.shape[0] > 0 else None
        foot_c = feet_np.mean(axis=0)  if feet_np.shape[0]  > 0 else None
        return hand_c, foot_c, hands_np, feet_np

    def angle_to_vertical_deg(hand_c, foot_c):
        # 以脚中心为角点，与竖直线的夹角（0°竖直）
        if hand_c is None or foot_c is None:
            return None
        dx = hand_c[0] - foot_c[0]
        dy = hand_c[1] - foot_c[1]   # 像素坐标 y 向下增大
        return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))

    def map_points_to_indices(points, coords):
        idxs = []
        for p in points:
            j = int(np.argmin(np.linalg.norm(coords - p, axis=1)))
            idxs.append(j)
        return idxs

    # --- 像素→厘米比例 ---
    def pixel_to_cm_scale():
        return float(pixel_dist_to_cm(np.array([0., 0.]), np.array([1., 0.])))
    PIXEL_TO_CM = pixel_to_cm_scale()

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords])

    climber = {
        "height": g.climber[0][0].item(),
        "ape_index": g.climber[0][1].item(),
        "flexibility": g.climber[0][2].item(),
        "leg_len_factor": g.climber[0][3].item(),
        "arm_span": g.climber[0][4].item(),
        "leg_span": g.climber[0][5].item()
    }
    arm_reach, leg_reach = get_reach_ranges(climber)

    # --- Standing Reach 及 75% 阈值（cm）---
    H = float(climber["height"])
    W = float(climber["arm_span"])
    standing_reach_cm = 55.581 + (0.121 * H) + (0.814 * W)
    reach_guard_cm = 0.75 * standing_reach_cm

    # 迭代直到成功或超步数
    for step in range(1, max_steps + 1):
        model.eval()
        with torch.no_grad():
            logits = model(g)
            coords = g.x[:, :2].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        # 成功：目标点非 0
        if preds[goal_idx] != 0:
            return step

        # 可及集合
        hand_idx = np.where((preds == 1) | (preds == 3))[0]
        foot_idx = np.where((preds == 2) | (preds == 3))[0]
        if len(hand_idx) == 0 and len(foot_idx) == 0:
            return -1

        # 决定动手还是动脚（直立角规则）
        hand_c, foot_c, hands_np, feet_np = centers_from_state(g)
        angle_deg = angle_to_vertical_deg(hand_c, foot_c)
        if angle_deg is None:
            move_what = "hands" if len(hand_idx) > 0 else ("feet" if len(foot_idx) > 0 else None)
        else:
            move_what = "feet" if angle_deg > angle_threshold_deg else "hands"

        # 姿态保护：若手脚竖直间距 > StandingReach 的 75%，优先移动“离目标更远”的那一类
        if (hand_c is not None) and (foot_c is not None):
            sep_cm = abs(hand_c[1] - foot_c[1]) * PIXEL_TO_CM
            if sep_cm > reach_guard_cm:
                hand_goal_cm = pixel_dist_to_cm(hand_c, goal_xy)
                foot_goal_cm = pixel_dist_to_cm(foot_c, goal_xy)
                if (hand_goal_cm >= foot_goal_cm and len(hand_idx) > 0) or len(foot_idx) == 0:
                    move_what = "hands"
                elif len(foot_idx) > 0:
                    move_what = "feet"

        # ---- 动手（脚上方优先）----
        if move_what == "hands" and len(hand_idx) > 0:
            hand_candidates = list(hand_idx)
            if (prefer_hands_above_feet and feet_np.size > 0):
                feet_y_min = float(feet_np[:, 1].min())
                thresh = feet_y_min - y_above_margin_hand
                above_feet = [i for i in hand_candidates if coords[i][1] < thresh]
                if len(above_feet) > 0:
                    hand_candidates = above_feet

            hand_dists_goal = sorted(
                [(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in hand_candidates],
                key=lambda x: x[1]
            )
            first_hand = hand_dists_goal[0][0]
            selected_hand_indices = [first_hand]

            if len(hand_candidates) > 1:
                first_xy = coords[first_hand]
                candidates = [i for i in hand_candidates if i != first_hand]
                second_hand = min(candidates, key=lambda j: pixel_dist_to_cm(coords[j], first_xy))
                selected_hand_indices.append(second_hand)
            elif len(hand_idx) > 1:
                first_xy = coords[first_hand]
                candidates = [i for i in hand_idx if i != first_hand]
                second_hand = min(candidates, key=lambda j: pixel_dist_to_cm(coords[j], first_xy))
                selected_hand_indices.append(second_hand)

            selected_hands = [coords[i] for i in selected_hand_indices]
            if len(selected_hands) > 0:
                g.hands = torch.tensor(selected_hands, dtype=torch.float, device=device)

        # ---- 动脚（朝手中心对齐）----
        elif move_what == "feet" and len(foot_idx) > 0:
            current_hand_indices = map_points_to_indices(hands_np, coords) if hands_np.size else []
            used_idxs = set(current_hand_indices)

            base_foot_candidates = [i for i in foot_idx if i not in used_idxs]
            if len(base_foot_candidates) == 0:
                selected_feet_indices = []
            else:
                foot_candidates = base_foot_candidates
                if foot_below_hands and hands_np.size > 0:
                    hand_y_max = float(np.max(hands_np[:, 1]))
                    filtered = [i for i in base_foot_candidates if coords[i][1] > hand_y_max + y_margin]
                    if len(filtered) > 0:
                        foot_candidates = filtered

                if len(foot_candidates) <= 2:
                    foot_dists = sorted([(i, pixel_dist_to_cm(coords[i], goal_xy)) for i in foot_candidates],
                                        key=lambda x: x[1])
                    selected_feet_indices = [i for i, _ in foot_dists[:2]]
                else:
                    cx = float(np.mean(hands_np[:, 0])) if hands_np.size > 0 else \
                         float(np.mean([coords[i][0] for i in foot_candidates]))
                    def d_goal(i): return pixel_dist_to_cm(coords[i], goal_xy)
                    best = None
                    cand = foot_candidates
                    for a in range(len(cand)):
                        for b in range(a+1, len(cand)):
                            i, j = cand[a], cand[b]
                            feet_center_x = 0.5 * (coords[i][0] + coords[j][0])
                            align_pen = abs(feet_center_x - cx) * PIXEL_TO_CM
                            score = foot_align_w * align_pen + foot_goal_w * (d_goal(i) + d_goal(j))
                            if (best is None) or (score < best[0]):
                                best = (score, (i, j))
                    selected_feet_indices = list(best[1]) if best is not None else []

            selected_feet = [coords[i] for i in selected_feet_indices]
            if len(selected_feet) > 0:
                g.feet = torch.tensor(selected_feet, dtype=torch.float, device=device)

        # ---- 重算节点特征（与原逻辑一致）----
        hands = g.hands.cpu().numpy()
        feet  = g.feet.cpu().numpy()
        new_x = g.x.clone().cpu().numpy()
        for i, p in enumerate(coords):
            hand_dists = [pixel_dist_to_cm(p, h) for h in hands]
            foot_dists = [pixel_dist_to_cm(p, f) for f in feet]
            is_hand_point = int(min(hand_dists) <= epsilon_cm)
            is_foot_point = int(min(foot_dists) <= epsilon_cm)
            new_x[i, 2] = np.clip(np.mean(hand_dists) / arm_reach, 0.0, 3.0)
            new_x[i, 3] = np.clip(np.min(hand_dists)  / arm_reach, 0.0, 3.0)
            new_x[i, 4] = np.clip(np.mean(foot_dists) / leg_reach, 0.0, 3.0)
            new_x[i, 5] = np.clip(np.min(foot_dists)  / leg_reach, 0.0, 3.0)
            new_x[i, 6] = float(is_hand_point)
            new_x[i, 7] = float(is_foot_point)
        g.x = torch.tensor(new_x, dtype=torch.float, device=device)

    # 超过最大步数仍未到达
    return -1


def plot_grouped_step_bars_counts(casual_steps, skilled_steps, elite_steps):
    def ok_steps(a):
        a = np.array(a)
        return a[a >= 0]

    C = ok_steps(casual_steps)
    S = ok_steps(skilled_steps)
    E = ok_steps(elite_steps)

    step_vals = sorted(set(np.unique(C)) | set(np.unique(S)) | set(np.unique(E)))
    if len(step_vals) == 0:
        print("No successful climbs to plot.")
        return

    def counts(arr):
        return np.array([np.sum(arr == v) for v in step_vals], dtype=int)

    c_cnt = counts(C);  s_cnt = counts(S);  e_cnt = counts(E)

    def success_rate(a):
        a = np.array(a)
        return float(np.mean(a >= 0)) if len(a) else 0.0
    rC, rS, rE = success_rate(casual_steps), success_rate(skilled_steps), success_rate(elite_steps)

    x = np.arange(len(step_vals))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_c = ax.bar(x - w, c_cnt, width=w, label=f'Casual (n={len(C)}, rate={rC:.0%})')
    bars_s = ax.bar(x     , s_cnt, width=w, label=f'Skilled (n={len(S)}, rate={rS:.0%})')
    bars_e = ax.bar(x + w, e_cnt, width=w, label=f'Elite (n={len(E)}, rate={rE:.0%})')

    # 给每个柱子添加数值标签
    def add_labels(bar_container):
        try:
            ax.bar_label(bar_container, padding=2, fontsize=9)
        except AttributeError:
            # 兼容老版本 Matplotlib
            for b in bar_container:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h, f'{int(h)}',
                        ha='center', va='bottom', fontsize=9)

    for bc in (bars_c, bars_s, bars_e):
        add_labels(bc)

    # 头顶留点空间，避免数字被挡
    top = max([c_cnt.max(initial=0), s_cnt.max(initial=0), e_cnt.max(initial=0)])
    ax.set_ylim(0, top * 1.15 + 1)

    ax.set_xticks(x)
    ax.set_xticklabels(step_vals)
    ax.set_xlabel('Steps to goal (successful only)')
    ax.set_ylabel('Count')
    ax.set_title('Step distribution by group (counts)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.show()