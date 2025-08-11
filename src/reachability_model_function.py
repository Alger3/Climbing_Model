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

def build_graph_reachability(route, hand_points, foot_points, climber, labels, epsilon_ratio=0.05):
    hand_R, foot_R = get_reach_ranges(climber)             # 与攀爬者相关的 reach
    eps_hand = max(1e-6, epsilon_ratio * hand_R)           # flag 阈值随 reach 缩放
    eps_foot = max(1e-6, epsilon_ratio * foot_R)

    node_features = []
    for p in route:
        hand_dists = [pixel_dist_to_cm(p, h) for h in hand_points] if len(hand_points)>0 else [1e9]
        foot_dists = [pixel_dist_to_cm(p, f) for f in foot_points] if len(foot_points)>0 else [1e9]

        dmin_h = float(np.min(hand_dists))
        dmin_f = float(np.min(foot_dists))

        feature = [
            float(p[0]), float(p[1]),
            float(np.mean(hand_dists)),
            dmin_h / max(hand_R, 1e-6),                  # 归一化最小手距
            float(np.mean(foot_dists)),
            dmin_f / max(foot_R, 1e-6),                  # 归一化最小脚距
            float(dmin_h <= eps_hand),                   # scaled flag
            float(dmin_f <= eps_foot)
        ]
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # KNN 边
    route_array = np.array(route)
    nbrs = NearestNeighbors(n_neighbors=min(10, len(route)), algorithm='auto').fit(route_array)
    _, indices = nbrs.kneighbors(route_array)
    edges = {(min(i, j), max(i, j)) for i, ns in enumerate(indices) for j in ns if i != j}
    edge_index = torch.tensor(list(edges), dtype=torch.long).T if edges else torch.zeros((2,0), dtype=torch.long)

    climber_feat = torch.tensor([
        climber['height'],
        climber['ape_index'],
        climber['flexibility'],
        climber['leg_len_factor']
    ], dtype=torch.float).unsqueeze(0)

    graph = Data(x=x, edge_index=edge_index, y=y, climber=climber_feat)
    graph.hands = torch.tensor(hand_points, dtype=torch.float) if len(hand_points)>0 else torch.empty((0,2), dtype=torch.float)
    graph.feet  = torch.tensor(foot_points, dtype=torch.float) if len(foot_points)>0 else torch.empty((0,2), dtype=torch.float)
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


# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
def plot_graph_prediction(graph, model, title, goal_xy=None):
    label_colors = {0:'gray', 1:'blue', 2:'orange', 3:'green'}
    label_names  = {0:'unreachable', 1:'hand', 2:'foot', 3:'both'}

    model.eval()
    with torch.no_grad():
        g = graph
        if not hasattr(g, "batch") or g.batch is None:
            g = Batch.from_data_list([g])
        g = g.to(next(model.parameters()).device)

        logits = model(g)
        preds = logits.argmax(dim=1).cpu().numpy()
        coords = g.x[:, :2].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8,6))
        for i, (x,y) in enumerate(coords):
            ax.scatter(x, y, color=label_colors[preds[i]], s=100, edgecolors='black')

        if hasattr(g, 'hands') and g.hands.numel()>0:
            for hx, hy in g.hands.cpu().numpy():
                ax.scatter(hx, hy, s=150, facecolors='none', edgecolors='red', linewidths=2, marker='s')
        if hasattr(g, 'feet') and g.feet.numel()>0:
            for fx, fy in g.feet.cpu().numpy():
                ax.scatter(fx, fy, s=150, facecolors='none', edgecolors='purple', linewidths=2, marker='s')

        if goal_xy is not None:
            gx, gy = goal_xy
            ax.scatter(gx, gy, s=200, facecolors='none', edgecolors='red', linewidths=3, marker='o')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label_names[i],
                   markerfacecolor=label_colors[i], markersize=10, markeredgecolor='black')
            for i in label_colors
        ] + [
            Line2D([0],[0],marker='s',color='r',label='Hand Start',markerfacecolor='none',markeredgewidth=2,markersize=10),
            Line2D([0],[0],marker='s',color='purple',label='Foot Start',markerfacecolor='none',markeredgewidth=2,markersize=10),
            Line2D([0],[0],marker='o',color='red',label='Goal',markerfacecolor='none',markeredgewidth=3,markersize=10),
        ]
        ax.legend(handles=legend_elements, title="Predicted Labels")

        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.invert_yaxis()
        plt.grid(True); plt.tight_layout(); plt.show()


# ------------------------------------------------------------
# 重算节点特征（在推理/仿真中改了 climber 或 hand/foot 后必须调用）
# ------------------------------------------------------------
def recompute_node_features_with_climber(g, epsilon_ratio=0.05):
    climber = {
        "height":      float(g.climber[0,0].item()),
        "ape_index":   float(g.climber[0,1].item()),
        "flexibility": float(g.climber[0,2].item()),
        "leg_len_factor": float(g.climber[0,3].item()),
    }
    hand_R, foot_R = get_reach_ranges(climber)
    eps_hand = max(1e-6, epsilon_ratio * hand_R)
    eps_foot = max(1e-6, epsilon_ratio * foot_R)

    coords = g.x[:, :2].cpu().numpy()
    hands  = g.hands.cpu().numpy() if hasattr(g, "hands") else []
    feet   = g.feet.cpu().numpy()  if hasattr(g, "feet")  else []

    new_x = g.x.clone().cpu().numpy()
    for i, p in enumerate(coords):
        hd = [pixel_dist_to_cm(p, h) for h in hands] if len(hands)>0 else [1e9]
        fd = [pixel_dist_to_cm(p, f) for f in feet]  if len(feet)>0  else [1e9]
        dmin_h = float(np.min(hd)); dmin_f = float(np.min(fd))

        new_x[i, 2] = float(np.mean(hd))
        new_x[i, 3] = dmin_h / max(hand_R, 1e-6)
        new_x[i, 4] = float(np.mean(fd))
        new_x[i, 5] = dmin_f / max(foot_R, 1e-6)
        new_x[i, 6] = float(dmin_h <= eps_hand)
        new_x[i, 7] = float(dmin_f <= eps_foot)

    g.x = torch.tensor(new_x, dtype=torch.float, device=g.x.device)


# ------------------------------------------------------------
# 带探索 + 前瞻的仿真（使用上面的特征重算）
# ------------------------------------------------------------
def simulate_climb_to_goal(
    graph, model, goal_xy, max_steps=30,
    top_k=4, beam=12, min_improve_cm=0.0, epsilon_ratio=0.05
):
    device = next(model.parameters()).device

    def to_batch(g):
        return g.to(device) if hasattr(g, "batch") and g.batch is not None else Batch.from_data_list([g]).to(device)

    def best_candidate_with_lookahead(g, preds, goal_idx, prev_sel):
        coords = g.x[:, :2].cpu().numpy()
        goal_d = np.array([pixel_dist_to_cm(p, goal_xy) for p in coords])

        hand_idx = np.where((preds == 1) | (preds == 3))[0]
        foot_idx = np.where((preds == 2) | (preds == 3))[0]
        if len(hand_idx) == 0 and len(foot_idx) == 0: return None

        hand_cands = sorted([(i, goal_d[i]) for i in hand_idx], key=lambda x: x[1])[:top_k]
        foot_cands = sorted([(i, goal_d[i]) for i in foot_idx], key=lambda x: x[1])[:top_k]

        hand_pairs = list(combinations([i for i,_ in hand_cands], min(2, len(hand_cands)))) or [()]
        foot_pairs = list(combinations([i for i,_ in foot_cands], min(2, len(foot_cands)))) or [()]

        cand_list = []
        for hs in hand_pairs:
            for fs in foot_pairs:
                if prev_sel is not None and (tuple(sorted(hs)) == prev_sel[0] and tuple(sorted(fs)) == prev_sel[1]): 
                    continue
                if set(hs) & set(fs): 
                    continue
                score_now = (sum(goal_d[list(hs)]) if hs else 0.0) + (sum(goal_d[list(fs)]) if fs else 0.0)
                cand_list.append((score_now, hs, fs))
        cand_list.sort(key=lambda x: x[0])
        cand_list = cand_list[:beam]

        best = None
        for _, hs, fs in cand_list:
            g_try = to_batch(copy.deepcopy(g))
            if hs: g_try.hands = torch.tensor(coords[list(hs)], dtype=torch.float, device=device)
            if fs: g_try.feet  = torch.tensor(coords[list(fs)], dtype=torch.float, device=device)
            recompute_node_features_with_climber(g_try, epsilon_ratio)

            with torch.no_grad():
                pred2 = model(g_try).argmax(dim=1).cpu().numpy()

            cand = np.concatenate([np.where((pred2==1)|(pred2==3))[0], np.where((pred2==2)|(pred2==3))[0]])
            min_next = float(np.min(goal_d[cand])) if len(cand)>0 else 1e9

            if (best is None) or (min_next < best[2]):
                best = (hs, fs, min_next)
        return best

    g = to_batch(copy.deepcopy(graph))
    recompute_node_features_with_climber(g)  # 确保初始与 climber 一致

    coords = g.x[:, :2].cpu().numpy()
    goal_idx = int(np.argmin([pixel_dist_to_cm(coord, goal_xy) for coord in coords]))
    prev_sel = None
    prev_best = float('inf')

    for step in range(1, max_steps+1):
        model.eval()
        with torch.no_grad():
            preds = model(g).argmax(dim=1).cpu().numpy()

        plot_graph_prediction(g.cpu(), model, f"Step {step}", goal_xy)

        if preds[goal_idx] != 0:
            print(f"Success: goal reachable (label={preds[goal_idx]}) at Step {step}")
            return

        best = best_candidate_with_lookahead(g, preds, goal_idx, prev_sel)
        if best is None:
            print("No candidates for next step."); return

        hs, fs, best_next = best
        if best_next >= prev_best - min_improve_cm:
            print(f"No obvious improvement (best_next={best_next:.2f} cm, prev={prev_best:.2f} cm). Exploring...")
        prev_best = best_next

        if len(hs)>0: g.hands = torch.tensor(coords[list(hs)], dtype=torch.float, device=device)
        if len(fs)>0: g.feet  = torch.tensor(coords[list(fs)], dtype=torch.float, device=device)
        prev_sel = (tuple(sorted(hs)), tuple(sorted(fs)))

        recompute_node_features_with_climber(g, epsilon_ratio)

    print(f"Climber cannot reach the goal in {max_steps} steps.")
