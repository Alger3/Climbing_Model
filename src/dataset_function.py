from route_parser import pixel_dist_to_cm
import torch
import numpy as np
from torch_geometric.data import Data
from route_parser import PIXEL_TO_CM
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

# Calculate the reach range of hand and leg
def get_reach_ranges(climber):
    height = climber["height"]
    ape_index = climber["ape_index"]
    flexibility = climber["flexibility"]
    leg_len_factor = climber["leg_len_factor"]

    arm_reach = 1.10 * (0.5 * height * ape_index) + flexibility * 4
    leg_reach = 1.05 * (height * leg_len_factor) + flexibility * 2
    
    return arm_reach, leg_reach

# Calculate which holds are reachable
def get_reachable_points(current_hand, current_foot, climber, route):
    hand_reach, foot_reach = get_reach_ranges(climber)
    reachable_hand = []
    reachable_foot = []

    for p in route:
        if any(pixel_dist_to_cm(h,p) <= hand_reach for h in current_hand):
            reachable_hand.append(p)
        if any(pixel_dist_to_cm(f,p) <= foot_reach for f in current_foot):
            reachable_foot.append(p)
    
    return reachable_hand, reachable_foot

def generate_labeled_route_no_sides(route, hand_points, foot_points, climber):
    hand_reach, foot_reach = get_reach_ranges(climber)

    labels = []
    for p in route:
        hand = any(pixel_dist_to_cm(p, h) <= hand_reach for h in hand_points)
        foot = any(pixel_dist_to_cm(p, f) <= foot_reach for f in foot_points)

        if hand and foot:
            labels.append(3)  # both hand and foot can reach
        elif hand:
            labels.append(1)  # only hand can reach
        elif foot:
            labels.append(2)  # only foot can reach
        else:
            labels.append(0)  # cant reach
    return labels

def sample_grap_points_by_radius(route, climber, max_trials=50, k=0.6):
    hand_reach, foot_reach = get_reach_ranges(climber)  # 这里的 hand/foot_reach 已是“单臂/单腿”半径
    base_R = k * min(hand_reach, foot_reach)

    # 若采样不到足够点，按比例逐步放宽半径，直到不超过 max_reach
    for scale in [1.0, 1.25, 1.5, 1.75, 2.0]:
        radius = min(base_R * scale, max(hand_reach, foot_reach))

        for _ in range(max_trials):
            center = route[np.random.randint(len(route))]
            dists_cm = np.array([pixel_dist_to_cm(p, center) for p in route])
            within_idxs = np.where(dists_cm <= radius)[0]

            if len(within_idxs) >= 4:
                selected = np.random.choice(within_idxs, 4, replace=False)
                points = [route[i] for i in selected]

                # 按 y 排：上面两点作为手，下面两点作为脚（按你现有约定）
                sorted_points = sorted(points, key=lambda p: p[1])
                hand_points = sorted_points[:2]
                foot_points = sorted_points[2:]

                # 额外几何约束（可选但很实用）：
                # 1) 手在脚之上
                if max(p[1] for p in hand_points) >= min(p[1] for p in foot_points):
                    continue
                # 2) 两只手之间不要太远（不超过 ~ 单臂+10cm）
                def dist(a,b): return pixel_dist_to_cm(a,b)
                if dist(hand_points[0], hand_points[1]) > (hand_reach + 10):
                    continue
                # 3) 脚之间也别太夸张（不超过 ~ 单腿+10cm）
                if dist(foot_points[0], foot_points[1]) > (foot_reach + 10):
                    continue

                return hand_points, foot_points, center

    return None, None, None

def plot_route_with_grab(route, hand_points, foot_points, center=None, goal=None):
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys,
                s=80, c='0.7', marker='o', label='holds', zorder=1)

    if center is not None:
        plt.scatter(center[0], center[1], color='orange', marker='x', s=150, label='Center')

    for p in hand_points:
        plt.scatter(p[0], p[1], s=150, facecolors='none', edgecolors='red', linewidths=2, marker='s', label='hand')

    for p in foot_points:
        plt.scatter(p[0], p[1], s=150, facecolors='none', edgecolors='purple', linewidths=2, marker='s', label='foot')
        
    if goal is not None:
        gx, gy = goal
        plt.scatter(gx, gy, s=200, facecolors='none', edgecolors='red', linewidths=3, marker='o', label='Goal')

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Grab Position (Pixel)")

    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', label='holds',
               markerfacecolor='0.7', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='s', color='red', label='hands',
               markerfacecolor='none', markersize=10, linestyle='None', linewidth=2),
        Line2D([0], [0], marker='s', color='purple', label='feet',
               markerfacecolor='none', markersize=10, linestyle='None', linewidth=2),
        Line2D([0], [0], marker='o', color='red', label='Goal',
                   markerfacecolor='none', markeredgewidth=3, markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    plt.show()

"""
Below functions are for the reachability model with features dataset
"""

def compute_hold_stats(holds_features):
    areas = np.array([max(1.0, hf.get("shape_area", 0.0) or 0.0) for hf in holds_features], dtype=float)
    circs = np.array([hf.get("shape_circularity", 0.0) or 0.0 for hf in holds_features], dtype=float)
    stats = {
        "area_p30": np.percentile(areas, 30),
        "area_p50": np.percentile(areas, 50),
        "area_p60": np.percentile(areas, 60),
        "area_p70": np.percentile(areas, 70),
        "circ_med": float(np.median(circs)),
        "circ_p40": np.percentile(circs, 40),
        "circ_p60": np.percentile(circs, 60),
    }
    return stats

def compute_hold_stats_v3(holds_features):
    """对本条 route 计算 area/circularity 的分位数，用于自适应阈值"""
    areas = np.array([h.get("shape_area", 0.0) for h in holds_features], dtype=float)
    circs = np.array([h.get("shape_circularity", 0.0) for h in holds_features], dtype=float)

    # 保护：NaN / 0 情况
    areas = np.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
    circs = np.nan_to_num(circs, nan=0.0, posinf=0.0, neginf=0.0)

    def qs(x):
        return {
            "q10": float(np.percentile(x, 10)),
            "q20": float(np.percentile(x, 20)),
            "q25": float(np.percentile(x, 25)),
            "q30": float(np.percentile(x, 30)),
            "q40": float(np.percentile(x, 40)),
            "med": float(np.median(x)),
        }

    return {"area": qs(areas), "circ": qs(circs)}


# Calculate whether climber can catch this hold
def is_grabbable_by_climber(climber, hold_features, use_hand):
    group_base = {
        "elite":   {"area": 500, "circularity": 0.5, "margin": 0.8},
        "skilled": {"area": 1300, "circularity": 0.6, "margin": 0.9},
        "casual":  {"area": 2300, "circularity": 0.7, "margin": 1.0}
    }

    group = climber.get("group")
    base = group_base.get(group)

    strength = climber.get("strength")
    weight = climber.get("weight")

    area_adj = -10 * strength + 5 * weight
    circ_adj = -0.004 * strength + 0.001 * weight

    if use_hand:
        area_scale = 0.95  # 手对面积的要求较低
        circ_scale = 0.95
    else:
        area_scale = 1.2  # 脚需要更大的面积和更高的圆度
        circ_scale = 1.1

    raw_area_thresh = (base["area"] + area_adj) * area_scale
    raw_circ_thresh = (base["circularity"] + circ_adj) * circ_scale

    margin = base["margin"]
    area_thresh = max(0, raw_area_thresh * margin)
    circ_thresh = max(0, raw_circ_thresh * margin)

    return (hold_features.get("shape_area") >= area_thresh) and (hold_features.get("shape_circularity") >= circ_thresh)

def is_grabbable_by_climber_v2(climber, hold_features, use_hand, stats):
    group = climber.get("group")
    if group == "casual":
        base_area = stats["area_p70"]
        base_circ = stats["circ_med"] + 0.03
    elif group == "skilled":
        base_area = stats["area_p50"]
        base_circ = stats["circ_med"]
    else: # elite
        base_area = stats["area_p30"]
        base_circ = stats["circ_med"] - 0.03

    if use_hand:
        area_mult = 0.90
        circ_add = -0.03
    else:
        area_mult = 1.05
        circ_add = +0.02

    weight   = float(climber.get("weight", 70.0))
    strength = float(climber.get("strength", 5.0))
    # 相对 70kg/5级，最多±30% 面积，圆度最多±0.08
    area_adj_mult = np.clip(1.0 + 0.002*(weight-70.0) - 0.03*(strength-5.0), 0.8, 1.2)
    circ_adj_add  = np.clip(0.0007*(weight-70.0) - 0.015*(strength-5.0), -0.05, 0.05)

    area_thresh = max(1.0, base_area * area_mult * area_adj_mult)
    circ_thresh = float(np.clip(base_circ + circ_add + circ_adj_add, 0.55, 0.90))

    # 4) 取当前点的形状
    hold_area = float(hold_features.get("shape_area", 0.0) or 0.0)
    hold_circ = float(hold_features.get("shape_circularity", 0.0) or 0.0)

    return (hold_area >= area_thresh) and (hold_circ >= circ_thresh)

def is_grabbable_by_climber_v3(climber, hold_feat, use_hand, stats):
    """
    更宽松的形状判定：
    - 用路线内分位数做自适应阈值（防止一刀切）
    - 手的要求低于脚
    - 强/弱体质做轻微调节
    """
    A = float(hold_feat.get("shape_area", 0.0))
    C = float(hold_feat.get("shape_circularity", 0.0))
    A = 0.0 if np.isnan(A) else A
    C = 0.0 if np.isnan(C) else C

    # 基准阈值：先按分位数，再给一个硬下限，避免过低
    if use_hand:
        A_thr = max(stats["area"]["q25"], stats["area"]["med"] * 0.50, 150.0)
        C_thr = max(stats["circ"]["q20"], 0.55)
    else:
        A_thr = max(stats["area"]["q40"], stats["area"]["med"] * 0.70, 220.0)
        C_thr = max(stats["circ"]["q30"], 0.58)

    # 体质调节（力量强→更宽松；体重大→略放宽）
    strength = float(climber.get("strength", 50.0))
    weight   = float(climber.get("weight",   65.0))
    # 把 strength 归一到大约 [-1, +1] 的影响范围
    s_gain = (strength - 50.0) / 50.0
    w_gain = (weight   - 65.0) / 65.0

    # 面积阈值最多放宽/收紧 ~15%
    A_thr *= (1.0 - 0.15 * s_gain + 0.05 * w_gain)

    return (A >= A_thr) and (C >= C_thr)


def get_reachability_features_label(route, climber, hand_points, foot_points, holds_features):
    hand_reach, foot_reach = get_reach_ranges(climber)
    stats = compute_hold_stats(holds_features)
    
    labels = []
    for i, p in enumerate(route):
        hold_feat = holds_features[i]

        hand = any(pixel_dist_to_cm(p, h) <= hand_reach for h in hand_points)
        foot = any(pixel_dist_to_cm(p, f) <= foot_reach for f in foot_points)

        hand_graspable = is_grabbable_by_climber_v2(climber, hold_feat, use_hand=True,  stats=stats)
        foot_graspable = is_grabbable_by_climber_v2(climber, hold_feat, use_hand=False,  stats=stats)

        # hand_graspable = is_grabbable_by_climber(climber, hold_feat, use_hand=False)

        hand_reach = hand and hand_graspable
        foot_reach = foot and foot_graspable

        if hand_reach and foot_reach:
            labels.append(3)  # both hand and foot can reach
        elif hand_reach:
            labels.append(1)  # only hand can reach
        elif foot_reach:
            labels.append(2)  # only foot can reach
        else:
            labels.append(0)  # cant reach
    return labels

from collections import Counter  
def get_reachability_features_label_v3(route, climber, hand_points, foot_points, holds_features,
                                       close_override=0.65, dbg=False):
    """
    - dist_ok: 到任一起始手/脚点的最小距离 <= reach
    - shape_ok: 由 is_grabbable_by_climber_v3 判定
    - 近距离豁免：若距离 < close_override*reach，则忽略形状直接通过
      （默认 0.65；你可以试 0.6~0.7）
    """
    hand_R, foot_R = get_reach_ranges(climber)
    stats = compute_hold_stats_v3(holds_features)

    labels = []
    diag = {k:0 for k in ["hand_dist","foot_dist","hand_shape","foot_shape","both","hand","foot","none"]}

    for i, p in enumerate(route):
        d_hand = min(pixel_dist_to_cm(p, h) for h in hand_points)
        d_foot = min(pixel_dist_to_cm(p, f) for f in foot_points)
        dist_ok_hand = (d_hand <= hand_R)
        dist_ok_foot = (d_foot <= foot_R)

        # 形状
        hf = holds_features[i]
        shape_ok_hand = is_grabbable_by_climber_v3(climber, hf, True,  stats)
        shape_ok_foot = is_grabbable_by_climber_v3(climber, hf, False, stats)

        # 近距离豁免
        close_hand = (d_hand <= close_override * hand_R)
        close_foot = (d_foot <= close_override * foot_R)

        hand_ok = dist_ok_hand and (shape_ok_hand or close_hand)
        foot_ok = dist_ok_foot and (shape_ok_foot or close_foot)

        if hand_ok and foot_ok: lab = 3
        elif hand_ok:           lab = 1
        elif foot_ok:           lab = 2
        else:                   lab = 0
        labels.append(lab)

        # 统计
        diag["hand_dist"]  += int(dist_ok_hand)
        diag["foot_dist"]  += int(dist_ok_foot)
        diag["hand_shape"] += int(shape_ok_hand)
        diag["foot_shape"] += int(shape_ok_foot)
        diag["both"]       += int(lab == 3)
        diag["hand"]       += int(lab == 1)
        diag["foot"]       += int(lab == 2)
        diag["none"]       += int(lab == 0)

    if dbg:
        print(f"hand_R={hand_R:.2f} cm, foot_R={foot_R:.2f} cm")
        print("诊断统计:", diag)
        print("最终标签分布:", Counter(labels))

    return labels