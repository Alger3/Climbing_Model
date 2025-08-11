from route_parser import pixel_dist_to_cm
import torch
import numpy as np
from torch_geometric.data import Data
from route_parser import PIXEL_TO_CM
import matplotlib.pyplot as plt
import math

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

def plot_route_with_grab(route, hand_points, foot_points, center=None):
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys, c='gray', label='Holds')

    if center is not None:
        plt.scatter(center[0], center[1], color='orange', marker='x', s=150, label='Center')

    for p in hand_points:
        plt.scatter(p[0], p[1], color='green', s=120, label='Hand')

    for p in foot_points:
        plt.scatter(p[0], p[1], color='blue', s=120, label='Foot')

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Grab Position (Pixel)")
    plt.show()

"""
Below functions are for the reachability model with features dataset
"""

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

BASE_LOG_AREA_HAND = 7.285095166950276
BASE_LOG_AREA_FOOT = 8.0305395163207
SCALE_LOG_AREA = 1.9669811876841052
CLIP_LOG_AREA = (5.12751431669736, 10.256193714082718)

BASE_CIRC_HAND = 0.7503203596084896
BASE_CIRC_FOOT = 0.8293995267947876
SCALE_CIRC = 0.19719044756542792
CLIP_CIRC = (0.4720785780627363, 0.9450321817290853)

def is_grabbable_by_climber_v2(climber, hold_feat, use_hand: bool, 
                               prob_threshold: float = 0.5):

    if not isinstance(hold_feat, dict):
        return False
    area = hold_feat.get("shape_area", None)
    circ = hold_feat.get("shape_circularity", None)
    if area is None or circ is None:
        return False

    log_area = float(np.log1p(area))
    log_area = float(np.clip(log_area, *CLIP_LOG_AREA))
    circ = float(np.clip(circ, *CLIP_CIRC))

    base_log_area = BASE_LOG_AREA_HAND if use_hand else BASE_LOG_AREA_FOOT
    base_circ = BASE_CIRC_HAND if use_hand else BASE_CIRC_FOOT

    strength = float(climber.get("strength", 90.0))
    flexibility = float(climber.get("flexibility", 6.0))

    k_s_log_area = 0.010  
    k_f_circ = 0.005   

    adj_log_area = base_log_area - k_s_log_area * (strength - 90.0)
    adj_circ = base_circ - k_f_circ * (flexibility - 6.0)

    alpha_circ = 1.2

    z = (log_area - adj_log_area) / max(SCALE_LOG_AREA, 1e-6) \
        + alpha_circ * (circ - adj_circ) / max(SCALE_CIRC, 1e-6)

    prob = 1.0 / (1.0 + math.exp(-z))
    return prob >= prob_threshold

def get_reachability_features_label(route, climber, hand_points, foot_points, holds_features):
    hand_reach, foot_reach = get_reach_ranges(climber)
    
    labels = []
    for i, p in enumerate(route):
        hold_feat = holds_features[i]

        hand = any(pixel_dist_to_cm(p, h) <= hand_reach for h in hand_points)
        foot = any(pixel_dist_to_cm(p, f) <= foot_reach for f in foot_points)

        # TODO: Change the function to version 2
        hand_graspable = is_grabbable_by_climber_v2(climber, hold_feat, use_hand=True)
        foot_graspable = is_grabbable_by_climber_v2(climber, hold_feat, use_hand=False)

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
    
