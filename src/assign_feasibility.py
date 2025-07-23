from math import dist
from route_parser import pixel_dist_to_cm

# Input is sorted route, only consider arm span...
def is_route_feasible_1(climber,points):
    height = climber["height"]
    ape_index = climber["ape_index"]
    weight = climber["weight"]
    flexibility = climber["flexibility"]
    strength = climber["strength"]

    # Add condition to judge complete or not
    reach = height * ape_index
    flex_bonus = (flexibility/10)*0.15
    strength_bonus = (strength/100)*0.2
    weight_penalty = weight*0.05

    max_reach = reach * (1 + strength_bonus + flex_bonus) - weight_penalty

    for i in range(len(points)-1):
        if pixel_dist_to_cm(points[i],points[i+1]) > max_reach:
            return False

    return True

# TODO: Need to complicated the algorithmn

# TODO: Figure what is for
def compute_subcomponents(route,climber_info):
    dx_list, dy_list, switches = [], [], 0
    prev_dx = None

    for i in range(len(route) - 1):
        x0, y0 = route[i]
        x1, y1 = route[i+1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dx_list.append(dx)
        dy_list.append(dy)

        if prev_dx is not None:
            if (dx > 20 and prev_dx < 0) or (dx < -20 and prev_dx > 0):
                switches += 1
        prev_dx = x1 - x0

    return {
        "max_dx": max(dx_list),
        "max_dy": max(dy_list),
        "mean_dx": sum(dx_list)/len(dx_list),
        "mean_dy": sum(dy_list)/len(dy_list),
        "switches": switches,
        "span_ratio": max(dx_list) / (climber_info["height"] * climber_info["ape_index"]),
        "dy_ratio": max(dy_list) / climber_info["height"]
    }