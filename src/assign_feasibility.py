from math import dist

# TODO: Need to complicated the algorithmn
def is_route_feasible(climber,points):
    height = climber["height"]
    ape_index = climber["ape_index"]
    weight = climber["weight"]
    flexibility = climber["flexibility"]
    strength = climber["strength"]

    # Add condition to judge complete or not
    reach = height * ape_index
    flex_bonus = (flexibility/10)*0.15
    strength_bonus = (strength/100)*0.2
    weight_punish = weight*0.05

    max_reach = reach * (1 + strength_bonus + flex_bonus) - weight_punish

    # TODO: I dont know how to define the metric to label whether they can pass
    for i in range(len(points)-1):
        if dist(points[i],points[i+1]) > max_reach:
            return False

    return True

# TODO: the Climbing wall points are not in order, need to sort them first
# TODO: Or need to only consider the closer point or try to follow the reachibility by Sophie
# TODO: 1. Use random one to pixel and cm 2. Use GNN 3. Use 分类模型
def simplest_feasible_arm_span(climber,points):
    height = climber["height"]
    ape_index = climber["ape_index"]

    arm_span = height * ape_index
    max_dis = max(dist(points[i],points[i+1]) for i in range(len(points)-1))

    return max_dis <= arm_span
