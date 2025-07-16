from math import dist
from route_parser import pixel_dist

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
    weight_penalty = weight*0.05

    max_reach = reach * (1 + strength_bonus + flex_bonus) - weight_penalty

    # TODO: I dont know how to define the metric to label whether they can pass
    for i in range(len(points)-1):
        if pixel_dist(points[i],points[i+1]) > max_reach:
            return False

    return True
