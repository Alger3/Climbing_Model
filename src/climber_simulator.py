import numpy as np

def simulator(group):
    if group == "casual":
        height = np.random.normal(165,5)
        weight = np.random.normal(70.5,14.78)
        ape_index = np.random.normal(1.00,0.03)
        strength = np.random.randint(60,80)
        flexibility = np.random.randint(4,6)
    elif group == "skilled":
        height = np.random.normal(170,5)
        weight = np.random.normal(69.9,10.24)
        ape_index = np.random.normal(1.02,0.02)
        strength = np.random.randint(75,95)
        flexibility = np.random.randint(6,8)
    elif group == "elite":
        height = np.random.normal(177,8)
        weight = np.random.normal(67.2,13.6)
        ape_index = np.random.normal(1.05,0.03)
        strength = np.random.randint(95,115)
        flexibility = np.random.randint(8,10)
    else:
        raise ValueError("Unknown group type! Please Use 'casual', 'skilled', or 'elite'.")
    
    return {
        "group": group,
        "height": round(height, 1),
        "weight": round(weight,1),
        "ape_index": round(ape_index, 2),
        "strength": strength,
        "flexibility": flexibility
    }