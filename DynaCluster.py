import torch

def cauchy_similarity(z, c, kappa):
    return 1 / (torch.pi * kappa * (1 + ((z - c) ** 2 / kappa ** 2)))

def decision_probability(z, c, kappa):
    s_ij = cauchy_similarity(z, c, kappa)
    return 1 / (1 + torch.exp(-s_ij))

def compute_reward(a_ij, y_ij, v):
    if a_ij == 1 and y_ij == 1:
        return v
    elif a_ij == 1 and y_ij == 0:
        return -v
    else:
        return 0
