
import numpy as np

seed = 42
import random
np.random.seed(seed)
random.seed(seed)

def generate_variations(central_patterns, labels, V, flip_prob):
    P, N_s = central_patterns.shape
    check = np.zeros(P * V)
    patterns = np.zeros((P * V, N_s))
    full_labels = np.zeros(P * V)
    for i in range(P):
        for j in range(V):
            flip_mask = np.random.rand(N_s) < flip_prob
            check[i * V + j] = np.sum(flip_mask)
            pattern = np.abs(central_patterns[i] - flip_mask.astype(int))
            patterns[i * V + j] = pattern
            full_labels[i * V + j] = labels[i]
    return patterns, full_labels, check
