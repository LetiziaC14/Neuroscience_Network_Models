
import numpy as np

seed = 42
import random
np.random.seed(seed)
random.seed(seed)

def generate_central_patterns(P, N_s):
    central_patterns = np.zeros((P, N_s), dtype=int)
    half = N_s // 2
    for i in range(P):
        pattern = np.array([1] * half + [0] * (N_s - half))
        np.random.shuffle(pattern)
        central_patterns[i] = pattern
    return central_patterns
    