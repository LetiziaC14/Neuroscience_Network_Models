#patterns shape(V*P, N_s).
#Check return the number of flipped values in patterns wrt central pattern. 
# flip_prob=$\Delta S/2$
import numpy as np

seed = 42
import random
np.random.seed(seed)
random.seed(seed)

def generate_variations(central_patterns, labels, V, flip_prob):
    P, N_s = central_patterns.shape
    
    # Convert labels to numpy array and ensure it has the correct size
    labels = np.asarray(labels)
    if labels.ndim == 0:
        # Scalar case - expand to array
        labels = np.full(P, labels)
    elif len(labels) != P:
        # If labels has wrong size, auto-generate sequential labels
        if len(labels) == 1:
            # Common case: labels from previous cell with different P
            labels = np.arange(P)
        else:
            raise ValueError(f"labels length ({len(labels)}) must match number of patterns P ({P})")
    
    # Handle V=0 case
    if V == 0:
        return np.zeros((0, N_s)), np.zeros(0), np.zeros(0)
    
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
