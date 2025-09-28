
import numpy as np
seed = 42
import random
np.random.seed(seed)
random.seed(seed)
def generate_central_patterns(P, N_s):
    return np.random.randint(0, 2, size=(P, N_s))

