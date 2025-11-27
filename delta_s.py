##we are considering a single pattern!
import numpy as np

def compute_delta_s(center, samples):
    center = np.asarray(center).astype(float).ravel()
    samples = np.asarray(samples).astype(float)
    if samples.shape[0] == 0:
        return 0.0
    N = center.size
    diffs = [np.sum(np.abs(sample - center)) / N for sample in samples]
    return 2 * np.mean(diffs)
