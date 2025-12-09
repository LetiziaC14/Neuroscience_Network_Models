# # delta_c.py
import numpy as np

# def compute_delta_c(central_corticals, noisy_corticals, full_labels, f):
#     """
#     Compute cortical cluster size ΔC based on noisy and central cortical patterns.

#     Parameters:
#         central_corticals: array of shape (P, N_C)
#             Cortical codes for cluster centers.
#         noisy_corticals: array of shape (P * V, N_C)
#             Cortical codes for noisy variants.
#         full_labels: array of shape (P * V,)
#             For each noisy pattern, the index (0..P-1) of its cluster center.
#         f: float
#             Sparseness level.

#     Returns:
#         delta_c: float
#             Estimated cortical cluster size ΔC.
#     """

#     central_corticals = np.asarray(central_corticals).astype(int)   # (P, N_C)
#     noisy_corticals   = np.asarray(noisy_corticals).astype(int)     # (P*V, N_C)
#     full_labels       = np.asarray(full_labels).astype(int)         # (P*V,)

#     # Sanity checks (optional, but useful while debugging)
#     assert noisy_corticals.shape[0] == full_labels.shape[0], \
#         "noisy_corticals and full_labels must have same number of samples"

#     P, N_C = central_corticals.shape

#     # Pick the correct center for each noisy sample via indexing
#     centers_for_noisy = central_corticals[full_labels]             # (P*V, N_C)

#     # Hamming distances: number of 0/1 mismatches for each noisy sample
#     diffs = np.sum(np.abs(noisy_corticals - centers_for_noisy), axis=1)  # (P*V,)

#     avg_diff = np.mean(diffs)
#     epsilon = 1e-8
#     delta_c = avg_diff / (2 * N_C * f * (1 - f) + epsilon)

#     return delta_c
# #computation of delta_c
# import numpy as np


def compute_delta_c_1(center_cortical, noisy_corticals, f):
    """
    Compute cortical cluster size ΔC based on noisy and central cortical patterns.

    Parameters:
        central_corticals: shape (P, N_c), cortical codes for cluster centers
        noisy_corticals: shape (P * V, N_c), cortical codes for noisy variants
        full_labels: shape (P * V,), maps each noisy pattern to its cluster index
        f: sparseness level
    """
    center_cortical = np.asarray(center_cortical).astype(int)
    #center_cortical = np.asarray(center_cortical).astype(int).ravel()
    noisy_corticals = np.asarray(noisy_corticals).astype(int)
    Nc = center_cortical.size
    diffs = [np.sum(np.abs(sample - center_cortical)) for sample in noisy_corticals] #Hamming distance: counts how many 0/1 mismatches there are
    avg_diff = np.mean(diffs)
    epsilon = 1e-8
    delta_c = avg_diff / (2 * Nc * f * (1 - f) + epsilon)

    # Normalize by 2 * N_C * f * (1 - f)
    #delta_c = avg_diff / (2 * Nc* f * (1 - f))
    return delta_c

# Simulation and analysis of cortical size
# It should approximate to  $\sqrt{ \frac{2}{\pi} \, |\log(f)| \, \Delta_S}$ 
# if $\Delta_S$ small.
# Checked.
# The variability inherent in stimuli belonging to the same cluster is amplified in the cortical layer
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_generation1 import generate_central_patterns
    from generate_var import generate_variations
    from delta_s import compute_delta_s

    np.random.seed(42)
    # Function to compute global threshold from all patterns, both central and noisy
    def compute_global_threshold(central_patterns, noisy_patterns, J, f):
        """
        Computes global threshold T from all h_mj values for central and noisy patterns.

        Parameters:
            central_patterns: shape (P, N_s)
            noisy_patterns : shape (P * V, N_s)
            J: synaptic weight matrix (N_c, N_s)
            f: desired cortical sparsity

        Returns:
            T such that only top f fraction of all h_mj are above it
        """
        h_all = []

        # Central patterns net input
        for center in central_patterns:
            # here, center is NOT shape (1, 2) it is a 1D array of shape (2,)
            h_center = J @ (center - 0.5) #shape: N_c
            h_all.append(h_center) #list of vectors

        # Noisy patterns net inputs
        for noisy in noisy_patterns:
            h_noisy = J @ (noisy - 0.5)
            h_all.append(h_noisy)

        # Flatten all h values from all neurons across all patterns
        h_all = np.concatenate(h_all)
        T = np.percentile(h_all, 100 * (1 - f)) #It sets the threshold T such that only the top f fraction of the values in h_all are greater than or equal to T
        return T
    def compute_cortical_patterns(patterns, J, T):
        """
        Compute cortical binary activity patterns from input patterns.

        Parameters:
            patterns: shape (num_patterns, N_s), binary input patterns
            J: synaptic weight matrix, shape (N_c, N_s)
            T: global threshold

        Returns:
            np.array: cortical binary patterns, shape (num_patterns, N_c)
        """
        centered_inputs = patterns - 0.5
        h = centered_inputs @ J.T     # shape: (num_patterns, N_c)
        C = (h >= T).astype(int)
        return C,h


    #parameters
    P = 100    # number of clusters
    V = 5       # noisy samples per cluster
    N_s = 100   # stimulus neurons
    N_c = 1000   # cortical neurons
    f = 0.1   # sparsity level
    flip_prob = 0.1  # noise level (DS ≈ 0.2)

    central_patterns = generate_central_patterns(P, N_s)
    labels = np.arange(P)
    noisy_patterns, full_labels, check = generate_variations(central_patterns, labels, V, flip_prob)

    # synaptic weights
    J = np.random.normal(0, 2 / np.sqrt(N_s), size=(N_c, N_s))

    # threshold
    T = compute_global_threshold(central_patterns, noisy_patterns, J, f)

    # cortical patterns
    central_corticals,h_c = compute_cortical_patterns(central_patterns, J, T)
    print(central_corticals.shape)

    noisy_corticals, h_n= compute_cortical_patterns(noisy_patterns, J, T)

    # delta s
    delta_s = compute_delta_s(central_patterns[0], noisy_patterns[0:V])
    print(f"ΔS = {delta_s:.4f}")

    # delta C
    delta_c = compute_delta_c(central_corticals[0], noisy_corticals[0:V], f)
    print(f"ΔC = {delta_c:.4f}")

    print(np.sqrt((2 / np.pi) * np.abs(np.log(f)) * delta_s))