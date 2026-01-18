import numpy as np
import matplotlib.pyplot as plt

# Assumptions: These imports exist in your environment
from data_generation2 import generate_central_patterns
from generate_var import generate_variations
# You will also need the training/testing functions imported:
# from your_module import train_perceptron_with_early_stopping, compute_global_threshold, test_on_noisy_patterns
# from your_module import train_perceptron_reservoir, test_perceptron

# ==========================================
# 1. SHARED CONFIGURATION
# ==========================================
# Unified list of sizes for the X-axis (N_C and N_R)
N_values = [100, 500, 1000, 5000, 10000, 20000]
n_runs = 10  

# ==========================================
# 2. SIMULATION 1: CORTICAL EXPANSION
# ==========================================
print("\n=== Starting Simulation 1: Cortical Expansion ===")

# Parameters specific to Sim 1
P_1 = 100
N_S_1 = 100
f_1 = 0.1
q_1 = 0.1
deltaS_learn_1 = 0.1
deltaS_test_1 = 0.2
eta_1 = 0.001

# J Generation Params
beta = 10
c = 2.0
mJ = beta / np.sqrt(N_S_1)
sigma = c / np.sqrt(N_S_1)

mean_errors_1 = []
std_errors_1 = []

for N_C in N_values:
    print(f"  Testing Cortical Size N_C = {N_C}")
    run_errors = []
    
    for r in range(n_runs):
        # Generate Data
        labels = np.random.choice([-1, 1], size=P_1)
        central_patterns = generate_central_patterns(P_1, N_S_1)
        
        # Generate Sparse J Matrix
        Q = (np.random.rand(N_C, N_S_1) < q_1).astype(float)
        J_dense = np.random.normal(mJ, sigma, size=(N_C, N_S_1))
        J = J_dense * Q
        
        # Train
        W, _ = train_perceptron_with_early_stopping(
            central_patterns, labels, J, f_1, 
            deltaS_learn=deltaS_learn_1, eta=eta_1
        )
        
        # Recalculate Threshold
        noisy_train, _, _ = generate_variations(
            central_patterns, labels, V=10, flip_prob=deltaS_learn_1 / 2
        )
        T = compute_global_threshold(central_patterns, noisy_train, J, f_1)
        
        # Test
        err = test_on_noisy_patterns(W, central_patterns, labels, J, T, f_1, deltaS_test=deltaS_test_1)
        run_errors.append(err)
        
    mean_errors_1.append(np.mean(run_errors))
    std_errors_1.append(np.std(run_errors))

# ==========================================
# 3. SIMULATION 2: RESERVOIR
# ==========================================
print("\n=== Starting Simulation 2: Reservoir ===")

# Parameters specific to Sim 2
P_2 = 100
N_S_2 = 100
f_2 = 0.1
deltaS_learn_2 = 0.1
deltaS_test_2 = 0.2
eta_2 = 0.01

# Reservoir Stability Params
g_rec_factor = 0.7
g_input_factor = 2.0
mu_in = 2.0 / np.sqrt(N_S_2)
sigma_in = g_input_factor / np.sqrt(N_S_2)

mean_errors_2 = []
std_errors_2 = []

for N_R in N_values:
    print(f"  Testing Reservoir Size N_R = {N_R}")
    
    # Recalculate scaling factors dependent on N_R
    mu_rec = -1.0 / np.sqrt(N_R)
    sigma_rec = g_rec_factor / np.sqrt(N_R)
    
    run_errors = []
    
    for r in range(n_runs):
        # Generate Data
        labels = np.random.choice([-1, 1], size=P_2)
        central_patterns = generate_central_patterns(P_2, N_S_2)
        
        # Generate Reservoir Matrices
        J_in = np.random.normal(mu_in, sigma_in, (N_R, N_S_2)).astype(np.float32)
        J_rec = np.random.normal(mu_rec, sigma_rec, (N_R, N_R)).astype(np.float32)
        
        # Train
        W, bias = train_perceptron_reservoir(
            central_patterns, labels, J_in, J_rec, f_2, 
            deltaS_learn=deltaS_learn_2, eta=eta_2
        )
        
        # Test
        err = test_perceptron(W, bias, central_patterns, labels, J_in, J_rec, f_2, deltaS_test=deltaS_test_2)
        run_errors.append(err)
        
    mean_errors_2.append(np.mean(run_errors))
    std_errors_2.append(np.std(run_errors))

# ==========================================
# 4. COMBINED PLOTTING
# ==========================================
plt.figure(figsize=(10, 7))

# Convert lists to arrays for easier math in plotting
x_vals = np.array(N_values)
y_mean_1 = np.array(mean_errors_1)
y_std_1 = np.array(std_errors_1)

y_mean_2 = np.array(mean_errors_2)
y_std_2 = np.array(std_errors_2)

# --- Plot Simulation 1 (Cortical) ---
plt.plot(x_vals, y_mean_1, marker='o', lw=2, color='crimson', label=f'Cortical Expansion (N_S={N_S_1})')
plt.fill_between(x_vals, y_mean_1 - y_std_1, y_mean_1 + y_std_1, color='crimson', alpha=0.2)

# --- Plot Simulation 2 (Reservoir) ---
plt.plot(x_vals, y_mean_2, marker='s', lw=2, color='darkorange', label=f'Reservoir (N_S={N_S_2})')
plt.fill_between(x_vals, y_mean_2 - y_std_2, y_mean_2 + y_std_2, color='darkorange', alpha=0.2)

# --- Logarithmic Scales & Labels ---
plt.xscale('log')
plt.yscale('log')

plt.xlabel("Number of Neurons (N)", fontsize=12, fontweight='bold')
plt.ylabel("Classification Error (log scale)", fontsize=12, fontweight='bold')
plt.title("Comparison: Cortical Expansion vs Reservoir Performance", fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()