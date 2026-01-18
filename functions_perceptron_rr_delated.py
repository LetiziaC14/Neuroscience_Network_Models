# ==========================================
# 1. FUNZIONI DI SUPPORTO (Dinamica & Dati)
# ==========================================

def generate_central_patterns(P, N):
    """Genera P pattern binari random di dimensione N"""
    return np.random.choice([0, 1], size=(P, N), p=[0.5, 0.5])

def generate_variations(patterns, labels, V, flip_prob):
    """
    Genera varianti rumorose dei pattern (flip dei bit).
    Ritorna: (patterns_rumorosi, labels_espanse, indici_originali)
    """
    P, N = patterns.shape
    noisy_patterns = []
    expanded_labels = []
    
    for i in range(P):
        pat = patterns[i]
        lab = labels[i]
        for _ in range(V):
            # Crea rumore flippando bit con probabilità flip_prob
            noise_mask = np.random.rand(N) < flip_prob
            new_pat = np.logical_xor(pat, noise_mask).astype(int)
            noisy_patterns.append(new_pat)
            expanded_labels.append(lab)
            
    return np.array(noisy_patterns), np.array(expanded_labels), None

def run_reservoir_dynamics(inputs, J_in, J_rec, tau=0.01, dt=0.001, steps=200):
    """
    Simula la rete ricorrente fino allo stato stazionario.
    inputs: (Batch_Size, N_S)
    Output: (Batch_Size, N_R) - Stato stazionario
    """
    # 1. Calcolo Input Drive costante (Statico)
    # I_drive ha shape (Batch, N_R)
    I_drive = inputs @ J_in.T
    
    # 2. Inizializzazione stato (Batch, N_R)
    a = np.zeros_like(I_drive, dtype=np.float32)
    
    # 3. Loop Temporale (Eulero)
    for _ in range(steps):
        # Calcolo campo ricorrente per tutto il batch
        rec_input = a @ J_rec.T
        
        total_input = I_drive + rec_input
        
        # Dinamica: tau * da/dt = -a + ReLU(Input)
        da = (-a + np.maximum(0, total_input)) / tau
        a += da * dt
        
    return a

def calibrate_bias(activity, f):
    """
    Calcola una soglia (bias) scalare globale basata sulla distribuzione
    di attività per ottenere una sparsità media f.
    activity: (Batch, N) array di stati analogici
    f: target sparsity (frazione di neuroni attivi)
    """
    # Appiattiamo l'attività per guardare la distribuzione globale
    # (Oppure si può fare la media delle soglie per pattern, qui usiamo percentile globale)
    flat_activity = activity.flatten()
    
    # Se f è 0 o 1 gestiamo i casi limite
    if f <= 0: return np.max(flat_activity) + 1.0
    if f >= 1: return np.min(flat_activity) - 1.0
    
    # Troviamo il valore che lascia il top f% dei neuroni sopra la soglia.
    # Questo corrisponde al percentile (1-f).
    bias = np.percentile(flat_activity, (1 - f) * 100)
    
    return bias


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. HELPER FUNCTIONS (Mocking missing data gen)
# ==========================================
def generate_central_patterns(P, N_S):
    # Generates random binary patterns (-1, 1) or (0, 1)
    return np.random.choice([0, 1], size=(P, N_S))

def generate_variations(patterns, labels, V, flip_prob):
    # Generates noisy variations
    P, N = patterns.shape
    variations = []
    var_labels = []
    
    for i in range(P):
        for _ in range(V):
            noise_mask = np.random.rand(N) < flip_prob
            var_pat = patterns[i].copy()
            # Flip bits (0->1, 1->0)
            var_pat[noise_mask] = 1 - var_pat[noise_mask]
            variations.append(var_pat)
            var_labels.append(labels[i])
            
    return np.array(variations), np.array(var_labels), None

# ==========================================
# 1. RESERVOIR DYNAMICS (Recurrent Physics)
# ==========================================
def run_reservoir_dynamics(inputs, J_in, J_rec, g_scale=1.0, bias=-1.0, steps=30):
    """
    Simulates the Recurrent Reservoir.
    
    Args:
        inputs: Input patterns (Batch x N_S)
        J_rec: Base recurrent matrix (inhibitory)
        g_scale: The tuning parameter 'g'. Scales J_rec.
        bias: Fixed threshold term (bias = -T). If T=1, bias=-1.0.
    """
    n_samples = inputs.shape[0]
    N_R = J_rec.shape[0]
    
    # Initialize states (Firing rates)
    a_R = np.zeros((n_samples, N_R))
    
    # Input Drive (Fixed)
    H_in = inputs @ J_in.T
    
    # Dynamics Loop
    # We explicitly scale J_rec by g_scale here
    J_rec_effective = g_scale * J_rec 
    
    for _ in range(steps):
        # Recurrent Input
        H_rec = a_R @ J_rec_effective.T
        
        # Total Potential
        H_total = H_in + H_rec + bias
        
        # Activation Function (Heaviside/Step for binary neurons)
        # Using 0.5 as cutoff for Heaviside step on potentials
        a_R = (H_total > 0).astype(float)
        
    return a_R

# ==========================================
# 2. TUNING FUNCTION (The "RC" Method)
# ==========================================
def tune_inhibition_g(inputs, J_in, J_rec, target_f, fixed_bias=-1.0, tolerance=0.005):
    """
    Adjusts the recurrent gain 'g' (inhibition strength) to meet target sparsity.
    Replaces 'calibrate_bias'.
    
    Logic: 
    - J_rec has negative mean (Inhibitory).
    - Increasing g_scale (magnitude) -> More Inhibition -> Lower Activity (f).
    - Decreasing g_scale -> Less Inhibition -> Higher Activity (f).
    """
    g_min, g_max = 0.0, 50.0  # Search range for gain
    g_opt = 1.0
    
    for _ in range(20): # Binary search steps
        g_mid = (g_min + g_max) / 2
        
        # Run dynamics with current g
        activity = run_reservoir_dynamics(inputs, J_in, J_rec, g_scale=g_mid, bias=fixed_bias)
        current_f = np.mean(activity)
        
        if abs(current_f - target_f) < tolerance:
            g_opt = g_mid
            break
            
        # In an inhibitory network:
        # If activity is too HIGH, we need MORE inhibition (increase g)
        if current_f > target_f:
            g_min = g_mid 
        # If activity is too LOW, we need LESS inhibition (decrease g)
        else:
            g_max = g_mid
            
    return g_mid

# ==========================================
# 3. TRAINING DEL PERCEPTRON
# ==========================================
def train_perceptron_reservoir(
    central_patterns, labels, 
    J_in, J_rec, 
    f, 
    deltaS_learn=0.1, eta=0.001, 
    max_epochs=50, patience=3
):
    P, N_S = central_patterns.shape
    N_R = J_rec.shape[0]
    flip_prob = deltaS_learn / 2

    # Weights Init
    W = np.random.normal(0, 1 / np.sqrt(N_R), size=N_R)
    block_errors = []
    
    # --- STEP 1: FIX T=1 AND TUNE G ---
    # Notes: "In the RC model, T=1 and relative strength of inhibition g is used to control sparsity."
    # bias = -T = -1.0
    fixed_bias = -1.0
    
    # Find the optimal Recurrent Gain 'g'
    g_opt = tune_inhibition_g(central_patterns, J_in, J_rec, f, fixed_bias=fixed_bias)
    
    # print(f"   [Tuning] Target f={f:.2f} -> Found g={g_opt:.2f}")

    for epoch in range(max_epochs):
        # Generate Noisy Batch
        noisy_patterns, epoch_labels, _ = generate_variations(central_patterns, labels, V=1, flip_prob=flip_prob)
        
        # Run Dynamics with TUNED g and FIXED bias
        sparse_states = run_reservoir_dynamics(
            noisy_patterns, J_in, J_rec, 
            g_scale=g_opt, bias=fixed_bias
        )
        
        # Centering (x - f)
        states_centered = sparse_states - f
        
        # Perceptron Update
        errors_in_epoch = 0
        for i in range(len(epoch_labels)):
            x_c = states_centered[i]
            y = epoch_labels[i]
            
            if y * np.dot(W, x_c) <= 0:
                W += eta * y * x_c 
                errors_in_epoch += 1
        
        err_rate = errors_in_epoch / len(epoch_labels)
        block_errors.append(err_rate)
        
        if len(block_errors) > patience and np.mean(block_errors[-patience:]) < 1e-3:
            break
            
    return W, g_opt, fixed_bias

def test_perceptron(W, g_opt, fixed_bias, central_patterns, labels, J_in, J_rec, f, deltaS_test=0.2):
    # Test set
    noisy_test, test_labels, _ = generate_variations(central_patterns, labels, V=5, flip_prob=deltaS_test/2)
    
    # Run dynamics using the SAME parameters found in training
    sparse_states = run_reservoir_dynamics(
        noisy_test, J_in, J_rec, 
        g_scale=g_opt, bias=fixed_bias
    )
    
    # Predict
    preds = np.sign((sparse_states - f) @ W)
    preds[preds == 0] = 1
    
    error = 1 - np.mean(preds == test_labels)
    return error

# ==========================================
# 4. SIMULATION MAIN LOOP
# ==========================================

# Parametri
P = 100             # Numero pattern
N_S = 100           # Input size
N_R = 1000          # Reservoir size
deltaS_learn = 0.1
deltaS_test = 0.2
eta = 0.01          
f_values = np.linspace(0.01, 0.2, 15)

# Init Params
g_input_factor = 2.0
mu_rec = -1.0 / np.sqrt(N_R)    # Negative mean for inhibition
sigma_rec = 1.0 / np.sqrt(N_R)  # Base sigma (will be scaled by g)

mu_in = 2.0 / np.sqrt(N_S)
sigma_in = g_input_factor / np.sqrt(N_S)

n_runs = 10  

all_results_mean = []
all_results_std = []

print(f"Avvio simulazione: RC Model (Fixed T=1, Tuning g)")

for f in f_values:
    print(f"--- Testing Sparsity f = {f:.3f} ---")
    run_errors = []
    
    for r in range(n_runs):
        # 1. Gen Data
        labels = np.random.choice([-1, 1], size=P)
        central_patterns = generate_central_patterns(P, N_S)
        
        # 2. Gen Matrices
        J_in = np.random.normal(mu_in, sigma_in, (N_R, N_S)).astype(np.float32)
        J_rec = np.random.normal(mu_rec, sigma_rec, (N_R, N_R)).astype(np.float32)
        
        # 3. Train (Returns W and the tuned parameters)
        W, g_opt, fixed_bias = train_perceptron_reservoir(
            central_patterns, labels, J_in, J_rec, f, 
            deltaS_learn=deltaS_learn, eta=eta
        )
        
        # 4. Test
        err = test_perceptron(W, g_opt, fixed_bias, central_patterns, labels, J_in, J_rec, f, deltaS_test=deltaS_test)
        run_errors.append(err)
        
    all_results_mean.append(np.mean(run_errors))
    all_results_std.append(np.std(run_errors))

# Plot
plt.figure(figsize=(8, 6))
plt.errorbar(
    f_values, all_results_mean, yerr=all_results_std,
    marker='o', lw=2, capsize=4, color='darkred', label='Recurrent Model')

plt.xlabel("Sparsity $f_R$", fontsize=12, fontweight='bold')
plt.ylabel(f"Classification Error", fontsize=12, fontweight='bold')
plt.title(f"Perceptron Error (Fixed T=1, Tuned g)", fontsize=14)
plt.grid(True, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()