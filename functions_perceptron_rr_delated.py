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