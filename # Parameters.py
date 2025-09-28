# Parameters
P = 1000
N_S = 1000
N_C = 10000
deltaS_learn = 0.1   # training noise
deltaS_test = 0.2    # testing noise
eta = 0.001
f_values = np.linspace(0.001, 0.1, 15)
final_errors = []

# Fixed labels and centers
labels = np.random.choice([-1, 1], size=P)
central_patterns = generate_central_patterns(P, N_S)

for f in f_values:
    print(f"Training with f = {f:.3f}")

    # Random projection J
    J = np.random.normal(0, 2/np.sqrt( N_S), size=(N_C, N_S))

    # Use early-stopping training on ΔS = 0.1 noisy patterns
    W, _ = train_perceptron_with_early_stopping(
        central_patterns, labels, J, f,
        deltaS_learn=deltaS_learn, eta=eta
    )

    # Compute threshold (reused for test too)
    noisy_train, _, _ = generate_variations(central_patterns, labels, V=10, flip_prob=deltaS_learn / 2)
    T = compute_global_threshold(central_patterns, noisy_train, J, f)

    # Test on ΔS = 0.2 noisy variants
    test_error = test_on_noisy_patterns(W, central_patterns, labels, J, T, f=f, deltaS_test=deltaS_test)
    final_errors.append(test_error)

# Plotting: error vs f
plt.figure(figsize=(8, 5))
plt.plot(f_values, final_errors, marker='o', lw=2, label='Perceptron (train ΔS=0.1, test ΔS=0.2)')
plt.xlabel("Cortical sparsity $f$")
plt.ylabel("Classification error on noisy patterns (ΔS=0.2)")
plt.title("Figure 4C: Perceptron error vs sparsity $f$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
