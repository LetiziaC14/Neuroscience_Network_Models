import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# PARAMETERS
N_s = 15000
N_r = 30000
np.random.seed(52)

# --- choose input sparsity (center activity)
aS = 0.5

# ---- generate ONE binary center pattern ----
center = (np.random.rand(N_s) < aS).astype(float)
center2 = (np.random.rand(N_s) < aS).astype(float)

# ---- synaptic weights J with POSITIVE mean ----
mJ = 10 / np.sqrt(N_s)
sigmaJ = 2 / np.sqrt(N_s)
J = np.random.normal(mJ, sigmaJ, size=(N_r, N_s))

# ---- feedforward currents H_i^{mu} ----
H = (center - 0.5) @ J.T

# ---- threshold and ReLU ----
T = 0.0
a_R = np.maximum(0, H - T)

print("Fraction of active R-neurons =", np.mean(a_R > 0))

# ---- KDE for a_R>0 ----
a_R_nonzero = a_R[a_R > 0]

xmin = 0
xmax = max(a_R.max(), 1e-6)
x_vals = np.linspace(xmin, xmax, 800)

if len(a_R_nonzero) > 1:
    kde = gaussian_kde(a_R_nonzero)
    pdf_vals = kde(x_vals)
else:
    pdf_vals = np.zeros_like(x_vals)

plt.figure(figsize=(9, 5))
sns.set_style("white")  
# Continuous KDE part
sns.lineplot(x=x_vals, y=pdf_vals, linewidth=2, label="PDF of a_R (a_R>0)")

# Spike at zero (still Matplotlib)
zero_mass = np.mean(a_R == 0)
plt.axvline(0, color='red', linestyle='--', alpha=0.8,
            label=f"Zero-mass = {zero_mass:.3f}")

plt.xlabel(r"Response $a_{R,i}^{(\mu)}$")
plt.ylabel("Density")
plt.title(r"Distribution of cortical responses $a_{R,i}^{(\mu)}$")
plt.legend()
plt.tight_layout()
plt.show()