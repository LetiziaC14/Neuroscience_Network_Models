#CHECK 1
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import delta_s
reload(delta_s)

# Use the functions defined earlier
from data_generation1 import generate_central_patterns
from generate_var import generate_variations
from delta_s import compute_delta_s

# Parameters for testing
P = 100
N_s = 100
V = 5
flip_prob = 0.1

# Generate random labels in {-1, +1}
labels = np.random.choice([-1, 1], size=P)

# Generate central patterns 1
central_patterns= generate_central_patterns(P, N_s)
#print("Central Patterns1:\n", central_patterns)

# Generate variations
patterns, full_labels, checks = generate_variations(central_patterns, labels, V, flip_prob)
#print("\nGenerated Variations:\n", patterns)

#Check if an average of 10% of the bits are flipped
print("Average flip proportion:", np.mean(checks) / N_s)

#print Delta S
for i in range(P):
  diffs = [np.sum(np.abs(sample - central_patterns[i])) / central_patterns[i].size for sample in patterns[i*V:(i+1)*V]]
  print("diffs",diffs)
  print("mean",np.mean(diffs))
  print(2*np.mean(diffs))
  delta_s = compute_delta_s(central_patterns[i], patterns[i*V:(i+1)*V])
  print(f"ΔS for pattern {i+1}", delta_s)


#------------------------------------------------------------------------
#Check in the case we have clusters with only central patterns (V=0)
import numpy as np
import matplotlib.pyplot as plt

# Use the functions defined earlier
from data_generation1 import generate_central_patterns
from generate_var import generate_variations
from delta_s import compute_delta_s

# Parameters for testing
P = 5
N_s = 10
V = 0

# Generate random labels in {-1, +1}
labels = np.random.choice([-1, 1], size=P)

# Generate central patterns 1
central_patterns= generate_central_patterns(P, N_s)
print("Central Patterns1:\n", central_patterns)

# Generate variations
patterns, full_labels, checks = generate_variations(central_patterns, labels, V, flip_prob)
print("\nGenerated Variations:\n", patterns)

#20% of the bits are flipped?
#for i in range(P*V):
#  print(checks[i], end=" ")
#print()
print("Average flip proportion:", np.mean(checks) / N_s)

delta_s = compute_delta_s(central_patterns, patterns)
print(f"ΔS = {delta_s:.4f}")
flip_prob = delta_s/2
patterns.shape


#------------------------------------------------------------------------
#CHECK 2
from data_generation2 import generate_central_patterns
from generate_var import generate_variations

# Parameters for testing
P = 5
N_s = 1000
V = 3
flip_prob = 0.3

# Generate random labels in {-1, +1}
labels = np.random.choice([-1, 1], size=P)

# Generate central patterns 2
central_patterns = generate_central_patterns(P, N_s)
print("Central Patterns2:\n", central_patterns)

#generate variations
patterns, full_labels, checks = generate_variations(central_patterns, labels, V, flip_prob)
print("\nGenerated Variations:\n", patterns)

print("Average flip proportion:", np.mean(checks) / N_s)