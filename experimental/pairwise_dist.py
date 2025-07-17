import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Define distances ----

# Pairwise distances between P1, P2, P3, P4 (6 values for upper triangle)
pairwise_dists = np.array([
    3.6,  # d(P1, P2)
    5.4,  # d(P1, P3)
    6.8,  # d(P1, P4)
    4.2,  # d(P2, P3)
    5.1,  # d(P2, P4)
    3.9   # d(P3, P4)
])

# Distances from point t to P1–P4
Dt = np.array([
    2.5,  # d(t, P1)
    3.1,  # d(t, P2)
    4.8,  # d(t, P3)
    2.9   # d(t, P4)
])

# ---- 2. Pairwise Distance Matrix Visualization ----
# We'll reconstruct the full 4x4 matrix from upper triangle for visualization
matrix = np.zeros((4, 4))
indices = np.triu_indices(4, k=1)
matrix[indices] = pairwise_dists
matrix += matrix.T  # make symmetric

fig1, ax1 = plt.subplots()
cax = ax1.matshow(matrix, cmap='viridis')
fig1.colorbar(cax)
for (i, j), val in np.ndenumerate(matrix):
    ax1.text(j, i, f'{val:.1f}', ha='center', va='center', color='white')
ax1.set_xticks(np.arange(4))
ax1.set_yticks(np.arange(4))
ax1.set_xticklabels([f'P{i+1}' for i in range(4)])
ax1.set_yticklabels([f'P{i+1}' for i in range(4)])
ax1.set_title("Pairwise Distance Matrix")
plt.tight_layout()

# ---- 3. 1D Distance Diagram ----
fig2, ax2 = plt.subplots(figsize=(12, 1))

# Compute marker size for 10-pixel radius
dpi = fig2.dpi
radius_px = 10
points_per_inch = 72
radius_pt = (radius_px / dpi) * points_per_inch
marker_size = (2 * radius_pt) ** 2

# Plot blue (pairwise) and yellow (Dt) distances on 1D line
ax2.scatter(pairwise_dists, np.zeros_like(pairwise_dists),
            facecolors='blue', edgecolors='black', s=marker_size, linewidth=1)

ax2.scatter(Dt, np.zeros_like(Dt),
            facecolors='yellow', edgecolors='black', s=marker_size, linewidth=1)

# Clean formatting
ax2.get_yaxis().set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax2.set_yticks([])
ax2.set_xlabel("Distance values")
ax2.set_title("1D Distribution of Distances")
plt.tight_layout()

plt.show()
