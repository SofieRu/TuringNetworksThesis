"""
Illustrative comparison: what eigenvectors look like for homogeneous vs
heterogeneous rings. Shows that homogeneous eigenvectors are clean sine waves
while heterogeneous eigenvectors are messy mixtures.

Uses the actual ring Jacobians from config 13 to make it authentic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heterogenous_ring_3954 import (
    compute_jacobian,
    find_steady_state,
    build_ring_jacobian_heterogeneous,
)

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a


# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
CONFIG_ID = 43
N_RING = 20
CV_HET = 0.30   # noise level for heterogeneous example
SEED = 42

np.random.seed(SEED)

# ============================================================================
# LOAD BASELINE
# ============================================================================

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']
row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0]

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])
dU, dV, dW = row['dU'], row['dV'], row['dW']
hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}

# ============================================================================
# BUILD BOTH RING JACOBIANS
# ============================================================================

# Homogeneous: use CV=0 so all cells identical
J_ring_hom, _, _ = build_ring_jacobian_heterogeneous(N_RING, baseline_params, hopping, 0.0)
#J_ring_hom, _, _ = build_ring_jacobian_heterogeneous(N_RING, baseline_params, hopping, CV_HET)

# Heterogeneous: use CV_HET
J_ring_het, _, _ = build_ring_jacobian_heterogeneous(N_RING, baseline_params, hopping, CV_HET)


# ============================================================================
# COMPUTE EIGENVECTORS
# ============================================================================

eigvals_hom, eigvecs_hom = np.linalg.eig(J_ring_hom)
eigvals_het, eigvecs_het = np.linalg.eig(J_ring_het)


# ============================================================================
# PICK REPRESENTATIVE EIGENVECTORS
# ============================================================================

def get_u_profile(eigvec, N):
    """Extract u-species spatial profile across cells."""
    reshaped = eigvec.reshape((N, 3))
    profile = np.real(reshaped[:, 0])  # take real part of u-component
    # normalise to unit amplitude for visual comparison
    max_abs = np.max(np.abs(profile))
    if max_abs > 0:
        profile = profile / max_abs
    return profile

# For the homogeneous case, pick the eigenvector that clearly shows m=3 pattern
# (three cycles across the ring — visually distinctive)
def find_eigvec_with_m(eigvals, eigvecs, target_m, N):
    """Find the eigenvector whose FFT dominant wavenumber is target_m,
    preferring one with a real-valued eigenvalue near zero."""
    best_j = None
    best_score = -np.inf
    for j in range(len(eigvals)):
        eigvec = eigvecs[:, j]
        profile = get_u_profile(eigvec, N)
        fft_mag = np.abs(np.fft.fft(profile))
        m_found = np.argmax(fft_mag[:N // 2 + 1])
        if m_found == target_m:
            # Prefer eigenvalues near zero (Turing-relevant)
            score = -np.abs(np.real(eigvals[j]))
            if score > best_score:
                best_score = score
                best_j = j
    return best_j

# Find m=3 mode in homogeneous
j_hom = find_eigvec_with_m(eigvals_hom, eigvecs_hom, target_m=4, N=N_RING)
profile_hom = get_u_profile(eigvecs_hom[:, j_hom], N_RING)

# For heterogeneous: pick any eigenvector near zero (representative "reaction-scale" mode)
real_parts_het = np.real(eigvals_het)
near_zero_indices = np.where(np.abs(real_parts_het) < 4)[0]
# Sort by |Re(λ)| and pick one that gives a nice mixed profile
#j_het = near_zero_indices[np.argmin(np.abs(real_parts_het[near_zero_indices]))]
rng = np.random.default_rng(SEED)
j_het = rng.choice(near_zero_indices)
profile_het = get_u_profile(eigvecs_het[:, j_het], N_RING)


# ============================================================================
# PLOT
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

cells = np.arange(N_RING)

# --- Left: homogeneous ---
ax = axes[0]
ax.plot(cells, profile_hom, 'o-', color='steelblue', linewidth=2.2, markersize=7)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_xlabel('Cell index', fontsize=11)
ax.set_ylabel('Eigenvector amplitude (u-species, normalised)', fontsize=11)
ax.set_title('Homogeneous ring (all cells identical):\n'
             f'eigenvector is a clean sine wave at $k_3$',
             fontsize=12)
ax.grid(alpha=0.3)
ax.set_xticks(np.arange(0, N_RING + 1, 5))

# --- Right: heterogeneous ---
ax = axes[1]
ax.plot(cells, profile_het, 'o-', color='crimson', linewidth=2.2, markersize=7)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_xlabel('Cell index', fontsize=11)
ax.set_title(f'Heterogeneous ring (CV = {CV_HET}):\n'
             f'eigenvector is a mixture of multiple wavenumbers',
             fontsize=12)
ax.grid(alpha=0.3)
ax.set_xticks(np.arange(0, N_RING + 1, 5))

fig.suptitle(
    'Loss of spatial symmetry: eigenvectors go from clean modes to mixed patterns',
    fontsize=13, y=1.02
)

plt.tight_layout()
plt.savefig('eigenvector_comparison_hom_vs_het.png', dpi=200, bbox_inches='tight')
print("Saved: eigenvector_comparison_hom_vs_het.png")
plt.close()