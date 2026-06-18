# """
# Dispersion relations at discrete ring wavenumbers under parameter noise.

# Setup: Ring of N=10 cells, all sharing the SAME noisy parameter set.
# Computes λ(k_m) at the allowed discrete wavenumbers, not a continuous sweep.

# Per PI feedback: each cell has identical noisy parameters (uniform ring,
# not heterogeneous ring). This gives a well-defined dispersion at the
# discrete k_m values.
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from heterogenous_ring_3954 import compute_jacobian, find_steady_state


# # ============================================================================
# # CONFIG
# # ============================================================================

# CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
# CONFIG_ID = 45
# N_RING = 100                             # number of cells in ring
# N_TRIALS = 10                            # noisy realisations per CV
# CV_VALUES = [0.0, 0.10, 0.20, 0.30]      # CV levels
# SEED = 42

# # Discrete wavenumbers allowed for ring of N cells (periodic boundary)
# # k_m = 2·sin(m·π/N), m = 0, 1, ..., N/2
# M_VALUES = np.arange(0, N_RING // 2 + 1)
# K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

# print(f"Discrete k_m values for N={N_RING}: {K_DISCRETE}")


# # ============================================================================
# # LOAD BASELINE TYPE-I PARAMETER SET
# # ============================================================================

# df = pd.read_csv(CSV_PATH)
# type_i = df[df['classification'] == 'Type-I']
# row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0]

# baseline_params = np.array([
#     row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
#     row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
#     row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
# ])
# baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
# dU, dV, dW = row['dU'], row['dV'], row['dW']

# print(f"\nLoaded config {CONFIG_ID}: {row['config_name']}")
# print(f"  Diffusion (dU, dV, dW) = ({dU}, {dV}, {dW})")
# print(f"  Steady state: ({baseline_ss[0]:.4f}, {baseline_ss[1]:.4f}, {baseline_ss[2]:.4f})")


# # ============================================================================
# # HELPER: dispersion at discrete k_m values
# # ============================================================================

# def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
#     """Return array of max Re(λ) at each discrete k_m."""
#     J = compute_jacobian(ss, params)
#     D = np.diag([dU, dV, dW])
#     max_reals = np.zeros(len(k_discrete))
#     for i, k in enumerate(k_discrete):
#         M = J - k**2 * D
#         max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
#     return max_reals


# # ============================================================================
# # COMPUTE DISPERSION FOR EACH CV (MULTIPLE TRIALS)
# # ============================================================================

# np.random.seed(SEED)
# dispersion_results = {CV: [] for CV in CV_VALUES}

# for CV in CV_VALUES:
#     if CV == 0.0:
#         # Baseline: no noise, one curve
#         disp = compute_discrete_dispersion(baseline_params, baseline_ss,
#                                             dU, dV, dW, K_DISCRETE)
#         dispersion_results[CV].append(disp)
#     else:
#         # Noisy trials
#         sigma = np.sqrt(np.log(1 + CV**2))
#         mu = -sigma**2 / 2
        
#         successful = 0
#         attempts = 0
#         max_attempts = N_TRIALS * 5
        
#         while successful < N_TRIALS and attempts < max_attempts:
#             attempts += 1
#             noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
#             params_noisy = baseline_params * noise_factors
            
#             ss_noisy = find_steady_state(params_noisy)
#             if ss_noisy is None:
#                 continue
            
#             disp = compute_discrete_dispersion(params_noisy, ss_noisy,
#                                                 dU, dV, dW, K_DISCRETE)
#             dispersion_results[CV].append(disp)
#             successful += 1
        
#         print(f"  CV={CV}: {successful}/{N_TRIALS} successful trials")


# # ============================================================================
# # PLOT — 4 SUBPLOTS, ONE PER CV
# # ============================================================================

# fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
# panel_colors = ['black', 'steelblue', 'darkorange', 'crimson']

# for ax, CV, color in zip(axes, CV_VALUES, panel_colors):
#     curves = dispersion_results[CV]
    
#     # Plot all trials as connected dots
#     for disp in curves:
#         ax.plot(K_DISCRETE, disp, 'o-', color=color, linewidth=1.5,
#                 markersize=8, alpha=0.5)
    
#     # Reference: Turing threshold
#     ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
#     # Annotate k_m positions on x-axis
#     ax.set_xticks(K_DISCRETE)
#     ax.set_xticklabels([f'{k:.2f}' for k in K_DISCRETE], rotation=0, fontsize=9)
    
#     # Add m labels above x-axis ticks
#     for m, k in zip(M_VALUES, K_DISCRETE):
#         ax.text(k, ax.get_ylim()[0], f'm={m}', ha='center', va='top',
#                 fontsize=8, color='gray', alpha=0.8)
    
#     title = (f'CV = {CV:.2f} (baseline)' if CV == 0.0
#              else f'CV = {CV:.2f}  ({len(curves)} trials)')
#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel('Wavenumber $k_m$', fontsize=11)
#     ax.grid(alpha=0.3)

# axes[0].set_ylabel('Max Re(λ)', fontsize=11)

# fig.suptitle(
#     f'Discrete Ring Dispersion Under Parameter Noise\n'
#     f'Config {CONFIG_ID} (Type-I), Topology #3954, N={N_RING} cells '
#     f'(uniform ring with shared noisy parameters)',
#     fontsize=13, y=1.02
# )

# plt.tight_layout()
# plt.savefig('dispersion_homogeneous_ring_3954_config13.png', dpi=200, bbox_inches='tight')
# print("\nSaved: dispersion_homogeneous_ring_3954_config13.png")
# plt.close()




"""
Dispersion relations at discrete ring wavenumbers under parameter noise.

Setup: Ring of N cells, all sharing the SAME noisy parameter set.
Computes λ(k_m) at the allowed discrete wavenumbers, not a continuous sweep.
Plots noisy trials against the CV=0 baseline plotted as a black line in all panels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heterogenous_ring_3954 import compute_jacobian, find_steady_state

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
CONFIG_ID = 45
N_RING = 30                             # number of cells in ring
N_TRIALS = 10                            # noisy realisations per CV
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]    # Added 0.4 noise level
SEED = 42

# Discrete wavenumbers allowed for ring of N cells (periodic boundary)
M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

print(f"Discrete k_m values for N={N_RING}: {K_DISCRETE}")


# ============================================================================
# LOAD BASELINE TYPE-I PARAMETER SET
# ============================================================================

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']
row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0]

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])
baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
dU, dV, dW = row['dU'], row['dV'], row['dW']

print(f"\nLoaded config {CONFIG_ID}: {row['config_name']}")


# ============================================================================
# HELPER: dispersion at discrete k_m values
# ============================================================================

def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
    """Return array of max Re(λ) at each discrete k_m."""
    J = compute_jacobian(ss, params)
    D = np.diag([dU, dV, dW])
    max_reals = np.zeros(len(k_discrete))
    for i, k in enumerate(k_discrete):
        M = J - k**2 * D
        max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
    return max_reals


# ============================================================================
# COMPUTE DISPERSION FOR EACH CV (MULTIPLE TRIALS)
# ============================================================================

np.random.seed(SEED)
dispersion_results = {CV: [] for CV in CV_VALUES}

for CV in CV_VALUES:
    if CV == 0.0:
        disp = compute_discrete_dispersion(baseline_params, baseline_ss,
                                            dU, dV, dW, K_DISCRETE)
        dispersion_results[CV].append(disp)
    else:
        sigma = np.sqrt(np.log(1 + CV**2))
        mu = -sigma**2 / 2
        
        successful = 0
        attempts = 0
        max_attempts = N_TRIALS * 5
        
        while successful < N_TRIALS and attempts < max_attempts:
            attempts += 1
            noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
            params_noisy = baseline_params * noise_factors
            
            ss_noisy = find_steady_state(params_noisy)
            if ss_noisy is None:
                continue
            
            disp = compute_discrete_dispersion(params_noisy, ss_noisy,
                                                dU, dV, dW, K_DISCRETE)
            dispersion_results[CV].append(disp)
            successful += 1
        
        print(f"  CV={CV}: {successful}/{N_TRIALS} successful trials")


# ============================================================================
# PLOT — 4 SUBPLOTS (CV 0.1 to 0.4), BASELINE (CV 0.0) IN BLACK EVERYWHERE
# ============================================================================

# ============================================================================
# PLOT — 4 SUBPLOTS (CV 0.1 to 0.4), BASELINE (CV 0.0) IN BLACK EVERYWHERE
# ============================================================================

# ============================================================================
# PLOT — 4 SUBPLOTS (CV 0.1 to 0.4), BASELINE (CV 0.0) IN BLACK EVERYWHERE
# ============================================================================

# ============================================================================
# PLOT — 4 SUBPLOTS (CV 0.1 to 0.4), BASELINE (CV 0.0) IN BLACK EVERYWHERE
# ============================================================================

noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

# 1. FIX: Flatten the baseline array from (1, 16) to (16,)
baseline_disp = dispersion_results[0.0][0]

for ax, CV, color in zip(axes, noisy_cvs, panel_colors):
    curves = dispersion_results[CV]
    
    # Plot baseline (CV = 0.0) as a solid black line with dots
    ax.plot(M_VALUES, baseline_disp, 'o-', color='black', linewidth=2.0, 
            markersize=8, label='Baseline (CV=0.0)', zorder=5)
    
    # Plot all noisy trials for this specific CV level
    for i, disp in enumerate(curves):
        label = f'Noisy Trials (CV={CV})' if i == 0 else ""
        ax.plot(M_VALUES, disp, 'o-', color=color, linewidth=1.2,
                markersize=6, alpha=0.4, zorder=3, label=label)
    
    # Reference: Turing threshold line
    ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Set grid ticks strictly to integer m steps so they stay perfectly spaced
    ax.set_xticks(M_VALUES)
    
    # Create combined text strings and rotate them diagonally
    tick_labels = [f'm={m}\n($k_m$={k:.2f})' for m, k in zip(M_VALUES, K_DISCRETE)]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
    ax.set_xlabel('Ring Modes ($m$)', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

# 2. FIX: Target index 0 of the array to apply the shared left y-label
axes[0].set_ylabel('Max Re(λ)', fontsize=11)

fig.suptitle(
    f'Discrete Ring Dispersion Under Parameter Noise (Config {CONFIG_ID}, N={N_RING} cells)\n'
    f'Black Line tracks Uniform Baseline (CV=0.0)',
    fontsize=13, y=1.04
)

plt.tight_layout()
plt.savefig('dispersion_homogeneous_ring_3954_noise_comparison.png', dpi=200, bbox_inches='tight')
print("\nSaved: dispersion_homogeneous_ring_3954_noise_comparison.png")
plt.close()

