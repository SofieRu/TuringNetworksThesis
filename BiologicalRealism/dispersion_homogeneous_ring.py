import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heterogenous_ring_3954 import compute_jacobian, find_steady_state

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

# ============================================================================
# CONFIG
# ============================================================================

# CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
# CONFIG_ID = 45
# N_RING = 30                             # number of cells in ring
# N_TRIALS = 10                            # noisy realisations per CV
# CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]    # Added 0.4 noise level
# SEED = 42

# # Discrete wavenumbers allowed for ring of N cells (periodic boundary)
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

# # ============================================================================
# # HELPER: dispersion at discrete k_m values
# # ============================================================================

# #Return array of max Re(λ) at each discrete k_m
# def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
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
#         disp = compute_discrete_dispersion(baseline_params, baseline_ss,
#                                             dU, dV, dW, K_DISCRETE)
#         dispersion_results[CV].append(disp)
#     else:
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
# # PLOT — 4 SUBPLOTS (CV 0.1 to 0.4), BASELINE (CV 0.0) IN BLACK EVERYWHERE
# # ============================================================================

# noisy_cvs = [0.1, 0.2, 0.3, 0.4]
# panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

# # Adjusted figure size slightly to accommodate the tilted labels safely
# fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

# # Extract the baseline array correctly as a 1D structure
# baseline_disp = dispersion_results[0.0][0]

# # Generate index array [0, 1, 2, ..., 15] to ensure perfectly even spacing
# x_indices = np.arange(len(K_DISCRETE))

# for ax, CV, color in zip(axes, noisy_cvs, panel_colors):
#     curves = dispersion_results[CV]
    
#     # Plot baseline (CV = 0.0) as a solid black line with dots
#     ax.plot(x_indices, baseline_disp, 'o-', color='black', linewidth=2.0, 
#             markersize=8, label='Baseline (CV=0.0)', zorder=5)
    
#     # Plot all noisy trials for this specific CV level
#     for i, disp in enumerate(curves):
#         label = f'Noisy Trials (CV={CV})' if i == 0 else ""
#         ax.plot(x_indices, disp, 'o-', color=color, linewidth=1.2,
#                 markersize=6, alpha=0.4, zorder=3, label=label)
    
#     # Reference: Turing threshold line
#     ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
#     # Lock grid lines and tick marks to the even indexes
#     ax.set_xticks(x_indices)
    
#     # Clean single-line formatting: looks like k_0=0.00, k_5=1.00, etc.
#     tick_labels = [f'k_{{{m}}}={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
#     ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
    
#     ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
#     ax.set_xlabel('Discrete Wavenumbers ($k_m$)', fontsize=11)
#     ax.grid(alpha=0.3, linestyle='--')
#     ax.legend(loc='upper right', fontsize=9)

# # Target the first axis panel specifically to set the outer shared y-label
# axes[0].set_ylabel('Max Re(λ)', fontsize=11)

# fig.suptitle(
#     f'Discrete Ring Dispersion Under Parameter Noise (Config {CONFIG_ID}, N={N_RING} cells)\n'
#     f'Black Line tracks Uniform Baseline (CV=0.0)',
#     fontsize=13, y=1.04
# )

# plt.tight_layout()
# plt.savefig('dispersion_homogeneous_ring_3954_noise_comparison.png', dpi=200, bbox_inches='tight')
# print("\nSaved: dispersion_homogeneous_ring_3954_noise_comparison.png")
# plt.close()










###### code doesnt run but its for having 3954 and 1754 in one plot just the robust one though!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the topology-specific functions with aliases to avoid namespace collisions
from heterogenous_ring_3954 import compute_jacobian as jac_3954, find_steady_state as ss_3954
# Replace 'heterogenous_ring_1754' with the exact name of your local module
from heterogenous_ring_1754 import compute_jacobian as jac_1754, find_steady_state as ss_1754

# ============================================================================
# RUN CONFIGURATION (SHARED)
# ============================================================================
N_RING = 30                             # number of cells in ring
N_TRIALS = 10                            # noisy realisations per CV
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]    # Noise levels
SEED = 42

# Discrete wavenumbers allowed for ring of N cells (periodic boundary)
M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
x_indices = np.arange(len(K_DISCRETE))

# ============================================================================
# MULTI-TOPOLOGY METADATA PREPARATION
# ============================================================================
topologies_meta = [
    {
        "id": "3954",
        "csv_path": '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv',
        "config_id": 45,
        "jacobian_func": jac_3954,
        "ss_func": ss_3954
    },
    {
        "id": "1754",
        "csv_path": '../TopologyRanking/Topology1754/1754_NEWTURINGCLASS_lhs_results_parameters.csv', # Verify this path matches yours
        "config_id": 35, # Adjust if Topology 1754 uses a different config ID
        "jacobian_func": jac_1754,
        "ss_func": ss_1754
    }
]
  
# ============================================================================
# HELPER: dispersion at discrete k_m values
# ============================================================================
def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete, jacobian_func):
    J = jacobian_func(ss, params)
    D = np.diag([dU, dV, dW])
    max_reals = np.zeros(len(k_discrete))
    for i, k in enumerate(k_discrete):
        M = J - k**2 * D
        max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
    return max_reals

# ============================================================================
# SETUP PLOT — 2 ROWS (Topologies), 4 COLUMNS (CV Levels)
# ============================================================================
noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

# nrows=2 creates an explicit row for each topology
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 11), sharey='row')

np.random.seed(SEED)

# ============================================================================
# ITERATE OVER TOPOLOGIES (ROWS)
# ============================================================================
for row_idx, topo in enumerate(topologies_meta):
    print(f"\nProcessing Topology {topo['id']}...")
    
    # Load parameters specific to this topology
    df = pd.read_csv(topo['csv_path'])
    type_i = df[df['classification'] == 'Type-I']
    row_data = type_i[(type_i['config_id'] == topo['config_id']) & (type_i['param_rank'] == 1)].iloc[0]

    baseline_params = np.array([
        row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
        row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
        row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
    ])
    baseline_ss = np.array([row_data['u_star'], row_data['v_star'], row_data['w_star']])
    dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']

    # Compute dispersion for each CV level
    dispersion_results = {CV: [] for CV in CV_VALUES}

    for CV in CV_VALUES:
        if CV == 0.0:
            disp = compute_discrete_dispersion(baseline_params, baseline_ss,
                                                dU, dV, dW, K_DISCRETE, topo['jacobian_func'])
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
                
                ss_noisy = topo['ss_func'](params_noisy)
                if ss_noisy is None:
                    continue
                
                disp = compute_discrete_dispersion(params_noisy, ss_noisy,
                                                    dU, dV, dW, K_DISCRETE, topo['jacobian_func'])
                dispersion_results[CV].append(disp)
                successful += 1
            
            print(f"  CV={CV}: {successful}/{N_TRIALS} successful trials")

    # ============================================================================
    # PLOT COLUMNS FOR THIS SPECIFIC TOPOLOGY ROW
    # ============================================================================
    baseline_disp = dispersion_results[0.0][0]
    
    for col_idx, (CV, color) in enumerate(zip(noisy_cvs, panel_colors)):
        ax = axes[row_idx, col_idx]  # Index into the 2D grid matrix
        curves = dispersion_results[CV]
        
        # Plot baseline (CV = 0.0)
        ax.plot(x_indices, baseline_disp, 'o-', color='black', linewidth=2.0, 
                markersize=8, label='Baseline (CV=0.0)', zorder=5)
        
        # Plot noisy realizations
        for i, disp in enumerate(curves):
            label = f'Noisy Trials (CV={CV})' if i == 0 else ""
            ax.plot(x_indices, disp, 'o-', color=color, linewidth=1.2,
                    markersize=6, alpha=0.4, zorder=3, label=label)
        
        # Reference line
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Handle labels and structural grid elements
        ax.set_xticks(x_indices)
        tick_labels = [f'k_{{{m}}}={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
        ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
        
        # Row-specific title enhancements
        ax.set_title(f"Topo {topo['id']} | CV = {CV:.2f} ({len(curves)} trials)", fontsize=11)
        ax.set_xlabel('Discrete Wavenumbers ($k_m$)', fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
    
    # Label the leftmost subplots with the specific Topology ID and Max Re(λ) label
    axes[row_idx, 0].set_ylabel(f"Topology {topo['id']}\nMax Re(λ)", fontsize=12, fontweight='bold')

# Global Layout Polishing
fig.suptitle(
    f'Discrete Ring Dispersion Comparison: Topology 3954 vs Topology 1754 (N={N_RING} cells)\n'
    f'Black Line tracks Uniform Baseline (CV=0.0)',
    fontsize=14, y=1.02
)

plt.tight_layout()
plt.savefig('dispersion_topology_comparison_noise.png', dpi=200, bbox_inches='tight')
print("\nSaved: dispersion_topology_comparison_noise.png")
plt.close()
