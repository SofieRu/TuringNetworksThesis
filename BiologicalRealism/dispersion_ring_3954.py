import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

















# CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
# CONFIG_IDS = [45, 4]                    # The two configurations to compare
# N_RING = 30                             # number of cells in ring
# N_TRIALS = 10                            # noisy realisations per CV
# CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]    # Noise levels
# SEED = 42

# # Discrete wavenumbers allowed for ring of N cells (periodic boundary)
# M_VALUES = np.arange(0, N_RING // 2 + 1)
# K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
# x_indices = np.arange(len(K_DISCRETE))

# print(f"Discrete k_m values for N={N_RING}: {K_DISCRETE}")

# # Load the main dataframe once
# df = pd.read_csv(CSV_PATH)
# type_i = df[df['classification'] == 'Type-I']

# # ============================================================================
# # HELPER: dispersion at discrete k_m values
# # ============================================================================

# def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
#     J = compute_jacobian(ss, params)
#     D = np.diag([dU, dV, dW])
#     max_reals = np.zeros(len(k_discrete))
#     for i, k in enumerate(k_discrete):
#         M = J - k**2 * D
#         max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
#     return max_reals

# # ============================================================================
# # INITIALISE PLOT: 2 Rows (one per Config), 4 Columns (one per CV)
# # ============================================================================

# noisy_cvs = [0.1, 0.2, 0.3, 0.4]
# panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

# fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey='row') # sharey='row' links y-axis per config

# # ============================================================================
# # MAIN LOOP OVER CONFIGURATIONS
# # ============================================================================

# for row_idx, config_id in enumerate(CONFIG_IDS):
#     print(f"\nProcessing Config ID: {config_id} (Row {row_idx + 1}/2)")
    
#     # --- LOAD CONFIG-SPECIFIC PARAMETER SET ---
#     row_data = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]

#     baseline_params = np.array([
#         row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
#         row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
#         row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
#     ])
#     baseline_ss = np.array([row_data['u_star'], row_data['v_star'], row_data['w_star']])
#     dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']

#     # --- COMPUTE DISPERSION FOR EACH CV ---
#     np.random.seed(SEED) # Reset seed for consistent noise generation across configs if desired
#     dispersion_results = {CV: [] for CV in CV_VALUES}

#     for CV in CV_VALUES:
#         if CV == 0.0:
#             disp = compute_discrete_dispersion(baseline_params, baseline_ss, dU, dV, dW, K_DISCRETE)
#             dispersion_results[CV].append(disp)
#         else:
#             sigma = np.sqrt(np.log(1 + CV**2))
#             mu = -sigma**2 / 2
            
#             successful = 0
#             attempts = 0
#             max_attempts = N_TRIALS * 5
            
#             while successful < N_TRIALS and attempts < max_attempts:
#                 attempts += 1
#                 noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
#                 params_noisy = baseline_params * noise_factors
                
#                 ss_noisy = find_steady_state(params_noisy)
#                 if ss_noisy is None:
#                     continue
                
#                 disp = compute_discrete_dispersion(params_noisy, ss_noisy, dU, dV, dW, K_DISCRETE)
#                 dispersion_results[CV].append(disp)
#                 successful += 1
            
#             print(f"  CV={CV}: {successful}/{N_TRIALS} successful trials")

#     # --- PLOT ROW FOR THIS CONFIG ---
#     baseline_disp = dispersion_results[0.0][0]
#     row_axes = axes[row_idx] # Grab the 4 subplots corresponding to this row

#     for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
#         curves = dispersion_results[CV]
        
#         # Plot baseline (CV = 0.0)
#         ax.plot(x_indices, baseline_disp, 'o-', color='black', linewidth=2.0, 
#                 markersize=8, label='Baseline (CV=0.0)', zorder=5)
        
#         # Plot all noisy trials
#         for i, disp in enumerate(curves):
#             label = f'Noisy Trials (CV={CV})' if i == 0 else ""
#             ax.plot(x_indices, disp, 'o-', color=color, linewidth=1.2,
#                     markersize=6, alpha=0.4, zorder=3, label=label)
        
#         # Reference: Turing threshold line
#         ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
#         ax.set_xticks(x_indices)
        
#         # Formatting X axes (only label the bottom row properly to save space, or label both)
#         tick_labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
#         ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
        
#         # Subplot Titles and Grid
#         ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=11)
#         if row_idx == 1: # Only put x-labels on bottom row for a clean layout
#             ax.set_xlabel('Discrete Wavenumbers ($k_m$)', fontsize=11)
            
#         ax.grid(alpha=0.3, linestyle='--')
#         ax.legend(loc='upper right', fontsize=8)

#     # Set the outer y-label only for the first subplot of this specific row
#     row_axes[0].set_ylabel(f'Config {config_id}\nMax Re(λ)', fontsize=12, fontweight='bold')

# # ============================================================================
# # SAVE AND FINISH
# # ============================================================================

# fig.suptitle(
#     f'Discrete Ring Dispersion Comparison (N={N_RING} cells)\n'
#     f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]} | Black Line = Uniform Baseline (CV=0.0)',
#     fontsize=14, y=0.98
# )

# #plt.tight_layout()
# plt.tight_layout(w_pad=0.5)
# fig.subplots_adjust(top=0.85)
# plt.savefig('dispersion_ring_comp_3954.png', dpi=200, bbox_inches='tight')
# print("\nSaved: dispersion_ring_comp_3954.png")
# plt.close()






CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
CONFIG_IDS = [45, 4]
N_RING = 30
N_TRIALS = 10
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEED = 42

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
x_indices = np.arange(len(K_DISCRETE))

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']

def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
    J = compute_jacobian(ss, params)
    D = np.diag([dU, dV, dW])
    max_reals = np.zeros(len(k_discrete))
    for i, k in enumerate(k_discrete):
        M = J - k**2 * D
        max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
    return max_reals


noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey='row')

for row_idx, config_id in enumerate(CONFIG_IDS):
    
    # Extract row matching loop context safely
    row_data = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]
    
    baseline_params = np.array([
        row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
        row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
        row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
    ])
    baseline_ss = np.array([row_data['u_star'], row_data['v_star'], row_data['w_star']])
    dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
    
    np.random.seed(SEED)
    dispersion_results = {CV: [] for CV in CV_VALUES}
    
    for CV in CV_VALUES:
        if CV == 0.0:
            disp = compute_discrete_dispersion(baseline_params, baseline_ss, dU, dV, dW, K_DISCRETE)
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
                disp = compute_discrete_dispersion(params_noisy, ss_noisy, dU, dV, dW, K_DISCRETE)
                dispersion_results[CV].append(disp)
                successful += 1

    baseline_disp = dispersion_results[0.0][0]
    row_axes = axes_multi[row_idx]
    
    for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
        curves = dispersion_results[CV]
        
        # Plot Baseline
        ax.plot(x_indices, baseline_disp, 'o-', color='black', linewidth=2.0, 
                markersize=8, label='Baseline (CV=0.0)' if (row_idx==0 and col_idx==0) else "", zorder=5)
        
        # Plot Noisy Trials
        for i, disp in enumerate(curves):
            label = f'Noisy (CV={CV})' if (i == 0 and row_idx == 0 and col_idx == 0) else ""
            ax.plot(x_indices, disp, 'o-', color=color, linewidth=1.2,
                    markersize=6, alpha=0.4, zorder=3, label=label)
            
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xticks(x_indices)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Clean Clean-up: Only show titles over the top row of subplots
        if row_idx == 0:
            ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
            
        # Clean Clean-up: Only show X axis text on the very bottom row panels
        if row_idx == 1:
            tick_labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
            ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=10)
            ax.set_xlabel('Discrete Wavenumbers ($k_m$)', fontsize=11)

    # Put a clear bold row marker on the left-most panel of each row
    row_axes[0].set_ylabel(f'Config {config_id}\nMax Re(λ)', fontsize=12)

# ============================================================================
# YOUR PROPOSED SPACING AND FINISHING
# ============================================================================
fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.08, hspace=0.08)

fig_multi.suptitle(
    f'Discrete Ring Dispersion Comparison (N={N_RING} cells)\n'
    f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]} | Black Line = Uniform Baseline (CV=0.0)',
    fontsize=14, y=0.95
)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
]

fig_multi.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.savefig('dispersion_comparison_configs.png', dpi=200, bbox_inches='tight')
print("\nSaved: dispersion_comparison_configs.png")
plt.close()
