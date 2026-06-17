import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from heterogenous_ring_1754 import compute_jacobian, find_steady_state

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV_PATH = '../TopologyRanking/Topology1754/1754_NEWTURINGCLASS_lhs_results_parameters.csv'


### figure 1: code for one plot for on config id in 1754 ###

CONFIG_ID = 45
CV_VALUES = [0, 0.1, 0.2, 0.3, 0.4]
N_TRIALS = 10

panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
k_values = np.arange(0.01, 10.01, 0.01)
np.random.seed(42)

# LOAD BASELINE TYPE-I PARAMETER SET
df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']
row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0] # choose rank 1!

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])

baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
dU, dV, dW = row['dU'], row['dV'], row['dW']

# HELPER: dispersion curve for a given parameter set

# compute max Re(λ) of [J - k²·D] for each k
def compute_dispersion(params, ss, dU, dV, dW, k_values):
    J = compute_jacobian(ss, params)
    D = np.diag([dU, dV, dW])
    max_reals = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        M = J - k**2 * D
        max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
    return max_reals

# COMPUTE MULTIPLE DISPERSION CURVES PER CV LEVEL

# Store curves per CV: dict of CV -> list of curves
dispersion_curves = {CV: [] for CV in CV_VALUES}

for CV in CV_VALUES:
    if CV == 0.0:
        disp = compute_dispersion(baseline_params, baseline_ss, dU, dV, dW, k_values)
        dispersion_curves[CV].append(disp)
    else:
        # Generate N_TRIALS noisy realisations
        sigma = np.sqrt(np.log(1 + CV**2))
        mu = -sigma**2 / 2
        
        successful = 0
        attempts = 0
        max_attempts = N_TRIALS * 5  # safety: try a bit more than N_TRIALS in case of failures
        
        while successful < N_TRIALS and attempts < max_attempts:
            attempts += 1
            noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
            params_noisy = baseline_params * noise_factors
            
            ss_noisy = find_steady_state(params_noisy)
            if ss_noisy is None:
                continue
            
            disp = compute_dispersion(params_noisy, ss_noisy, dU, dV, dW, k_values)
            dispersion_curves[CV].append(disp)
            successful += 1
        
        print(f"  CV={CV}: {successful}/{N_TRIALS} successful trials "
              f"({attempts} attempts, {attempts - successful} discarded)")

# Find the peak k of the baseline (CV=0) curve
baseline_curve = dispersion_curves[0.0][0]
peak_idx = np.argmax(baseline_curve)
peak_k_baseline = k_values[peak_idx]

# PLOT 4 SUBPLOTS, ONE PER CV
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

# Slice CV_VALUES from index 1 onward to skip 0.0 in the loop setup
for ax, CV, color in zip(axes, CV_VALUES[1:], panel_colors):
    curves = dispersion_curves[CV]
    
    # 1. Plot the noisy trial curves for this CV level
    for disp in curves:
        ax.plot(k_values, disp, color=color, linewidth=1.5, alpha=0.7)
    
    ax.plot(k_values, baseline_curve, color='black', linewidth=2, linestyle='-', alpha=0.8) # OVERLAY THE BASELINE (CV=0) on every subplot for direct comparison!
        
    # Reference: Turing threshold lines
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(peak_k_baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Panel title and labels
    ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12, pad=10)
    ax.set_xlabel('Wavenumber k', fontsize=11)
    ax.tick_params(axis='both', labelsize=9.5)
    ax.grid(alpha=0.3)

# Y-axis label only on leftmost subplot
axes[0].set_ylabel('Max Re(λ)', fontsize=11)

fig.suptitle(
    f'Dispersion Relations Under Parameter Heterogeneity\n'
    f'Config {CONFIG_ID} (Turing Type-I), Topology #1754 with {N_TRIALS} noise realisations per CV',
    fontsize=13, y=0.97, fontweight="semibold"
)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, linestyle='-', label='Baseline (CV=0)'),
    mlines.Line2D([], [], color='red', linewidth=1.2, linestyle='--', label='Turing Threshold'),
    mlines.Line2D([], [], color='black', linewidth=1.2, linestyle='--', label='Baseline Peak k')
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    fontsize=10
)

fig.subplots_adjust(
    left=0.06, 
    right=0.96, 
    top=0.78,     # Gives the header plenty of breathing room at the top
    bottom=0.15,  # Lifts the bottom edge to make room for the unified legend
    wspace=0.10   # Increases horizontal spacing so subplots look wider and less narrow
)

plt.savefig('dispersion_noise_subplots_1754_config45.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: dispersion_noise_subplots_1754_config45.png")
plt.close()






# ==============================================================================
# figure 2 two row comparison  
# ==============================================================================

CONFIG_IDS = [45, 4]
fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey='row')
np.random.seed(42)

for row_idx, config_id in enumerate(CONFIG_IDS):
    row = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]

    # FIX: Keeping this active loops baseline unpacking so Row 2 correctly switches data sets
    baseline_params_multi = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    baseline_ss_multi = np.array([row['u_star'], row['v_star'], row['w_star']])
    dU, dV, dW = row['dU'], row['dV'], row['dW']
    
    dispersion_curves_multi = {CV: [] for CV in CV_VALUES}
    
    # Calculate row baseline
    disp_base_multi = compute_dispersion(baseline_params_multi, baseline_ss_multi, dU, dV, dW, k_values)
    dispersion_curves_multi[0.0].append(disp_base_multi)
    peak_k_baseline_multi = k_values[np.argmax(disp_base_multi)]
    
    # Generate noisy realisations per row configuration
    for CV in CV_VALUES[1:]:
        sigma = np.sqrt(np.log(1 + CV**2))
        mu = -sigma**2 / 2
        successful = 0
        attempts = 0
        max_attempts = N_TRIALS * 5
        
        while successful < N_TRIALS and attempts < max_attempts:
            attempts += 1
            noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params_multi))
            params_noisy = baseline_params_multi * noise_factors
            
            ss_noisy = find_steady_state(params_noisy)
            if ss_noisy is None:
                continue
            
            disp = compute_dispersion(params_noisy, ss_noisy, dU, dV, dW, k_values)
            dispersion_curves_multi[CV].append(disp)
            successful += 1
            
    baseline_curve_multi = dispersion_curves_multi[0.0][0]

    # Render layout elements for the current row matrix
    for col_idx, (CV, color) in enumerate(zip(CV_VALUES[1:], panel_colors)):
        ax = axes_multi[row_idx, col_idx]
        
        for disp in dispersion_curves_multi[CV]:
            ax.plot(k_values, disp, color=color, linewidth=1.2, alpha=0.35)
        
        ax.plot(k_values, baseline_curve_multi, color='black', linewidth=2.0, linestyle='-', alpha=0.8)
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(peak_k_baseline_multi, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        if row_idx == 0:
            ax.set_title(f'CV = {CV:.2f} ({len(dispersion_curves_multi[CV])} trials)', fontsize=11, pad=10)
        if row_idx == 1:
            ax.set_xlabel('Wavenumber k', fontsize=10.5, color='#333333')
            
        ax.tick_params(axis='both', labelsize=9.5)
        ax.grid(alpha=0.3, linestyle=':')
        
    label_suffix = " (Robust)" if config_id == 45 else " (Fragile)"
    axes_multi[row_idx, 0].set_ylabel(f'Config {config_id}{label_suffix}\nMax Re(λ)', fontsize=11, color='#333333')

# Visual adjustment setups for the 2-row layout
title_multi = fig_multi.suptitle(
    f'Dispersion Relations Under Parameter Heterogeneity\nTopology #1754 with {N_TRIALS} Noise Realisations per CV',
    fontsize=12, y=0.96, fontweight='semibold'
)

legend_handles_multi = [
    mlines.Line2D([], [], color='black', linewidth=2, linestyle='-', label='Baseline (CV=0)'),
    mlines.Line2D([], [], color='red', linewidth=1.2, linestyle=':', label='Turing Threshold'),
    mlines.Line2D([], [], color='black', linewidth=1.1, linestyle='--', label='Baseline Peak k')
]
leg_multi = fig_multi.legend(handles=legend_handles_multi, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=10)

fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.12, wspace=0.12, hspace=0.08)

plt.savefig('dispersion_noise_2row_comparison.png', dpi=200, bbox_inches='tight', bbox_extra_artists=[title_multi, leg_multi])
plt.close(fig_multi)






# CODE FOR HAVING IT ALL IN ONE PLOT!

# # ============================================================================
# # COMPUTE A DISPERSION CURVE FOR EACH CV LEVEL
# # ============================================================================

# k_values = np.arange(0.01, 10.01, 0.01)
# np.random.seed(42)

# dispersion_curves = {}

# for CV in CV_VALUES:
#     if CV == 0.0:
#         # No noise: baseline parameters
#         params_noisy = baseline_params.copy()
#     else:
#         # Lognormal noise: σ² = ln(1+CV²), μ = -σ²/2 (so mean factor = 1)
#         sigma = np.sqrt(np.log(1 + CV**2))
#         mu = -sigma**2 / 2
#         noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
#         params_noisy = baseline_params * noise_factors
    
#     # Use baseline steady state — approximation, OK for small perturbations
#     disp = compute_dispersion(params_noisy, baseline_ss, dU, dV, dW, k_values)
#     dispersion_curves[CV] = disp

# for CV in CV_VALUES:
#     if CV == 0.0:
#         params_noisy = baseline_params.copy()
#         ss_for_disp = baseline_ss
    
#     else:
#         sigma = np.sqrt(np.log(1 + CV**2))
#         mu = -sigma**2 / 2
#         noise_factors = np.random.lognormal(mu, sigma, size=len(baseline_params))
#         params_noisy = baseline_params * noise_factors
        
#         # Find the actual steady state for the noisy parameters
#         ss_for_disp = find_steady_state(params_noisy)
#         if ss_for_disp is None:
#             print(f"  CV={CV}: no steady state found, skipping")
#             continue
    
#     disp = compute_dispersion(params_noisy, ss_for_disp, dU, dV, dW, k_values)
#     dispersion_curves[CV] = disp

# # ============================================================================
# # PLOT
# # ============================================================================

# fig, ax = plt.subplots(figsize=(10, 6))

# # Distinct colours for each CV
# colors = ['black', 'steelblue', 'hotpink', 'orange', 'purple']
# # linestyles = ['-', '--', '--', '--', '--']

# # for (CV, color) in zip(CV_VALUES, colors): # , ls , linestyles 
# #     label = f'CV = {CV:.2f}' if CV > 0 else f'CV = {CV:.2f} (baseline)'
# #     ax.plot(k_values, dispersion_curves[CV],
# #             color=color, linewidth=2.5, label=label) # linestyle=ls,

# for (CV, color) in zip(CV_VALUES, colors): # , ls , linestyles 
#     label = f'CV = {CV:.2f}' if CV > 0 else f'CV = {CV:.2f} (baseline)'
#     lw = 2 if CV == 0 else 1.5
#     ax.plot(k_values, dispersion_curves[CV],
#             color=color, linewidth=lw, label=label) # linestyle=ls,


# # Reference lines
# ax.axhline(0, color='red', linestyle=':', alpha=0.6, linewidth=1.5,
#            label='Turing threshold (Re(λ)=0)')

# ax.set_xlabel('Wavenumber k', fontsize=13)
# ax.set_ylabel('Max Re(λ)', fontsize=13)
# ax.set_title(f'Dispersion Relation Under Parameter Heterogeneity\n'
#              f'Config {CONFIG_ID} (Type-I), Topology #1754',
#              fontsize=13, pad=12)
# ax.legend(fontsize=10, loc='best', framealpha=0.95)
# ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig('dispersion_noise_1754_config45.png', dpi=200, bbox_inches='tight')
# print(f"\nSaved: dispersion_noise_1754_config45.png")
# plt.close()