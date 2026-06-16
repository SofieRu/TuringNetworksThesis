import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from heterogenous_ring_3954 import compute_jacobian, find_steady_state

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
CONFIG_ID = 45
CV_VALUES = [0, 0.1, 0.2, 0.3, 0.4]  

# LOAD BASELINE TYPE-I PARAMETER SET

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']
row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0] # choose rank 1!

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])
baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
dU, dV, dW = row['dU'], row['dV'], row['dW']

print(f"Loaded config {CONFIG_ID}: {row['config_name']}")
print(f"  Diffusion (dU, dV, dW) = ({dU}, {dV}, {dW})")
print(f"  Steady state: ({baseline_ss[0]:.4f}, {baseline_ss[1]:.4f}, {baseline_ss[2]:.4f})")

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

k_values = np.arange(0.01, 10.01, 0.01)
N_TRIALS = 10
np.random.seed(42)

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

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

# 4 distinct colors for the 4 panels
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

# Slice CV_VALUES from index 1 onward to skip 0.0 in the loop setup
for ax, CV, color in zip(axes, CV_VALUES[1:], panel_colors):
    curves = dispersion_curves[CV]
    
    # 1. Plot the noisy trial curves for this CV level
    for disp in curves:
        ax.plot(k_values, disp, color=color, linewidth=1.5, alpha=0.7)
    
    # 2. OVERLAY THE BASELINE (CV=0) on every subplot for direct comparison!
    ax.plot(k_values, baseline_curve, color='black', linewidth=2, linestyle='-', alpha=0.8, label='Baseline (CV=0)')
        
    # Reference: Turing threshold lines
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(peak_k_baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Panel title and labels
    ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
    ax.set_xlabel('Wavenumber k', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc='upper right', fontsize=9)

# Y-axis label only on leftmost subplot
axes[0].set_ylabel('Max Re(λ)', fontsize=11)
#axes[0].legend(frameon=False, loc='upper right') # Shows the baseline marker legend

# fig.suptitle(
#     f'Dispersion Relations Under Parameter Heterogeneity\n'
#     f'Config {CONFIG_ID} (Turing Type-I), Topology #3954 with {N_TRIALS} noise realisations per CV',
#     fontsize=13, y=1.02, fontweight="semibold"
# )

# plt.tight_layout()
# plt.savefig('dispersion_noise_subplots_3954_config45.png', dpi=200, bbox_inches='tight')
# print(f"\nSaved: dispersion_noise_subplots_3954_config45.png")
# plt.close()

# Create one unified, clean legend horizontally at the bottom so subplots stay wide!
legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, linestyle='-', label='Baseline (CV=0)'),
    mlines.Line2D([], [], color='red', linewidth=1.2, linestyle=':', label='Turing Threshold'),
    mlines.Line2D([], [], color='black', linewidth=1.1, linestyle='--', label='Baseline Peak k')
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    fontsize=9.5
)

# Tight manual adjustments to give the bottom legend room
fig.subplots_adjust(
    left=0.06, 
    right=0.96, 
    top=0.83, 
    bottom=0.18,  # Raised from 0.12 to give the legend space without squishing the graphs
    wspace=0.18
)

plt.savefig('dispersion_noise_subplots_3954_config45.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: dispersion_noise_subplots_3954_config45.png")
plt.close()



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
#              f'Config {CONFIG_ID} (Type-I), Topology #3954',
#              fontsize=13, pad=12)
# ax.legend(fontsize=10, loc='best', framealpha=0.95)
# ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig('dispersion_noise_3954_config45.png', dpi=200, bbox_inches='tight')
# print(f"\nSaved: dispersion_noise_3954_config45.png")
# plt.close()