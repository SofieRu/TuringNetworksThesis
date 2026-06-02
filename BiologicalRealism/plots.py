#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import fsolve

from homogenous_ring import (
    hopping,
)

# for when we run bifurcation plots!!
# from homogenous_ring import (
#     CONFIG_TO_TEST,
#     build_ring_jacobian_homogeneous,
#     steady_state_expected,
#     baseline_params,
#     hopping,
# )

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

### Load results for #3954 ###
# Config 13 (robust)
with open('3954_cv_sweep_high_config13_N10.pkl', 'rb') as f:
    cv_data_13 = pickle.load(f)

with open('3954_sensitivity_results_config13_N10.pkl', 'rb') as f:
    sens_data_13 = pickle.load(f)

# Config 2 (fragile)
with open('3954_cv_sweep_low_config2_N10.pkl', 'rb') as f:
    cv_data_2 = pickle.load(f)

with open('3954_sensitivity_results_config2_N10.pkl', 'rb') as f:
    sens_data_2 = pickle.load(f)

def extract_cv_arrays(cv_data):
    results = cv_data['results']
    CV_values = np.array([r['CV'] for r in results])
    mean_eigs = np.array([r['mean_eig'] for r in results])
    min_eigs = np.array([r['min_eig'] for r in results])
    max_eigs = np.array([r['max_eig'] for r in results])
    std_eigs = np.array([r['std_eig'] for r in results])
    all_eigenvalues = [r['all_eigenvalues'] for r in results]
    robustness = np.array([r['robustness'] for r in results])
    return {
        'CV': CV_values, 'mean': mean_eigs, 'min': min_eigs, 'max': max_eigs,
        'std': std_eigs, 'all': all_eigenvalues, 'robustness': robustness,
        'config_name': cv_data.get('config_name', 'unknown'),
        'config_id': cv_data.get('config_id', '?'),
        'hopping': cv_data.get('hopping', {})
    }

cv13 = extract_cv_arrays(cv_data_13)
cv2 = extract_cv_arrays(cv_data_2)

# Helper for diffusion string in plot titles
def diff_str(hopping):
    return f"dU={hopping['h_u']}, dV={hopping['h_v']}, dW={hopping['h_w']}"


# ========================================================================================================================================================
#                                                              BIOLOGICAL REALISM PLOTS
# ========================================================================================================================================================

# ============================================================================
# FIGURE 1A: Mean with full range Config 13
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(cv13['CV'], cv13['min'], cv13['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)

# plot mean line
ax.plot(cv13['CV'], cv13['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)

# horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

# labels and title
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Turing Growth Rate vs Parameter Heterogeneity\n'f'Config 13 (robust): {diff_str(cv13["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# save the plots
plt.tight_layout()
plt.savefig('config13_fig1_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved as config13_fig1_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2A: BOXPLOT Config 13
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot

bp = ax.boxplot(cv13['all'],
                positions=range(len(cv13['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

# color boxes
for patch in bp['boxes']:
    patch.set_facecolor('slateblue')
    patch.set_alpha(0.8)

# horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv13['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv13['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)

ax.set_title(f'3954 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'Config 13 (robust): {diff_str(cv13["hopping"])}',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)  # Was fontsize=11
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('config13_fig2_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: config13_fig2_boxplot_cv_sweep.png")
plt.close()


# ============================================================================
# FIGURE 1B: Mean with full range — Config 2
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(cv2['CV'], cv2['min'], cv2['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)
ax.plot(cv2['CV'], cv2['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Turing Growth Rate vs Parameter Heterogeneity\n' f'Config 2 (fragile): {diff_str(cv2["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('config2_fig1_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved: config2_fig1_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2B: Boxplot — Config 2
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(cv2['all'],
                positions=range(len(cv2['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

for patch in bp['boxes']:
    patch.set_facecolor('slateblue')
    patch.set_alpha(1.0)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv2['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv2['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'Config 2 (fragile): {diff_str(cv2["hopping"])} — 1000 trials per CV',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('config2_fig2_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: config2_fig2_boxplot_cv_sweep.png")
plt.close()




# ========================================================================================================================================================
#                                                              SENSITIVITY ANALYSIS PLOTS
# ========================================================================================================================================================

# ============================================================================
# FIGURE 3: SENSITIVITY ANALYSIS
# ============================================================================

PARAM_LABELS = {
    'alpha_u': 'u basal production',
    'beta_u': 'u regulated production',
    'K_uu': 'u self-activation (K)',
    'K_vu': 'v-u inhibition (K)',
    'delta_u': 'u degradation',
    'alpha_v': 'v basal production',
    'beta_v': 'v regulated production',
    'K_uv': 'u-v activation (K)',
    'K_wv': 'w-v inhibition (K)',
    'delta_v': 'v degradation',
    'alpha_w': 'w basal production',
    'beta_w': 'w regulated production',
    'K_ww': 'w self-activation (K)',
    'K_uw': 'u-w inhibition (K)',
    'K_vw': 'v-w inhibition (K)',
    'delta_w': 'w degradation'
    }


def plot_sensitivity(sens_data, config_label, save_name):
    param_names = sens_data['param_names']
    sensitivities = np.array(sens_data['sensitivities'])
    sens_plus = np.array(sens_data['sensitivities_plus'])
    sens_minus = np.array(sens_data['sensitivities_minus'])

    # Separate clean values from NaN (bifurcation-crossing)
    nan_mask = np.isnan(sensitivities)
    # Sort: clean values descending, then NaN entries at the bottom
    clean_idx = np.where(~nan_mask)[0]
    nan_idx = np.where(nan_mask)[0]
    clean_idx_sorted = clean_idx[np.argsort(sensitivities[clean_idx])[::-1]]
    sorted_indices = np.concatenate([clean_idx_sorted, nan_idx])

    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_labels = [PARAM_LABELS[name] for name in sorted_names]
    sorted_sens = sensitivities[sorted_indices]
    sorted_is_nan = np.isnan(sorted_sens)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars: use a tiny placeholder height for NaN entries so the bar is visible
    bar_heights = np.where(sorted_is_nan, 1e-4, sorted_sens)
    bars = ax.bar(range(len(sorted_labels)), bar_heights, color='steelblue', alpha=0.9)

    # Color top 3 clean bars (stiff) — only count non-NaN
    n_clean = (~sorted_is_nan).sum()
    for i in range(min(3, n_clean)):
        bars[i].set_color('mediumvioletred')

    # Color bottom 3 clean bars (sloppy) — last 3 non-NaN
    sloppy_start = max(0, n_clean - 3)
    for i in range(sloppy_start, n_clean):
        bars[i].set_color('silver')

    # Mark NaN bars with hatching, label "bifurcation"
    for i in range(n_clean, len(sorted_labels)):
        bars[i].set_color('lightgray')
        bars[i].set_hatch('///')
        bars[i].set_alpha(0.6)
        # Add explicit "bifurcation" annotation above the bar
        #ax.text(i, 2e-4, 'bifurcation\ncrossed', ha='center', va='bottom', fontsize=8, color='darkred', rotation=0, fontweight='bold')
        
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Change in Turing Growth Rate per 10% Parameter Change (log scale)', fontsize=12)
    ax.set_title(f'3954 Parameter Sensitivity Analysis: {config_label}\n'
                 f'Which Parameters Control Turing Patterns?',
                 fontsize=13, pad=15)

    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-4)
    ax.grid(True, alpha=0.3, axis='y', which='major')

    # Legend
    legend_elements = [
        Patch(facecolor='mediumvioletred', alpha=0.9, label='Stiff (critical)'),
        Patch(facecolor='steelblue', alpha=0.9, label='Moderate'),
        Patch(facecolor='silver', alpha=0.9, label='Sloppy (tolerant)'),
        Patch(facecolor='lightgray', alpha=0.6, hatch='///',
              label='Bifurcation crossing (no local sensitivity)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


# ============================================================================
# FIGURE 3A + 3B : Sensitivity Config 13 and Config 2
# ============================================================================
plot_sensitivity(sens_data_13, 'Config 13 (robust)', 'config13_fig3_sensitivity.png')

plot_sensitivity(sens_data_2, 'Config 2 (fragile)', 'config2_fig3_sensitivity.png')







# ========================================================================================================================================================
#                                                  FIGURE X: ROBUSTNESS VS CV — N=10 vs N=20 COMPARISON
# ========================================================================================================================================================

# Load all four CV sweep pickles
# (Adjust filenames to match what you actually have)
with open('3954_cv_sweep_high_config13_N10.pkl', 'rb') as f:
    cv13_N10_data = pickle.load(f)
with open('3954_cv_sweep_high_config13_N20.pkl', 'rb') as f:
    cv13_N20_data = pickle.load(f)
with open('3954_cv_sweep_low_config2_N10.pkl', 'rb') as f:
    cv2_N10_data = pickle.load(f)
with open('3954_cv_sweep_low_config2_N20.pkl', 'rb') as f:
    cv2_N20_data = pickle.load(f)

# Extract robustness arrays using your existing helper
cv13_N10 = extract_cv_arrays(cv13_N10_data)
cv13_N20 = extract_cv_arrays(cv13_N20_data)
cv2_N10  = extract_cv_arrays(cv2_N10_data)
cv2_N20  = extract_cv_arrays(cv2_N20_data)

# Build the 4-curve list
robustness_curves = [
    {
        'label': '#3954 robust (config 13, N=10)',
        'CV': cv13_N10['CV'], 'robustness': cv13_N10['robustness'],
        'color': 'darkblue', 'marker': 'o', 'linestyle': '-',
    },
    {
        'label': '#3954 robust (config 13, N=20)',
        'CV': cv13_N20['CV'], 'robustness': cv13_N20['robustness'],
        'color': 'darkblue', 'marker': 'o', 'linestyle': '--',
    },
    {
        'label': '#3954 fragile (config 2, N=10)',
        'CV': cv2_N10['CV'], 'robustness': cv2_N10['robustness'],
        'color': 'crimson', 'marker': 's', 'linestyle': '-',
    },
    {
        'label': '#3954 fragile (config 2, N=20)',
        'CV': cv2_N20['CV'], 'robustness': cv2_N20['robustness'],
        'color': 'crimson', 'marker': 's', 'linestyle': '--',
    },
]

fig, ax = plt.subplots(figsize=(11, 6))

for curve in robustness_curves:
    ax.plot(curve['CV'], curve['robustness'],
            marker=curve['marker'], color=curve['color'],
            linestyle=curve['linestyle'],
            linewidth=2.5, markersize=9,
            label=curve['label'], zorder=3)

ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
           label='50% threshold', zorder=2)
ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, zorder=1)

ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Robustness (% of trials with Turing instability)', fontsize=13)
ax.set_title('Robustness to Heterogeneity across Ring Sizes\n'
             'Same qualitative behaviour at N=10 and N=20',
             fontsize=13, pad=12)

ax.set_xlim(-0.01, 0.42)
ax.set_ylim(-5, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, alpha=0.3)

# Legend split into two columns for readability
ax.legend(fontsize=10, loc='center right', ncol=1, framealpha=0.95)

plt.tight_layout()
plt.savefig('fig5_robustness_vs_cv_N10_vs_N20.png', dpi=300, bbox_inches='tight')
print("Saved: fig5_robustness_vs_cv_N10_vs_N20.png")
plt.close()































################################## BACKUP ##################################

# # Sort by sensitivity
# sorted_indices = np.argsort(sensitivities)[::-1]
# sorted_names = [param_names[i] for i in sorted_indices]
# sorted_labels = [PARAM_LABELS[name] for name in sorted_names]
# sorted_sens = [sensitivities[i] for i in sorted_indices]

# # Create plot
# fig, ax = plt.subplots(figsize=(12, 7))

# # Create bars
# bars = ax.bar(range(len(sorted_labels)), sorted_sens, color='steelblue', alpha=0.9)

# # Color top 3 (stiff) in red
# for i in range(min(3, len(bars))):
#     bars[i].set_color('mediumvioletred')
#     bars[i].set_alpha(0.9)

# # Color bottom 3 (sloppy) in gray
# for i in range(max(0, len(bars)-3), len(bars)):
#     bars[i].set_color('silver')
#     bars[i].set_alpha(0.9)

# # Labels
# ax.set_xticks(range(len(sorted_labels)))
# ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=10)
# ax.set_ylabel('Change in Turing Growth Rate per 10% Parameter Change (log scale)', fontsize=12)
# ax.set_title('3954 Parameter Sensitivity Analysis: Stiff vs Sloppy\nWhich Parameters Control Turing Patterns?', fontsize=13, pad=15)

# # Only 4 clean tick marks
# # ax.set_yticks([0.001, 0.01, 0.1, 1.0])
# # ax.set_yticklabels(['0.001', '0.01', '0.1', '1.0'])
# ax.set_yscale('log')
# ax.grid(True, alpha=0.3, axis='y', which='major')

# # Legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='mediumvioletred', alpha=0.8, label='Stiff (critical)'),
#     Patch(facecolor='steelblue', alpha=0.8, label='Moderate'),
#     Patch(facecolor='silver', alpha=0.8, label='Sloppy (tolerant)')
# ]
# ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# # Annotation
# ratio = sorted_sens[0] / sorted_sens[-1]
# # ax.text(0.02, 0.98, 
# #         f'Stiffest: {sorted_labels[0]}\nS = {sorted_sens[0]:.3f}\n\n'
# #         f'Sloppiest: {sorted_labels[-1]}\nS = {sorted_sens[-1]:.5f}\n\n'
# #         f'Ratio: {ratio:.0f}',
# #         transform=ax.transAxes,
# #         fontsize=10,
# #         verticalalignment='top',
# #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# plt.tight_layout()
# plt.savefig('config4_sensitivity_stiff_vs_sloppy.png', dpi=300, bbox_inches='tight') # CHANGE HERE BASED ON FILE
# print("Saved: config4_sensitivity_stiff_vs_sloppy.png")


# # ANOTHER TYPE OF PLOT

# # Group by node
# u_params = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u']
# v_params = ['alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v']
# w_params = ['alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']

# # Sum sensitivities per node
# u_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in u_params)
# v_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in v_params)
# w_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in w_params)

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.bar(['u (activator)', 'v (inhibitor)', 'w (immobile)'], 
#        [u_total, v_total, w_total], color=['green', 'blue', 'purple'])
# ax.set_ylabel('Total Sensitivity')
# ax.set_title('3954 Which Gene Controls Pattern Formation?')

# ============================================================================
# FIGURE 3: MEAN VS STD PLOT
# ============================================================================

# fig, ax = plt.subplots(figsize=(8, 8))

# # Plot data points
# ax.plot(mean_eigs, std_eigs, 'o-', markersize=10, linewidth=2, 
#         color='darkblue', label='Observed')

# # Reference line: std = mean (Poisson-like)
# max_val = max(mean_eigs.max(), std_eigs.max())
# ax.plot([0, max_val], [0, max_val], '--', color='red', linewidth=2,
#         label='Std = Mean (Poisson)', alpha=0.7)

# # Annotate each point with CV value
# for i, cv in enumerate(CV_values):
#     ax.annotate(f'CV={cv:.2f}', 
#                 xy=(mean_eigs[i], std_eigs[i]),
#                 xytext=(5, 5), textcoords='offset points',
#                 fontsize=9, alpha=0.7)

# # Labels
# ax.set_xlabel('Mean Re(λ)', fontsize=13)
# ax.set_ylabel('Std Re(λ)', fontsize=13)
# ax.set_title('3954 Mean-Variance Relationship\n'
#              'Config 4: dU=1.0, dV=0.1, dW=0.0', fontsize=13, pad=15)
# ax.legend(fontsize=11)
# ax.grid(True, alpha=0.3)
# ax.set_aspect('equal')  # Makes it easier to see if slope = 1

# # Compute slope (fit line through origin)
# slope = np.sum(mean_eigs * std_eigs) / np.sum(mean_eigs**2)
# ax.plot([0, max_val], [0, slope*max_val], ':', color='green', linewidth=2,
#         label=f'Best fit: Std = {slope:.2f} × Mean', alpha=0.7)

# # Update legend
# ax.legend(fontsize=10)

# plt.tight_layout()
# plt.savefig('config4_fig3_mean_vs_std.png', dpi=300, bbox_inches='tight')
# print("Saved: config4_fig3_mean_vs_std.png")
# plt.show()

# Print relationship
# print("\n" + "="*70)
# print("MEAN-VARIANCE RELATIONSHIP:")
# print("="*70)
# for i, cv in enumerate(CV_values):
#     ratio = std_eigs[i] / mean_eigs[i] if mean_eigs[i] > 0 else 0
#     print(f"CV={cv:.2f}: Mean={mean_eigs[i]:.4f}, Std={std_eigs[i]:.4f}, "
#           f"Ratio={ratio:.4f}")


# ============================================================================
# FIGURE: RING EIGENVALUE CONVERGENCE WITH N (homogeneous baseline)
# ============================================================================

# import matplotlib.pyplot as plt
# import numpy as np

# # Data from the multi-N homogeneous check
# N_values = np.array([5, 10, 20, 30])

# eig_config_13 = np.array([
#     0.140789,   # N=5
#     0.143233,   # N=10
#     0.143233,   # N=20
#     0.143314,   # N=30
# ])

# eig_config_2 = np.array([
#     -0.086013,  # N=5  (below Turing threshold)
#     0.086159,   # N=10
#     0.135297,   # N=20
#     0.172997,   # N=30
# ])

# # Continuous-tissue reference values (from Obj 1 max_growth_rate)
# cont_13 = 0.143
# cont_2  = 0.173

# fig, ax = plt.subplots(figsize=(10, 6.5))

# # ===== Config 13 (always above threshold) =====
# ax.plot(N_values, eig_config_13, '-', color='darkblue', linewidth=2.5,
#         zorder=4)
# ax.scatter(N_values, eig_config_13, color='darkblue', s=120, zorder=5,
#            edgecolors='black', linewidths=0.8,
#            label='Config 13 (robust)')

# # ===== Config 2: separate below-threshold (open) from above-threshold (filled) =====
# below_threshold = eig_config_2 < 0
# ax.plot(N_values, eig_config_2, '-', color='crimson', linewidth=2.5,
#         zorder=4)

# # Filled markers for above-threshold points
# ax.scatter(N_values[~below_threshold], eig_config_2[~below_threshold],
#            color='crimson', s=120, zorder=5,
#            edgecolors='black', linewidths=0.8,
#            label='Config 2 (fragile)')

# # Open marker for the below-threshold point
# ax.scatter(N_values[below_threshold], eig_config_2[below_threshold],
#            facecolors='white', edgecolors='crimson', linewidths=2.5,
#            s=120, zorder=5,
#            label='Config 2 below Turing threshold')

# # ===== Continuous-tissue reference lines =====
# ax.axhline(y=cont_13, color='darkblue', linestyle=':', linewidth=1.5,
#            alpha=0.7, label='Config 13 continuous limit')
# ax.axhline(y=cont_2, color='crimson', linestyle=':', linewidth=1.5,
#            alpha=0.7, label='Config 2 continuous limit')

# # ===== Turing threshold =====
# ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5,
#            alpha=0.6, label='Turing threshold (Re(λ)=0)')

# # ===== Annotation: highlight the N=5 below-threshold point =====
# ax.annotate('Not Turing\nat N=5',
#             xy=(5, -0.086), xytext=(8, -0.13),
#             fontsize=9, color='crimson',
#             arrowprops=dict(arrowstyle='->', color='crimson', alpha=0.6))

# # ===== Axes and labels =====
# ax.set_xlabel('N (number of cells in ring)', fontsize=13)
# ax.set_ylabel('Max Re(λ) of homogeneous ring', fontsize=13)
# ax.set_title('Ring Eigenvalue Convergence with N\n'
#              'Discrete sampling approaches continuous-tissue limit',
#              fontsize=13, pad=12)

# ax.set_xticks(N_values)
# ax.set_xlim(3, 32)
# ax.grid(True, alpha=0.3)

# ax.legend(fontsize=9, loc='center right', framealpha=0.95)

# plt.tight_layout()
# plt.savefig('fig_convergence_N.png', dpi=300, bbox_inches='tight')
# print("Saved: fig_convergence_N.png")
# plt.close()

# ========================================================================================================================================================
# #                                                              FIGURE 5: ROBUSTNESS VS CV — COMPARISON
# # ========================================================================================================================================================

# # Build a list of configs to plot. Each entry is one line on the plot.
# # Easy to extend later when you add 1754 data.

# robustness_curves = [
#     {
#         'label': '#3954 robust (config 13)',
#         'CV': cv13['CV'],
#         'robustness': cv13['robustness'],
#         'color': 'darkblue',
#         'marker': 'o',
#     },
#     {
#         'label': '#3954 fragile (config 2)',
#         'CV': cv2['CV'],
#         'robustness': cv2['robustness'],
#         'color': 'crimson',
#         'marker': 's',
#     },
#     # When you have 1754 data, just uncomment and fill in:
#     # {
#     #     'label': '#1754 robust (config X)',
#     #     'CV': cv1754_robust['CV'],
#     #     'robustness': cv1754_robust['robustness'],
#     #     'color': 'darkgreen',
#     #     'marker': '^',
#     # },
#     # {
#     #     'label': '#1754 fragile (config Y)',
#     #     'CV': cv1754_fragile['CV'],
#     #     'robustness': cv1754_fragile['robustness'],
#     #     'color': 'darkorange',
#     #     'marker': 'D',
#     # },
# ]


# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot each robustness curve
# for curve in robustness_curves:
#     ax.plot(curve['CV'], curve['robustness'],
#             marker=curve['marker'], color=curve['color'],
#             linewidth=2.5, markersize=9,
#             label=curve['label'], zorder=3)

# # Reference line at 50% robustness
# ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
#            label='50% threshold', zorder=2)

# # Reference line at 0% (full collapse)
# ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, zorder=1)

# # Labels and styling
# ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
# ax.set_ylabel('Robustness (% of trials with Turing instability)', fontsize=13)
# ax.set_title('Robustness to Parameter Heterogeneity\n'
#              'Fraction of trials maintaining max Re(λ) > 0',
#              fontsize=13, pad=15)

# ax.set_xlim(-0.01, 0.42)
# ax.set_ylim(-5, 105)
# ax.set_yticks([0, 25, 50, 75, 100])
# ax.grid(True, alpha=0.3)

# ax.legend(fontsize=11, loc='center right', framealpha=0.95)

# plt.tight_layout()
# plt.savefig('fig5_robustness_vs_cv.png', dpi=300, bbox_inches='tight')
# print("Saved: fig5_robustness_vs_cv.png")
# plt.close()


# ========================================================================================================================================================
# #                                                              FIGURE 4: BIFURCATION DIAGRAM (Re(λ) vs K_uu)
# # ========================================================================================================================================================

# import matplotlib.pyplot as plt
# import pandas as pd
# import sys
# sys.path.append('.')
# from homogenous_ring import ode_system, build_ring_jacobian_homogeneous

# # Load both configs' baselines from the Obj 1 CSV
# df_params = pd.read_csv('../TopologyRanking/Topology3954/3954_NEW_lhs_results_parameters.csv')

# def load_config(config_id):
#     row = df_params[(df_params['config_id'] == config_id) &
#                     (df_params['param_rank'] == 1)].iloc[0]
#     baseline_params = np.array([
#         row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
#         row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
#         row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
#     ])
#     baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
#     hopping_dict = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}
#     return baseline_params, baseline_ss, hopping_dict

# params_13, ss_13, hopping_13 = load_config(13)
# params_2,  ss_2,  hopping_2  = load_config(2)

# ALT_STATE_SEED = np.array([0.007, 0.003, 1.95])
# PARAM_INDEX = 2  # K_uu
# N_CELLS = 10


# def sweep_bifurcation_eigenvalue(baseline_params, baseline_ss, hopping_dict,
#                                   param_idx, sweep_range=(0.80, 1.20),
#                                   n_points=300):
#     """Sweep one parameter and record max Re(λ) of the homogeneous ring
#     for both Branch A (baseline state) and Branch B (alternative state)."""
    
#     baseline_val = baseline_params[param_idx]
#     sweep_factors = np.linspace(sweep_range[0], sweep_range[1], n_points)
    
#     branch_A_eig = []
#     branch_B_eig = []
    
#     for factor in sweep_factors:
#         p = baseline_params.copy()
#         p[param_idx] = baseline_val * factor
        
#         # ---- Branch A: start from baseline steady state ----
#         ss_A = fsolve(ode_system, baseline_ss, args=(p,))
#         if np.max(np.abs(ode_system(ss_A, p))) < 1e-8 and np.all(ss_A > 0):
#             if np.max(np.abs(ss_A - baseline_ss) / baseline_ss) < 0.5:
#                 # Build the ring Jacobian at this state/params
#                 J_ring = build_ring_jacobian_homogeneous(N_CELLS, ss_A, p, hopping_dict)
#                 eig = np.max(np.real(np.linalg.eigvals(J_ring)))
#                 branch_A_eig.append((factor, eig))
        
#         # ---- Branch B: start from alternative seed ----
#         ss_B = fsolve(ode_system, ALT_STATE_SEED, args=(p,))
#         if np.max(np.abs(ode_system(ss_B, p))) < 1e-8 and np.all(ss_B > 0):
#             if np.max(np.abs(ss_B - baseline_ss) / baseline_ss) > 0.5:
#                 J_ring = build_ring_jacobian_homogeneous(N_CELLS, ss_B, p, hopping_dict)
#                 eig = np.max(np.real(np.linalg.eigvals(J_ring)))
#                 branch_B_eig.append((factor, eig))
    
#     return np.array(branch_A_eig), np.array(branch_B_eig)


# # ===== Run the sweeps =====
# print("Computing bifurcation sweep for config 13...")
# branch_A_13, branch_B_13 = sweep_bifurcation_eigenvalue(params_13, ss_13, hopping_13, PARAM_INDEX)
# print(f"  Branch A: {len(branch_A_13)} points")
# print(f"  Branch B: {len(branch_B_13)} points")

# print("Computing bifurcation sweep for config 2...")
# branch_A_2, branch_B_2 = sweep_bifurcation_eigenvalue(params_2, ss_2, hopping_2, PARAM_INDEX)
# print(f"  Branch A: {len(branch_A_2)} points")
# print(f"  Branch B: {len(branch_B_2)} points")


# # ===== Build the figure =====

# def plot_eigenvalue_bifurcation(ax, baseline_params, baseline_ss,
#                                   branch_A, branch_B, baseline_eig, title):
#     """Plot one panel of the bifurcation figure."""
    
#     baseline_val = baseline_params[PARAM_INDEX]
#     x_min = baseline_val * 0.78
#     x_max = baseline_val * 1.22
    
#     # ---- Determine y-axis range from data ----
#     y_vals = [baseline_eig]
#     if len(branch_A) > 0:
#         y_vals.extend(branch_A[:, 1].tolist())
#     if len(branch_B) > 0:
#         y_vals.extend(branch_B[:, 1].tolist())
    
#     y_data_min = min(y_vals)
#     y_data_max = max(y_vals)
#     y_padding = max(0.05, 0.15 * (y_data_max - y_data_min))
#     y_min = y_data_min - y_padding
#     y_max = y_data_max + y_padding
    
#     # ---- Plot the Turing threshold first so everything sits on top ----
#     ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
#                label='Turing threshold (Re(λ)=0)', zorder=2)
    
#     # ---- Shade only if there's a real gap in Branch A within the sweep ----
#     if len(branch_A) > 0:
#         branch_A_x = branch_A[:, 0] * baseline_val
#         a_left = branch_A_x.min()
#         a_right = branch_A_x.max()
#         sweep_width = x_max - x_min
#         threshold = 0.05 * sweep_width
        
#         has_left_gap = (a_left - x_min) > threshold
#         has_right_gap = (x_max - a_right) > threshold
        
#         if has_left_gap:
#             ax.axvspan(x_min, a_left, alpha=0.15, color='red', zorder=0)
#         if has_right_gap:
#             ax.axvspan(a_right, x_max, alpha=0.15, color='red', zorder=0)
    
#     # ---- Plot Branch A ----
#     if len(branch_A) > 0:
#         ax.plot(branch_A[:, 0] * baseline_val, branch_A[:, 1], '-',
#                 color='darkblue', linewidth=3.5,
#                 label='Branch A (baseline state)', zorder=5)
    
#     # ---- Plot Branch B ----
#     if len(branch_B) > 0:
#         ax.plot(branch_B[:, 0] * baseline_val, branch_B[:, 1], '-',
#                 color='darkred', linewidth=3.5,
#                 label='Branch B (alternative state)', zorder=5)
    
#     # ---- Bifurcation point markers (where Branch A ends) ----
#     if len(branch_A) > 0:
#         branch_A_x = branch_A[:, 0] * baseline_val
#         sweep_width = x_max - x_min
#         threshold = 0.05 * sweep_width
        
#         if (branch_A_x.min() - x_min) > threshold:
#             ax.plot(branch_A_x.min(),
#                     branch_A[np.argmin(branch_A_x), 1],
#                     marker='*', color='red', markersize=22,
#                     markeredgecolor='black', markeredgewidth=1,
#                     zorder=10, linestyle='None',
#                     label='Bifurcation point')
#         if (x_max - branch_A_x.max()) > threshold:
#             label = None if (branch_A_x.min() - x_min) > threshold else 'Bifurcation point'
#             ax.plot(branch_A_x.max(),
#                     branch_A[np.argmax(branch_A_x), 1],
#                     marker='*', color='red', markersize=22,
#                     markeredgecolor='black', markeredgewidth=1,
#                     zorder=10, linestyle='None',
#                     label=label)
    
#     # ---- Baseline vertical line and dot ----
#     ax.axvline(x=baseline_val, color='black', linestyle=':', linewidth=1.5,
#                alpha=0.7, zorder=3)
#     ax.plot(baseline_val, baseline_eig, 'o', color='black', markersize=11,
#             zorder=11, label='Baseline (working state)')
    
#     # ---- ±10% perturbation lines ----
#     for sign, label in [(+1, '+10%'), (-1, '−10%')]:
#         x_pert = baseline_val * (1 + sign * 0.10)
#         ax.axvline(x=x_pert, color='gray', linestyle='--', linewidth=1.5,
#                    alpha=0.6, zorder=2)
#         # Place labels at bottom for clarity
#         ax.text(x_pert, y_min + (y_max - y_min) * 0.03, label,
#                 fontsize=10, color='dimgray', ha='center', va='bottom',
#                 fontweight='bold')
    
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xlabel('K_uu (perturbed)', fontsize=12)
#     ax.set_ylabel('Max Re(λ) of ring Jacobian', fontsize=12)
#     ax.set_title(title, fontsize=13, pad=10)
#     ax.grid(True, alpha=0.3)


# # ===== Build the combined figure =====
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# # For baseline eigenvalues, recompute at the actual baseline
# J_baseline_13 = build_ring_jacobian_homogeneous(N_CELLS, ss_13, params_13, hopping_13)
# baseline_eig_13 = np.max(np.real(np.linalg.eigvals(J_baseline_13)))

# J_baseline_2 = build_ring_jacobian_homogeneous(N_CELLS, ss_2, params_2, hopping_2)
# baseline_eig_2 = np.max(np.real(np.linalg.eigvals(J_baseline_2)))

# plot_eigenvalue_bifurcation(ax1, params_13, ss_13, branch_A_13, branch_B_13,
#                              baseline_eig_13,
#                              'Config 13 (robust)\n'
#                              'Smooth response, no bifurcation')

# plot_eigenvalue_bifurcation(ax2, params_2, ss_2, branch_A_2, branch_B_2,
#                              baseline_eig_2,
#                              'Config 2 (fragile)\n'
#                              '±10% perturbation crosses bifurcation: Turing lost')

# # Combined legend below both panels
# handles, labels = ax2.get_legend_handles_labels()
# from matplotlib.patches import Patch
# handles.append(Patch(facecolor='red', alpha=0.15,
#                      label='Branch A absent (bifurcation crossed)'))
# labels.append('Branch A absent (bifurcation crossed)')

# # Deduplicate labels (in case both panels have the same legend entry)
# seen = set()
# unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
# handles_u, labels_u = zip(*unique)

# fig.legend(handles_u, labels_u, loc='lower center', ncol=3, fontsize=10,
#            bbox_to_anchor=(0.5, -0.05), frameon=True)

# fig.suptitle('Bifurcation Diagram: Turing Growth Rate vs K_uu',
#              fontsize=15, y=1.00)

# plt.tight_layout(rect=[0, 0.05, 1, 0.98])
# plt.savefig('fig4_bifurcation_eigenvalue.png', dpi=300, bbox_inches='tight')
# print("Saved: fig4_bifurcation_eigenvalue.png")
# plt.close()