#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

# Load results
with open('3954_cv_sweep_low_config4.pkl', 'rb') as f: # CHANGE HERE BASED ON FILE
    data = pickle.load(f)

results = data['results']

# Extract data
CV_values = [r['CV'] for r in results]
mean_eigs = [r['mean_eig'] for r in results]
min_eigs = [r['min_eig'] for r in results]
max_eigs = [r['max_eig'] for r in results]
std_eigs = [r['std_eig'] for r in results]

CV_values = np.array(CV_values)
mean_eigs = np.array(mean_eigs)
min_eigs = np.array(min_eigs)
max_eigs = np.array(max_eigs)
std_eigs = np.array(std_eigs)

# ========================================================================================================================================================
#                                                              BIOLOGICAL REALISM PLOTS
# ========================================================================================================================================================

# ============================================================================
# FIGURE 1: SIMPLIFIED: Mean with full range
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(CV_values, min_eigs, max_eigs, alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)

# Plot mean line
ax.plot(CV_values, mean_eigs, 'o-', color='darkblue', linewidth=2.5, 
        markersize=8, label='Mean Re(λ)', zorder=3)

# Horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

# Labels
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('3954 Turing Growth Rate vs Parameter Heterogeneity\n'
             'Config 4: dU=1.0, dV=0.1, dW=0.0', fontsize=13, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('config4_fig1_mean_range_vs_cv.png', dpi=300, bbox_inches='tight') # CHANGE HERE BASED ON FILE
print("Saved: config4_fig1_mean_range_vs_cv.png")
plt.show()


# ============================================================================
# FIGURE 2: BOXPLOT: Distribution of Re(λ) at each CV
# ============================================================================

# Extract data
CV_values = [r['CV'] for r in results]
all_eigenvalues = [r['all_eigenvalues'] for r in results]
robustness = [r['robustness'] for r in results]

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot
bp = ax.boxplot(all_eigenvalues, 
                positions=range(len(CV_values)),
                widths=0.6,
                patch_artist=True,
                showfliers=True,  # Show outliers
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

# Color boxes
for patch in bp['boxes']:
    patch.set_facecolor('palevioletred')
    patch.set_alpha(1.0)

# Horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
           label='Turing threshold (Re(λ)=0)', zorder=10)

# X-axis labels
ax.set_xticks(range(len(CV_values)))
ax.set_xticklabels([f'{cv:.2f}' for cv in CV_values])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('3954 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             'Config 4: dU=1.0, dV=0.1, dW=0.0 (1000 trials per CV)', 
             fontsize=13, pad=15)

ax.legend(fontsize=9, framealpha=0.9)  # Was fontsize=11
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('config4_fig2_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight') # CHANGE HERE BASED ON FILE
print("Saved: config4_fig2_boxplot_cv_sweep.png")



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





# ========================================================================================================================================================
#                                                              SENSITIVITY ANALYSIS PLOTS
# ========================================================================================================================================================

# ============================================================================
# FIGURE 4: SENSITIVITY ANALYSIS
# ============================================================================

with open('sensitivity_results_config4.pkl', 'rb') as f: # CHANGE HERE BASED ON FILE
    sens_data = pickle.load(f)

param_names = sens_data['param_names']
sensitivities = sens_data['sensitivities']

# Better parameter labels
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

# Sort by sensitivity
sorted_indices = np.argsort(sensitivities)[::-1]
sorted_names = [param_names[i] for i in sorted_indices]
sorted_labels = [PARAM_LABELS[name] for name in sorted_names]
sorted_sens = [sensitivities[i] for i in sorted_indices]

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
bars = ax.bar(range(len(sorted_labels)), sorted_sens, color='steelblue', alpha=0.9)

# Color top 3 (stiff) in red
for i in range(min(3, len(bars))):
    bars[i].set_color('mediumvioletred')
    bars[i].set_alpha(0.9)

# Color bottom 3 (sloppy) in gray
for i in range(max(0, len(bars)-3), len(bars)):
    bars[i].set_color('silver')
    bars[i].set_alpha(0.9)

# Labels
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Change in Turing Growth Rate per 10% Parameter Change (log scale)', fontsize=12)
ax.set_title('3954 Parameter Sensitivity Analysis: Stiff vs Sloppy\nWhich Parameters Control Turing Patterns?', fontsize=13, pad=15)

# Only 4 clean tick marks
# ax.set_yticks([0.001, 0.01, 0.1, 1.0])
# ax.set_yticklabels(['0.001', '0.01', '0.1', '1.0'])
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y', which='major')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='mediumvioletred', alpha=0.8, label='Stiff (critical)'),
    Patch(facecolor='steelblue', alpha=0.8, label='Moderate'),
    Patch(facecolor='silver', alpha=0.8, label='Sloppy (tolerant)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Annotation
ratio = sorted_sens[0] / sorted_sens[-1]
# ax.text(0.02, 0.98, 
#         f'Stiffest: {sorted_labels[0]}\nS = {sorted_sens[0]:.3f}\n\n'
#         f'Sloppiest: {sorted_labels[-1]}\nS = {sorted_sens[-1]:.5f}\n\n'
#         f'Ratio: {ratio:.0f}',
#         transform=ax.transAxes,
#         fontsize=10,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('config4_sensitivity_stiff_vs_sloppy.png', dpi=300, bbox_inches='tight') # CHANGE HERE BASED ON FILE
print("Saved: config4_sensitivity_stiff_vs_sloppy.png")


# ANOTHER TYPE OF PLOT

# Group by node
u_params = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u']
v_params = ['alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v']
w_params = ['alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']

# Sum sensitivities per node
u_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in u_params)
v_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in v_params)
w_total = sum(sensitivities[i] for i, p in enumerate(param_names) if p in w_params)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(['u (activator)', 'v (inhibitor)', 'w (immobile)'], 
       [u_total, v_total, w_total], color=['green', 'blue', 'purple'])
ax.set_ylabel('Total Sensitivity')
ax.set_title('3954 Which Gene Controls Pattern Formation?')
