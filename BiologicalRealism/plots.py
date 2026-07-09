#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import fsolve
import matplotlib.lines as mlines

from heterogenous_ring_3954 import (
    hopping,
)

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

def extract_cv_arrays(cv):
    results = cv['results']
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
        'config_name': cv.get('config_name', 'unknown'),
        'config_id': cv.get('config_id', '?'),
        'hopping': cv.get('hopping', {})
    }

# Helper for diffusion string in plot titles
def diff_str(hopping):
    return f"dU={hopping['h_u']}, dV={hopping['h_v']}, dW={hopping['h_w']}"


# LOADING ALL FILES
with open('3954_cv_sweep_high_config43_N10.pkl', 'rb') as f: 
    cv_robust_3954_N10 = pickle.load(f)
with open('3954_cv_sweep_high_config43_N20.pkl', 'rb') as f:
    cv_robust_3954_N20 = pickle.load(f)
with open('3954_cv_sweep_high_config43_N30.pkl', 'rb') as f:
    cv_robust_3954_N30 = pickle.load(f)

with open('3954_cv_sweep_low_config17_N10.pkl', 'rb') as f:
    cv_fragile_3954_N10 = pickle.load(f)
with open('3954_cv_sweep_low_config17_N20.pkl', 'rb') as f:
    cv_fragile_3954_N20 = pickle.load(f)
with open('3954_cv_sweep_low_config17_N30.pkl', 'rb') as f:
    cv_fragile_3954_N30 = pickle.load(f)

with open('1754_cv_sweep_high_config43_N10.pkl', 'rb') as f:
    cv_robust_1754_N10 = pickle.load(f)
with open('1754_cv_sweep_high_config43_N20.pkl', 'rb') as f:
    cv_robust_1754_N20 = pickle.load(f)
with open('1754_cv_sweep_high_config43_N30.pkl', 'rb') as f:
    cv_robust_1754_N30 = pickle.load(f)

with open('1754_cv_sweep_low_config14_N10.pkl', 'rb') as f:
    cv_fragile_1754_N10 = pickle.load(f)
with open('1754_cv_sweep_low_config14_N20.pkl', 'rb') as f:
    cv_fragile_1754_N20 = pickle.load(f)
with open('1754_cv_sweep_low_config14_N30.pkl', 'rb') as f:
    cv_fragile_1754_N30 = pickle.load(f)

# Extract robustness arrays using your existing helper
cv_robust_3954_N10 = extract_cv_arrays(cv_robust_3954_N10)
cv_robust_3954_N20 = extract_cv_arrays(cv_robust_3954_N20)
cv_robust_3954_N30 = extract_cv_arrays(cv_robust_3954_N30)
cv_fragile_3954_N10 = extract_cv_arrays(cv_fragile_3954_N10)
cv_fragile_3954_N20 = extract_cv_arrays(cv_fragile_3954_N20)
cv_fragile_3954_N30 = extract_cv_arrays(cv_fragile_3954_N30)

cv_robust_1754_N10 = extract_cv_arrays(cv_robust_1754_N10)
cv_robust_1754_N20 = extract_cv_arrays(cv_robust_1754_N20)
cv_robust_1754_N30 = extract_cv_arrays(cv_robust_1754_N30)
cv_fragile_1754_N10 = extract_cv_arrays(cv_fragile_1754_N10)
cv_fragile_1754_N20 = extract_cv_arrays(cv_fragile_1754_N20)
cv_fragile_1754_N30 = extract_cv_arrays(cv_fragile_1754_N30)

# SENSITIVITY ANALYSIS FILES
with open('3954_sensitivity_results_config43_N10.pkl', 'rb') as f:
    sens_3954_robust = pickle.load(f)
with open('3954_sensitivity_results_config17_N10.pkl', 'rb') as f:
    sens_3954_fragile = pickle.load(f)
with open('1754_sensitivity_results_config43_N10.pkl', 'rb') as f:
    sens_1754_robust = pickle.load(f)
with open('1754_sensitivity_results_config14_N10.pkl', 'rb') as f:
    sens_1754_fragile = pickle.load(f)


# ========================================================================================================================================================
#                                                              BIOLOGICAL REALISM PLOTS
# ========================================================================================================================================================

################ TOPOLOGY 3954 ################

# ============================================================================
# FIGURE 1A: Mean with full range Config 13
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(cv_robust_3954_N10['CV'], cv_robust_3954_N10['min'], cv_robust_3954_N10['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)

# plot mean line
ax.plot(cv_robust_3954_N10['CV'], cv_robust_3954_N10['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)

# horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

# labels and title
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Turing Growth Rate vs Parameter Heterogeneity\n'f'ID 43 (robust): {diff_str(cv_robust_3954_N10["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# save the plots
plt.tight_layout()
plt.savefig('3954_config43_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved as 3954_config43_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2A: BOXPLOT Config 43
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot
bp = ax.boxplot(cv_robust_3954_N10['all'],
                positions=range(len(cv_robust_3954_N10['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

# color boxes
for patch in bp['boxes']:
    patch.set_facecolor('cornflowerblue') # slateblue
    patch.set_alpha(0.8)

# horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv_robust_3954_N10['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv_robust_3954_N10['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)

ax.set_title(f'3954 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'ID 43 (robust): {diff_str(cv_robust_3954_N10["hopping"])}',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)  # Was fontsize=11
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('3954_config43_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: 3954_config43_boxplot_cv_sweep.png")
plt.close()


# ============================================================================
# FIGURE 1B: Mean with full range  
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(cv_fragile_3954_N10['CV'], cv_fragile_3954_N10['min'], cv_fragile_3954_N10['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)
ax.plot(cv_fragile_3954_N10['CV'], cv_fragile_3954_N10['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Turing Growth Rate vs Parameter Heterogeneity\n' f'ID 17 (fragile): {diff_str(cv_fragile_3954_N10["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3954_config17_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved: 3954_config17_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2B: Boxplot
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(cv_fragile_3954_N10['all'],
                positions=range(len(cv_fragile_3954_N10['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

for patch in bp['boxes']:
    patch.set_facecolor('lightskyblue')
    patch.set_alpha(1.0)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv_fragile_3954_N10['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv_fragile_3954_N10['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'3954 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'ID 17 (fragile): {diff_str(cv_fragile_3954_N10["hopping"])} 1000 trials per CV',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('3954_config17_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: 3954_config17_boxplot_cv_sweep.png")
plt.close()








################ TOPOLOGY 1754 ################

# ============================================================================
# FIGURE 1A: Mean with full range Config 43
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(cv_robust_1754_N10['CV'], cv_robust_1754_N10['min'], cv_robust_1754_N10['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)

# plot mean line
ax.plot(cv_fragile_1754_N10['CV'], cv_fragile_1754_N10['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)

# horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'1754 Turing Growth Rate vs Parameter Heterogeneity\n'f'ID 43 (robust): {diff_str(cv_fragile_1754_N10["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1754_config43_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved as config43_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2A: BOXPLOT Config 43
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot
bp = ax.boxplot(cv_robust_1754_N10['all'],
                positions=range(len(cv_robust_1754_N10['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

# color boxes
for patch in bp['boxes']:
    patch.set_facecolor('blueviolet')
    patch.set_alpha(0.8)

# horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv_robust_1754_N10['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv_robust_1754_N10['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)

ax.set_title(f'1754 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'ID 43(robust): {diff_str(cv_robust_1754_N10["hopping"])}',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)  # Was fontsize=11
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('1754_config43_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: 1754_config43_boxplot_cv_sweep.png")
plt.close()


# ============================================================================
# FIGURE 1B: Mean with full range  Config 12
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(cv_fragile_1754_N10['CV'], cv_fragile_1754_N10['min'], cv_fragile_1754_N10['max'], alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)
ax.plot(cv_fragile_1754_N10['CV'], cv_fragile_1754_N10['mean'], 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean Re(λ)', zorder=3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'1754 Turing Growth Rate vs Parameter Heterogeneity\n' f'ID(fragile): {diff_str(cv_fragile_1754_N10["hopping"])}', fontsize=12, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1754_config14_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved: 1754_config14_mean_range_vs_cv.png")
plt.close()

# ============================================================================
# FIGURE 2B: Boxplot  Config 12
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(cv_fragile_1754_N10['all'],
                positions=range(len(cv_fragile_1754_N10['CV'])),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

for patch in bp['boxes']:
    patch.set_facecolor('mediumorchid')
    patch.set_alpha(1.0)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold (Re(λ)=0)', zorder=10)

ax.set_xticks(range(len(cv_fragile_1754_N10['CV'])))
ax.set_xticklabels([f'{cv:.2f}' for cv in cv_fragile_1754_N10['CV']])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=12)
ax.set_ylabel('Max Re(λ)', fontsize=12)
ax.set_title(f'1754 Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             f'ID 14 (fragile): {diff_str(cv_fragile_1754_N10["hopping"])} with 1000 trials per CV',
             fontsize=12, pad=15)

ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('1754_config14_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: 1754_config14_boxplot_cv_sweep.png")
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
    'K_uu': 'u self-activation',
    'K_vu': 'v-u inhibition',
    'delta_u': 'u degradation',
    'alpha_v': 'v basal production',
    'beta_v': 'v regulated production',
    'K_uv': 'u-v activation',
    'K_wv': 'w-v inhibition',
    'delta_v': 'v degradation',
    'alpha_w': 'w basal production',
    'beta_w': 'w regulated production',
    'K_ww': 'w self-activation ',
    'K_uw': 'u-w inhibition',
    'K_vw': 'v-w inhibition',
    'delta_w': 'w degradation'
    }


def plot_sensitivity(sens, config_label, save_name):
    param_names = sens['param_names']
    sensitivities = np.array(sens['sensitivities'])
    sens_plus = np.array(sens['sensitivities_plus'])
    sens_minus = np.array(sens['sensitivities_minus'])

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

    # Color top 3 clean bars (stiff)  only count non-NaN
    n_clean = (~sorted_is_nan).sum()
    for i in range(min(3, n_clean)):
        bars[i].set_color('mediumvioletred')

    # Color bottom 3 clean bars (sloppy)  last 3 non-NaN
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
    ax.set_title(f'Parameter Sensitivity Analysis: {config_label}\n'
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
# FIGURE 3A + 3B : Sensitivity 
# ============================================================================
plot_sensitivity(sens_3954_robust, 'ID 43 (robust)', '3954_config43_sensitivity.png')

plot_sensitivity(sens_3954_fragile, 'ID 17 (fragile)', '3954_config17_sensitivity.png')

plot_sensitivity(sens_1754_robust, 'ID 43 (robust)', '1754_config43_sensitivity.png')

plot_sensitivity(sens_1754_fragile, 'ID 14 (fragile)', '1754_config14_sensitivity.png')





# ========================================================================================================================================================
#                                                  FIGURE: ROBUSTNESS VS CV  N=10 vs N=20 COMPARISON
# ========================================================================================================================================================

robustness_curves = [
    {'label': '#3954 Robust (ID 43, N=10)','CV': cv_robust_3954_N10['CV'], 'robustness': cv_robust_3954_N10['robustness'],'color': 'blue', 'marker': 'o', 'linestyle': '-',},
    {'label': '#3954 Robust (ID 43, N=20)','CV': cv_robust_3954_N20['CV'], 'robustness': cv_robust_3954_N20['robustness'],'color': 'blue', 'marker': 'o', 'linestyle': '--',},
    {'label': '#3954 Robust (ID 43, N=30)','CV': cv_robust_3954_N30['CV'], 'robustness': cv_robust_3954_N30['robustness'],'color': 'blue', 'marker': 'o', 'linestyle': ':',},
    
    {'label': '#3954 Fragile (ID 17, N=10)','CV':  cv_fragile_3954_N10['CV'], 'robustness': cv_fragile_3954_N10['robustness'],'color': 'cornflowerblue', 'marker': 's', 'linestyle': '-',},
    {'label': '#3954 Fragile (ID 17, N=20)','CV':  cv_fragile_3954_N20['CV'], 'robustness': cv_fragile_3954_N20['robustness'],'color': 'cornflowerblue', 'marker': 's', 'linestyle': '--',},
    {'label': '#3954 Fragile (ID 17, N=30)','CV':  cv_fragile_3954_N30['CV'], 'robustness': cv_fragile_3954_N30['robustness'],'color': 'cornflowerblue', 'marker': 's', 'linestyle': ':',},
    
    {'label': '#1754 Robust (ID 43, N=10)','CV': cv_robust_1754_N10['CV'], 'robustness': cv_robust_1754_N10['robustness'],'color': 'purple', 'marker': '^', 'linestyle': '-',},
    {'label': '#1754 Robust (ID 43, N=20)','CV': cv_robust_1754_N20['CV'], 'robustness': cv_robust_1754_N20['robustness'],'color': 'purple', 'marker': '^', 'linestyle': '--',},
    {'label': '#1754 Robust (ID 43, N=30)','CV': cv_robust_1754_N30['CV'], 'robustness': cv_robust_1754_N30['robustness'],'color': 'purple', 'marker': '^', 'linestyle': ':',},
    
    {'label': '#1754 Fragile (ID 14, N=10)','CV': cv_fragile_1754_N10['CV'], 'robustness': cv_fragile_1754_N10['robustness'],'color': 'mediumorchid', 'marker': 'D', 'linestyle': '-',},
    {'label': '#1754 Fragile (ID 14, N=20)','CV': cv_fragile_1754_N20['CV'], 'robustness': cv_fragile_1754_N20['robustness'],'color': 'mediumorchid', 'marker': 'D', 'linestyle': '--',},
    {'label': '#1754 Fragile (ID 14, N=30)','CV': cv_fragile_1754_N30['CV'], 'robustness': cv_fragile_1754_N30['robustness'],'color': 'mediumorchid', 'marker': 'D', 'linestyle': ':',},
]

# combined plot with two subplots for each topology
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, topo in zip(axes, ['#3954', '#1754']):
    # Plot reference background guides
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Extract and plot matching topology metrics
    for curve in robustness_curves:
        if topo in curve['label']:
            ax.plot(
                curve['CV'], curve['robustness'],
                marker=curve['marker'], color=curve['color'],
                linestyle=curve['linestyle'], linewidth=2, markersize=6.5,
                markeredgecolor='white', markeredgewidth=1.0,
                label=curve['label'], zorder=3
            )
            
    ax.set_title(f'Topology {topo} Stability Profiles', fontsize=12, fontweight='semibold', pad=12)
    ax.set_xlabel('CV (Coefficient of Variation)', fontsize=10.5, color='#333333', labelpad=8)
    ax.set_xlim(-0.01, 0.42)
    ax.set_ylim(-3, 103)
    ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc', zorder=0)
    ax.tick_params(axis='both', labelsize=9.5)

axes[0].set_ylabel('Robustness (% of trials with Turing instability)', fontsize=10.5, color='#333333', labelpad=8)

title_obj = fig.suptitle(
    'Robustness to Heterogeneity Across Ring Sizes (N=10, N=20, N=30)\n'
    'Comparison of Topological Stability Profiles Under Stochastic Parameter Variation',
    fontsize=12.5, y=0.96, fontweight='semibold', color='#111111'
)

# # Expanded unified legend handles matching the 3-step N scale
# legend_handles = [
#     mlines.Line2D([], [], color='blue', marker='o', linestyle='-', label='#3954 Robust (ID 43, N=10)'),
#     mlines.Line2D([], [], color='blue', marker='o', linestyle='--', label='#3954 Robust (Config 45, N=20)'),
#     mlines.Line2D([], [], color='blue', marker='o', linestyle=':', label='#3954 Robust (Config 45, N=30)'),
#     mlines.Line2D([], [], color='cornflowerblue', marker='s', linestyle='-', label='#3954 Fragile (Config 4, N=10)'),
#     mlines.Line2D([], [], color='cornflowerblue', marker='s', linestyle='--', label='#3954 Fragile (Config 4, N=20)'),
#     mlines.Line2D([], [], color='cornflowerblue', marker='s', linestyle='--', label='#3954 Fragile (Config 4, N=30)'),
#     mlines.Line2D([], [], color='purple', marker='^', linestyle='-', label='#1754 Robust (Config 35, N=10)'),
#     mlines.Line2D([], [], color='purple', marker='^', linestyle='--', label='#1754 Robust (Config 35, N=20)'),
#     mlines.Line2D([], [], color='purple', marker='^', linestyle=':', label='#1754 Robust (Config 35, N=30)'),
#     mlines.Line2D([], [], color='mediumorchid', marker='D', linestyle='-', label='#1754 Fragile (Config 12, N=10)'),
#     mlines.Line2D([], [], color='mediumorchid', marker='D', linestyle='--', label='#1754 Fragile (Config 12, N=20)'),
#     mlines.Line2D([], [], color='mediumorchid', marker='D', linestyle='--', label='#1754 Fragile (Config 12, N=30)'),
# ]

# Automatically generate the handles directly from your curves dictionary
legend_handles = [
    mlines.Line2D(
        [],
        [],
        color=curve["color"],
        marker=curve["marker"],
        linestyle=curve["linestyle"],
        label=curve["label"],
    )
    for curve in robustness_curves
]

leg_obj = fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),  # Lowered further down to fit 3 horizontal legend rows comfortably
    ncol=4,                       
    frameon=False,
    fontsize=9.5
)

fig.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.22, wspace=0.14)

plt.savefig('thesis_robustness_N10_to_N30_combined.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[title_obj, leg_obj])
plt.close()

