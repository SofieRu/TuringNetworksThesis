#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import fsolve

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

### Load results for #3954 ###
# Config 13 (robust)
with open('3954_cv_sweep_high_config13.pkl', 'rb') as f:
    cv_data_13 = pickle.load(f)

with open('sensitivity_results_NEW_config13.pkl', 'rb') as f:
    sens_data_13 = pickle.load(f)

# Config 2 (fragile)
with open('3954_cv_sweep_low_config2.pkl', 'rb') as f:
    cv_data_2 = pickle.load(f)

with open('sensitivity_results_NEW_config2.pkl', 'rb') as f:
    sens_data_2 = pickle.load(f)

### Extract data ###
# results = data['results']
# CV_values = [r['CV'] for r in results]
# mean_eigs = [r['mean_eig'] for r in results]
# min_eigs = [r['min_eig'] for r in results]
# max_eigs = [r['max_eig'] for r in results]
# std_eigs = [r['std_eig'] for r in results]

# CV_values = np.array(CV_values)
# mean_eigs = np.array(mean_eigs)
# min_eigs = np.array(min_eigs)
# max_eigs = np.array(max_eigs)
# std_eigs = np.array(std_eigs)

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
                flierprops=dict(marker='o', markersize=3, alpha=0.3))


# color boxes
for patch in bp['boxes']:
    patch.set_facecolor('palevioletred')
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
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

for patch in bp['boxes']:
    patch.set_facecolor('palevioletred')
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
#                                                              BIFURCATION DIAGRAMS
# ========================================================================================================================================================

# Import from homogenous_ring.py for steady-state computation
import pandas as pd
import sys
sys.path.append('.')
from homogenous_ring import ode_system

# Load both configs' baseline parameters and steady states from the Obj 1 CSV
df_params = pd.read_csv('../TopologyRanking/Topology3954/3954_NEW_lhs_results_parameters.csv')

def load_config(config_id):
    row = df_params[(df_params['config_id'] == config_id) &
                    (df_params['param_rank'] == 1)].iloc[0]
    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    return baseline_params, baseline_ss

params_13, ss_13 = load_config(13)
params_2,  ss_2  = load_config(2)

# Alternative state seed (the "high-w" state we discovered when debugging)
ALT_STATE_SEED = np.array([0.007, 0.003, 1.95])

# Parameter to sweep: K_uu (index 2 in the parameter array)
PARAM_TO_SWEEP = 'K_uu'
PARAM_INDEX = 2


def sweep_bifurcation(baseline_params, baseline_ss, param_idx,
                     sweep_range=(0.80, 1.20), n_points=200):
    """For each parameter value, find Branch A (from baseline_ss)
    and Branch B (from ALT_STATE_SEED). Return both branches."""
    
    baseline_val = baseline_params[param_idx]
    sweep_factors = np.linspace(sweep_range[0], sweep_range[1], n_points)
    
    branch_A_w = []  # list of (factor, w*) for the baseline branch
    branch_B_w = []  # list of (factor, w*) for the alternative branch
    
    for factor in sweep_factors:
        perturbed_params = baseline_params.copy()
        perturbed_params[param_idx] = baseline_val * factor
        
        # Branch A: start from baseline steady state
        ss_A = fsolve(ode_system, baseline_ss, args=(perturbed_params,))
        res_A = ode_system(ss_A, perturbed_params)
        if np.max(np.abs(res_A)) < 1e-8 and np.all(ss_A > 0):
            # Only accept if it stays "near" baseline (else it found Branch B)
            rel_dist = np.max(np.abs(ss_A - baseline_ss) / baseline_ss)
            if rel_dist < 0.5:
                branch_A_w.append((factor, ss_A[2]))
        
        # Branch B: start from the high-w alternative seed
        ss_B = fsolve(ode_system, ALT_STATE_SEED, args=(perturbed_params,))
        res_B = ode_system(ss_B, perturbed_params)
        if np.max(np.abs(res_B)) < 1e-8 and np.all(ss_B > 0):
            # Only accept if it's clearly different from baseline (Branch B)
            rel_dist = np.max(np.abs(ss_B - baseline_ss) / baseline_ss)
            if rel_dist > 0.5:
                branch_B_w.append((factor, ss_B[2]))
    
    return np.array(branch_A_w), np.array(branch_B_w)


def plot_bifurcation(baseline_params, baseline_ss, config_label, save_name,
                     param_name='K_uu', param_idx=2):
    """Plot bifurcation diagram for one config."""
    
    branch_A, branch_B = sweep_bifurcation(baseline_params, baseline_ss, param_idx)
    
    baseline_val = baseline_params[param_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Branch A (baseline branch)
    if len(branch_A) > 0:
        ax.plot(branch_A[:, 0] * baseline_val, branch_A[:, 1], '-',
                color='darkblue', linewidth=2.5, label='Branch A (baseline state)')
    
    # Plot Branch B (alternative branch) if it exists
    if len(branch_B) > 0:
        ax.plot(branch_B[:, 0] * baseline_val, branch_B[:, 1], '-',
                color='darkred', linewidth=2.5, label='Branch B (alternative state)')
    
    # Mark baseline value
    ax.axvline(x=baseline_val, color='black', linestyle=':', linewidth=1.5,
               alpha=0.6, label=f'Baseline ({param_name})')
    ax.plot(baseline_val, baseline_ss[2], 'o', color='black', markersize=10,
            zorder=10, label='Baseline steady state')
    
    # Mark ±10% perturbation positions
    ax.axvline(x=baseline_val * 1.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=baseline_val * 0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(baseline_val * 1.1, ax.get_ylim()[1] * 0.95, ' +10%',
            fontsize=9, color='gray', va='top')
    ax.text(baseline_val * 0.9, ax.get_ylim()[1] * 0.95, '-10% ',
            fontsize=9, color='gray', va='top', ha='right')
    
    ax.set_xlabel(f'{param_name}', fontsize=13)
    ax.set_ylabel('Steady-state w concentration (w*)', fontsize=13)
    ax.set_title(f'3954 Bifurcation Diagram: {config_label}\n'
                 f'Sweeping {param_name} from -20% to +20% of baseline',
                 fontsize=13, pad=15)
    ax.legend(fontsize=10, loc='center left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


# ============================================================================
# FIGURE 4A: Bifurcation diagram — Config 13
# ============================================================================
plot_bifurcation(params_13, ss_13, 'Config 13 (robust)',
                 'config13_fig4_bifurcation_Kuu.png',
                 param_name='K_uu', param_idx=PARAM_INDEX)

# ============================================================================
# FIGURE 4B: Bifurcation diagram — Config 2
# ============================================================================
plot_bifurcation(params_2, ss_2, 'Config 2 (fragile)',
                 'config2_fig4_bifurcation_Kuu.png',
                 param_name='K_uu', param_idx=PARAM_INDEX)







# ========================================================================================================================================================
#                                                              BIFURCATION DIAGRAMS — COMBINED
# ========================================================================================================================================================

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('.')
from homogenous_ring import ode_system

# Load both configs' baselines
df_params = pd.read_csv('../TopologyRanking/Topology3954/3954_NEW_lhs_results_parameters.csv')

def load_config(config_id):
    row = df_params[(df_params['config_id'] == config_id) &
                    (df_params['param_rank'] == 1)].iloc[0]
    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    return baseline_params, baseline_ss

params_13, ss_13 = load_config(13)
params_2,  ss_2  = load_config(2)

ALT_STATE_SEED = np.array([0.007, 0.003, 1.95])
PARAM_INDEX = 2  # K_uu

def sweep_bifurcation(baseline_params, baseline_ss, param_idx,
                     sweep_range=(0.80, 1.20), n_points=300):
    baseline_val = baseline_params[param_idx]
    sweep_factors = np.linspace(sweep_range[0], sweep_range[1], n_points)
    branch_A = []
    branch_B = []
    for factor in sweep_factors:
        p = baseline_params.copy()
        p[param_idx] = baseline_val * factor
        
        ss_A = fsolve(ode_system, baseline_ss, args=(p,))
        if np.max(np.abs(ode_system(ss_A, p))) < 1e-8 and np.all(ss_A > 0):
            if np.max(np.abs(ss_A - baseline_ss) / baseline_ss) < 0.5:
                branch_A.append((factor, ss_A[2]))
        
        ss_B = fsolve(ode_system, ALT_STATE_SEED, args=(p,))
        if np.max(np.abs(ode_system(ss_B, p))) < 1e-8 and np.all(ss_B > 0):
            if np.max(np.abs(ss_B - baseline_ss) / baseline_ss) > 0.5:
                branch_B.append((factor, ss_B[2]))
    
    return np.array(branch_A), np.array(branch_B)


# Run sweeps for both
branch_A_13, branch_B_13 = sweep_bifurcation(params_13, ss_13, PARAM_INDEX)
branch_A_2,  branch_B_2  = sweep_bifurcation(params_2,  ss_2,  PARAM_INDEX)


def plot_panel(ax, baseline_params, baseline_ss, branch_A, branch_B,
               title, perturbation_label_pos='top'):
    baseline_val = baseline_params[PARAM_INDEX]
    
    x_min = baseline_val * 0.78
    x_max = baseline_val * 1.22
    
    y_vals = []
    if len(branch_A) > 0:
        y_vals.extend(branch_A[:, 1].tolist())
    if len(branch_B) > 0:
        y_vals.extend(branch_B[:, 1].tolist())
    y_vals.append(baseline_ss[2])
    
    y_min = -0.05
    y_max = max(y_vals) * 1.15 if y_vals else 2.2
    
    # ===== FIX: Only shade if there are GAPS within the swept range =====
    # A real bifurcation means Branch A exists for part of the range but disappears.
    # If Branch A spans nearly the full sweep, there's no bifurcation — no shading.
    if len(branch_A) > 0:
        branch_A_x = branch_A[:, 0] * baseline_val
        a_left = branch_A_x.min()
        a_right = branch_A_x.max()
        sweep_width = x_max - x_min
        
        # Only mark as "bifurcation crossed" if Branch A leaves a significant gap
        # within the swept range (more than 5% of sweep width on either side)
        bifurcation_threshold = 0.05 * sweep_width
        
        has_left_bifurcation = (a_left - x_min) > bifurcation_threshold
        has_right_bifurcation = (x_max - a_right) > bifurcation_threshold
        
        if has_left_bifurcation:
            ax.axvspan(x_min, a_left, alpha=0.18, color='red', zorder=0)
            # Star marker at bifurcation point
            ax.plot(a_left, branch_A[np.argmin(branch_A_x), 1],
                    marker='*', color='red', markersize=22,
                    markeredgecolor='black', markeredgewidth=1,
                    zorder=10, linestyle='None', label='Bifurcation point')
        
        if has_right_bifurcation:
            ax.axvspan(a_right, x_max, alpha=0.18, color='red', zorder=0)
            # Star marker
            label_for_star = None if has_left_bifurcation else 'Bifurcation point'
            ax.plot(a_right, branch_A[np.argmax(branch_A_x), 1],
                    marker='*', color='red', markersize=22,
                    markeredgecolor='black', markeredgewidth=1,
                    zorder=10, linestyle='None', label=label_for_star)
    
    # ===== Plot the branches =====
    if len(branch_A) > 0:
        ax.plot(branch_A[:, 0] * baseline_val, branch_A[:, 1], '-',
                color='darkblue', linewidth=3.5,
                label='Branch A (baseline state)', zorder=5)
    
    if len(branch_B) > 0:
        ax.plot(branch_B[:, 0] * baseline_val, branch_B[:, 1], '-',
                color='darkred', linewidth=3.5,
                label='Branch B (alternative state)', zorder=5)
    
    # ===== Mark baseline =====
    ax.axvline(x=baseline_val, color='black', linestyle=':', linewidth=1.5,
               alpha=0.7, zorder=3)
    ax.plot(baseline_val, baseline_ss[2], 'o', color='black', markersize=11,
            zorder=11, label='Baseline (working state)')
    
    # ===== Mark ±10% perturbation positions =====
    for sign, label in [(+1, '+10%'), (-1, '−10%')]:
        x_pert = baseline_val * (1 + sign * 0.10)
        ax.axvline(x=x_pert, color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.6, zorder=2)
        ax.text(x_pert, y_min + (y_max - y_min) * 0.03, label,
                fontsize=10, color='dimgray', ha='center', va='bottom',
                fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('K_uu (perturbed)', fontsize=12)
    ax.set_ylabel('Steady-state w concentration (w*)', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)


# ===== Build the figure =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

plot_panel(ax1, params_13, ss_13, branch_A_13, branch_B_13,
           'Config 13 (robust)\nNo bifurcation: ±10% perturbation stays on Branch A')

plot_panel(ax2, params_2, ss_2, branch_A_2, branch_B_2,
           'Config 2 (fragile)\nBifurcation: ±10% perturbation forces jump to Branch B')

# Add a single combined legend below both panels
handles, labels = ax2.get_legend_handles_labels()
# Add the red shading explanation
from matplotlib.patches import Patch
handles.append(Patch(facecolor='red', alpha=0.18, label='Branch A absent (bifurcation crossed)'))
labels.append('Branch A absent (bifurcation crossed)')

fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

# Overall title
fig.suptitle('Bifurcation Diagram: Steady-State w* vs K_uu',
             fontsize=15, y=1.00)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig('fig4_bifurcation_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: fig4_bifurcation_comparison.png")
plt.close()












# ========================================================================================================================================================
#                                                              FIGURE 5: ROBUSTNESS VS CV — COMPARISON
# ========================================================================================================================================================

# Build a list of configs to plot. Each entry is one line on the plot.
# Easy to extend later when you add 1754 data.

robustness_curves = [
    {
        'label': '#3954 robust (config 13)',
        'CV': cv13['CV'],
        'robustness': cv13['robustness'],
        'color': 'darkblue',
        'marker': 'o',
    },
    {
        'label': '#3954 fragile (config 2)',
        'CV': cv2['CV'],
        'robustness': cv2['robustness'],
        'color': 'crimson',
        'marker': 's',
    },
    # When you have 1754 data, just uncomment and fill in:
    # {
    #     'label': '#1754 robust (config X)',
    #     'CV': cv1754_robust['CV'],
    #     'robustness': cv1754_robust['robustness'],
    #     'color': 'darkgreen',
    #     'marker': '^',
    # },
    # {
    #     'label': '#1754 fragile (config Y)',
    #     'CV': cv1754_fragile['CV'],
    #     'robustness': cv1754_fragile['robustness'],
    #     'color': 'darkorange',
    #     'marker': 'D',
    # },
]


fig, ax = plt.subplots(figsize=(10, 6))

# Plot each robustness curve
for curve in robustness_curves:
    ax.plot(curve['CV'], curve['robustness'],
            marker=curve['marker'], color=curve['color'],
            linewidth=2.5, markersize=9,
            label=curve['label'], zorder=3)

# Reference line at 50% robustness
ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
           label='50% threshold', zorder=2)

# Reference line at 0% (full collapse)
ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, zorder=1)

# Labels and styling
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Robustness (% of trials with Turing instability)', fontsize=13)
ax.set_title('Robustness to Parameter Heterogeneity\n'
             'Fraction of trials maintaining max Re(λ) > 0',
             fontsize=13, pad=15)

ax.set_xlim(-0.01, 0.42)
ax.set_ylim(-5, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, alpha=0.3)

ax.legend(fontsize=11, loc='center right', framealpha=0.95)

plt.tight_layout()
plt.savefig('fig5_robustness_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved: fig5_robustness_vs_cv.png")
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

