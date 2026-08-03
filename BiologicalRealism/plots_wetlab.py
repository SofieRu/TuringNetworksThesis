#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Forces high-quality silent rendering; prevents headless display crashes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# Required cluster environments:
# module load matplotlib/3.9.2-gfbf-2024a
# module load SciPy-bundle/2024.05-gfbf-2024a
# pip install seaborn --user

from heterogenous_ring_3954_earlyversion import (
    build_ring_jacobian_heterogeneous, 
    fourier_projectors, 
    projected_dispersion, 
    is_turing_ring
)

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
TARGET_CONFIG = 40  # Unified target configuration
N_RING = 20
N_TRIALS = 20 
SEED = 42
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
NOISY_CVS = [0.1, 0.2, 0.3, 0.4]
PANEL_COLORS = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
PROJECTORS = fourier_projectors(N_RING)

# ============================================================================
# 2. DATA LOADING & SIMULATION
# ============================================================================
print("Loading wet lab sweep pickle files for Config 40...", flush=True)
with open('3954_cv_sweep_wetlab_config40_N10.pkl', 'rb') as f:
    cv_lab_3954_N10 = pickle.load(f)
with open('3954_cv_sweep_wetlab_config40_N20.pkl', 'rb') as f:
    cv_lab_3954_N20 = pickle.load(f)
with open('3954_cv_sweep_wetlab_config40_N30.pkl', 'rb') as f:
    cv_lab_3954_N30 = pickle.load(f)

print(f"Reading CSV parameters from: {CSV_PATH}", flush=True)
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Could not find CSV at path '{CSV_PATH}'!")
    raise

type_i = df[df['classification'] == 'Type-I']

def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV):
    res = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
    if res is None:
        return None
    return projected_dispersion(res, PROJECTORS)

# Extract and run dispersion simulation strictly for Config 40
print(f"Beginning dispersion simulations for Config {TARGET_CONFIG}...", flush=True)
row_data = type_i[(type_i['config_id'] == TARGET_CONFIG) & (type_i['param_rank'] == 1)].iloc[0]
baseline_params = np.array([
    row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
    row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
    row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
])
hopping = {'h_u': row_data['dU'], 'h_v': row_data['dV'], 'h_w': row_data['dW']}

np.random.seed(SEED)
dispersion_results = {cv: [] for cv in CV_VALUES}
turing_flags = {cv: [] for cv in CV_VALUES}

for cv in CV_VALUES:
    if cv == 0.0:
        disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, 0.0)
        if disp is not None:
            dispersion_results[cv].append(disp)
            turing_flags[cv].append(is_turing_ring(disp))
    else:
        successful, attempts, max_attempts = 0, 0, N_TRIALS * 100
        while successful < N_TRIALS and attempts < max_attempts:
            attempts += 1
            disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, cv)
            if disp is None or np.isnan(disp).any():
                continue
            dispersion_results[cv].append(disp)
            turing_flags[cv].append(is_turing_ring(disp))
            successful += 1
            
baseline_disp = dispersion_results[0.0][0] if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE))

# ============================================================================
# 3. FIGURE LAYOUT & PLOT GENERATION
# ============================================================================
print("Generating multi-panel plot layout using GridSpec...", flush=True)
fig = plt.figure(figsize=(15, 10))

# 3 Rows total: 
# Row 0: 4 Dispersion plots for Config 40
# Row 1: Boxplot (left 2 cols) & Robustness Plot (right 2 cols)
gs = GridSpec(2, 4, height_ratios=[1, 1.4], hspace=0.35, wspace=0.25)

# ----------------------------------------------------------------------------
# SUBPLOT 1: DISPERSION PLOTS (Top Row - 1 Row, 4 Columns)
# ----------------------------------------------------------------------------
for col_idx, cv in enumerate(NOISY_CVS):
    ax = fig.add_subplot(gs[0, col_idx])
    curves = dispersion_results[cv]
    
    # Plot baseline curve (CV = 0)
    ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=1.8, markersize=6, zorder=5)
    
    # Plot simulated trials
    for disp in curves:
        ax.plot(K_DISCRETE, disp, 'o-', color=PANEL_COLORS[col_idx], linewidth=1.0, markersize=4, alpha=0.3, zorder=3)
        
    ax.axhline(0, color='red', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.grid(alpha=0.3, linestyle='--')
    
    chosen_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    filtered_ticks = [K_DISCRETE[i] for i in chosen_indices]
    
    ax.set_title(f'CV = {cv:.2f}', fontsize=10, fontweight='semibold')
    labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
    ax.set_xticks(filtered_ticks)
    ax.set_xticklabels([labels[i] for i in chosen_indices], rotation=40, ha='right', fontsize=8)
    ax.set_xlabel("Wavenumber $k_m$", fontsize=10)
        
    if col_idx == 0:
        ax.set_ylabel(f'Config {TARGET_CONFIG}\nMax Re(λ)', fontsize=10, fontweight='semibold')

fig.text(0.5, 0.95, f'Topology 3954 (Config {TARGET_CONFIG}) Heterogeneous Ring Dispersion (N={N_RING} cells)', 
         ha='center', fontsize=12, fontweight='bold')

# ----------------------------------------------------------------------------
# SUBPLOT 2: BOXPLOT (Bottom Left)
# ----------------------------------------------------------------------------
ax_box = fig.add_subplot(gs[1, 0:2])
bp = ax_box.boxplot(cv_lab_3954_N10['all'], positions=range(len(cv_lab_3954_N10['CV'])), 
                    widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color='black', linewidth=1.5), 
                    flierprops=dict(marker='o', markersize=3, alpha=0.2))

for patch in bp['boxes']:
    patch.set_facecolor('lightskyblue')
    patch.set_alpha(0.8)

ax_box.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Turing threshold (Re(λ)=0)', zorder=10)
ax_box.set_xticks(range(len(cv_lab_3954_N10['CV'])))
ax_box.set_xticklabels([f'{cv:.2f}' for cv in cv_lab_3954_N10['CV']], fontsize=9)
ax_box.set_xlabel('CV (Coefficient of Variation)', fontsize=11)
ax_box.set_ylabel('Max Re(λ)', fontsize=11)
ax_box.set_title(f'Config {TARGET_CONFIG}: Growth Rate Distribution (N=10)', fontsize=11, fontweight='bold', pad=10)
ax_box.legend(fontsize=9, loc='upper right')
ax_box.grid(True, alpha=0.3, axis='y')

# ----------------------------------------------------------------------------
# SUBPLOT 3: ROBUSTNESS LINE PLOT (Bottom Right)
# ----------------------------------------------------------------------------
ax_rob = fig.add_subplot(gs[1, 2:4])
robustness_curves = [
    {'label': 'N=10', 'data': cv_lab_3954_N10, 'linestyle': '-'},
    {'label': 'N=20', 'data': cv_lab_3954_N20, 'linestyle': '--'},
    {'label': 'N=30', 'data': cv_lab_3954_N30, 'linestyle': ':'},
]

ax_rob.axhline(y=50, color='gray', linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
ax_rob.axhline(y=0, color='black', linewidth=0.8, alpha=0.2, zorder=1)

for curve in robustness_curves:
    ax_rob.plot(
        curve['data']['CV'], curve['data']['robustness'], 
        marker='o', color='blue', linestyle=curve['linestyle'], linewidth=2, 
        markersize=6, markeredgecolor='white', markeredgewidth=1.0, 
        label=curve['label'], zorder=3
    )

ax_rob.set_title(f'Config {TARGET_CONFIG}: Robustness Across Ring Sizes', fontsize=11, fontweight='bold', pad=10)
ax_rob.set_xlabel('CV (Coefficient of Variation)', fontsize=11)
ax_rob.set_ylabel('Robustness (% trials with Turing instability)', fontsize=11)
ax_rob.set_xlim(-0.02, 0.42)
ax_rob.set_ylim(-3, 103)
ax_rob.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
ax_rob.legend(fontsize=9, title="Ring Size")

# ============================================================================
# 4. EXPORT & SAVE
# ============================================================================
plt.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.95)

output_filename = '3954_config40_analysis_panel.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Successfully saved consolidated Config 40 figure: {output_filename}", flush=True)
plt.close()
