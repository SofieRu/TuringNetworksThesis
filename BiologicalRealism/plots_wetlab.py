#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle

from heterogenous_ring_3954_earlyversion import (
    compute_jacobian, 
    find_steady_state, 
    build_ring_jacobian_heterogeneous, 
    fourier_projectors, 
    projected_dispersion, 
    is_turing_ring
)

# Load data structures safely
with open('3954_cv_sweep_wetlab_config40_N10.pkl', 'rb') as f:
    cv_lab_3954_N10 = pickle.load(f)
with open('3954_cv_sweep_wetlab_config40_N20.pkl', 'rb') as f:
    cv_lab_3954_N20 = pickle.load(f)
with open('3954_cv_sweep_wetlab_config40_N30.pkl', 'rb') as f:
    cv_lab_3954_N30 = pickle.load(f)

CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
CONFIG_IDS = [40] 
TARGET_CONFIG = CONFIG_IDS[0] # Extracted integer for cleaner string printing
N_RING = 20
N_TRIALS = 20
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEED = 42

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
PROJECTORS = fourier_projectors(N_RING)

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']

def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV):
    J_ring, steady_states, params_list, _ = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
    if J_ring is None:
        return None
    return projected_dispersion(J_ring, PROJECTORS)

# ======================================================================
# PLOT 1: HETEROGENEOUS RING DISPERSION (SINGLE ROW)
# ======================================================================
noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

fig_multi, axes_multi = plt.subplots(1, 4, figsize=(14, 4.5), sharex=True, sharey=True)

for row_idx, config_id in enumerate(CONFIG_IDS):
    row_data = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]
    
    baseline_params = np.array([
        row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
        row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
        row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
    ])
    
    dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
    hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}
    
    np.random.seed(SEED)
    dispersion_results = {CV: [] for CV in CV_VALUES}
    turing_flags = {CV: [] for CV in CV_VALUES}
    
    for CV in CV_VALUES:
        if CV == 0.0:
            disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, 0.0)
            if disp is not None:
                dispersion_results[CV].append(disp)
                turing_flags[CV].append(is_turing_ring(disp))
        else:
            successful = 0
            attempts = 0
            max_attempts = N_TRIALS * 100
            while successful < N_TRIALS and attempts < max_attempts:
                attempts += 1
                disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, CV)
                if disp is None or np.isnan(disp[0]):
                    continue
                dispersion_results[CV].append(disp)
                turing_flags[CV].append(is_turing_ring(disp))
                successful += 1
                
    baseline_disp = (dispersion_results[0.0][0] if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE)))
    
    row_axes = axes_multi 
    
    for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
        curves = dispersion_results[CV]
        flags = turing_flags[CV]
        
        ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8, 
                label='Baseline (CV=0.0)' if col_idx == 0 else "", zorder=5)
        
        for i, (disp, is_t) in enumerate(zip(curves, flags)):
            c = color
            ax.plot(K_DISCRETE, disp, 'o-', color=c, linewidth=1.2, markersize=5, alpha=0.4, zorder=3)
            
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        chosen_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10]
        filtered_ticks = [K_DISCRETE[i] for i in chosen_indices]
        ax.grid(alpha=0.3, linestyle='--')
        
        ax.set_title(f'CV = {CV:.2f}', fontsize=12)
        
        labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
        filtered_labels = [labels[i] for i in chosen_indices]
        ax.set_xticks(filtered_ticks)
        ax.set_xticklabels(filtered_labels, rotation=40, ha='right', fontsize=10)
        ax.set_xlabel("Wavenumber $k_m$", fontsize=12)
        
    # FIXED: row_axes is 1D, indexing row_axes[0] here causes layout issues/errors
    row_axes[0].set_ylabel(f'Config {config_id}\nMax Re(λ)', fontsize=12)

fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.78, bottom=0.28, wspace=0.04)
fig_multi.suptitle(
    f'Topology 3954 Heterogeneous Ring Dispersion (Fourier-projected, N={N_RING} cells)\n'
    f'Robust Config {TARGET_CONFIG} Only', fontsize=14, y=0.97)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
]
fig_multi.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, fontsize=11)

plt.savefig('3954_heterogeneous_dispersion_single_config.png', dpi=200, bbox_inches='tight')
print("Saved as 3954_heterogeneous_dispersion_single_config.png")
plt.close()
