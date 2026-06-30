# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from heterogenous_ring_3954 import (compute_jacobian, find_steady_state, build_ring_jacobian_heterogeneous)

# CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
# CONFIG_ID = 43
# N_RING = 30
# N_TRIALS = 20
# CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
# SEED = 42

# M_VALUES = np.arange(0, N_RING // 2 + 1)
# K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

# df = pd.read_csv(CSV_PATH)
# type_i = df[df['classification'] == 'Type-I']
# row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0]

# baseline_params = np.array([
#     row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
#     row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
#     row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
# ])
# dU, dV, dW = row['dU'], row['dV'], row['dW']
# hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}


# def find_dominant_k(eigenvector, N):
#     reshaped = eigenvector.reshape((N, 3))
#     fft_per_species = np.abs(np.fft.fft(reshaped, axis=0))
#     fft_mag = np.sum(fft_per_species, axis=1)
#     relevant_magnitudes = fft_mag[:N // 2 + 1]
#     return int(np.argmax(relevant_magnitudes))


# def get_all_eigenvalues_with_km(baseline_params, hopping, N, CV):
#     """Return list of (k_m, Re(λ)) tuples — all 90 eigenvalues with their assigned k_m."""
#     J_ring, _, _ = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
#     if J_ring is None:
#         return None
    
#     eigenvalues, eigenvectors = np.linalg.eig(J_ring)
#     real_parts = np.real(eigenvalues)
    
#     points = []
#     for j in range(len(eigenvalues)):
#         m = find_dominant_k(eigenvectors[:, j], N)
#         k = K_DISCRETE[m]
#         points.append((k, real_parts[j]))
#     return points


# np.random.seed(SEED)

# # 🛠️ FIXED: Canvas size shifted to (1, 5) to safely fit all 5 CV values
# fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)

# for ax_idx, (ax, CV) in enumerate(zip(axes, CV_VALUES)):
    
#     # 🛠️ FIXED: Baseline CV=0 needs only 1 trial since there is no variance
#     current_trials = 1 if CV == 0.0 else N_TRIALS
#     colors = plt.cm.viridis(np.linspace(0.2, 0.8, current_trials))
    
#     trials_successful = 0
#     attempts = 0
#     while trials_successful < current_trials and attempts < current_trials * 10:
#         attempts += 1
#         points = get_all_eigenvalues_with_km(baseline_params, hopping, N_RING, CV)
#         if points is None:
#             continue
        
#         # Plot all 90 points for this trial
#         ks = [p[0] for p in points]
#         res = [p[1] for p in points]
        
#         # If it's the baseline, color it solid black so it stands out as your control reference
#         plot_color = 'black' if CV == 0.0 else colors[trials_successful]
#         plot_alpha = 1.0 if CV == 0.0 else 0.4
        
#         ax.scatter(ks, res, alpha=plot_alpha, s=25, color=plot_color,
#                    edgecolors='black', linewidths=0.3, zorder=3)
#         trials_successful += 1
    
#     ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
#     ax.set_xlabel('Dominant $k_m$', fontsize=11)
    
#     title_text = f'CV = {CV:.1f}\n(Baseline Control)' if CV == 0.0 else f'CV = {CV:.1f}\n({trials_successful} noisy trials)'
#     ax.set_title(title_text, fontsize=11)
#     ax.grid(alpha=0.3, linestyle='--')
    
#     # 🛠️ FIXED: Sliced with [::2] to show every second discrete wavenumber cleanly
#     ax.set_xticks(K_DISCRETE[::2])  
#     ax.set_xticklabels([f'{k:.2f}' for k in K_DISCRETE[::2]], rotation=45, fontsize=8)

# axes[0].set_ylabel('Real Part of Eigenvalues: Re(λ)', fontsize=12)

# fig.suptitle(
#     f'All Eigenvalues Per Trial — Heterogeneous Ring (N={N_RING}), Config {CONFIG_ID}\n'
#     f'Each dot is one eigenvalue. Shows full structural splitting across the spectrum.',
#     fontsize=14, y=1.02
# )

# plt.tight_layout()
# plt.savefig('eigenvalue_scatter_3954_config43.png', dpi=200, bbox_inches='tight')
# print("Saved: eigenvalue_scatter_3954_config43.png")
# plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heterogenous_ring_3954 import (compute_jacobian, find_steady_state, build_ring_jacobian_heterogeneous)

CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
CONFIG_ID = 43
N_RING = 20
N_TRIALS = 10
NOISY_CV_VALUES = [0.1, 0.2, 0.3, 0.4]  # Split from baseline
SEED = 42

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']
row = type_i[(type_i['config_id'] == CONFIG_ID) & (type_i['param_rank'] == 1)].iloc[0]

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])
dU, dV, dW = row['dU'], row['dV'], row['dW']
hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}


def find_dominant_k(eigenvector, N):
    reshaped = eigenvector.reshape((N, 3))
    fft_per_species = np.abs(np.fft.fft(reshaped, axis=0))
    fft_mag = np.sum(fft_per_species, axis=1)
    relevant_magnitudes = fft_mag[:N // 2 + 1]
    return int(np.argmax(relevant_magnitudes))


def get_all_eigenvalues_with_km(baseline_params, hopping, N, CV):
    """Return list of (k_m, Re(λ)) tuples — all 90 eigenvalues with their assigned k_m."""
    J_ring, _, _ = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
    if J_ring is None:
        return None
    
    eigenvalues, eigenvectors = np.linalg.eig(J_ring)
    real_parts = np.real(eigenvalues)
    
    points = []
    for j in range(len(eigenvalues)):
        m = find_dominant_k(eigenvectors[:, j], N)
        k = K_DISCRETE[m]
        points.append((k, real_parts[j]))
    return points


np.random.seed(SEED)

# 1. PRE-COMPUTE BASELINE CONTROL (CV = 0.0) ONCE
print("Computing baseline control spectrum...")
baseline_points = get_all_eigenvalues_with_km(baseline_params, hopping, N_RING, 0.0)
if baseline_points is not None:
    baseline_ks = [p[0] for p in baseline_points]
    baseline_res = [p[1] for p in baseline_points]
else:
    baseline_ks, baseline_res = [], []

# 2. SETUP CLEAN 4-PANEL CANVAS (Fits all noisy CV values up to 0.4)
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']

for ax_idx, (ax, CV, base_color) in enumerate(zip(axes, NOISY_CV_VALUES, panel_colors)):
    
    # Plot Baseline Control as a solid background layer first
    if len(baseline_ks) > 0:
        ax.scatter(baseline_ks, baseline_res, alpha=0.8, s=35, color='black',
                   edgecolors='black', linewidths=0.5, zorder=4, label='Baseline (CV=0.0)')
    
    # Generate trial-specific gradient colors for the noise spectrum
    #colors = plt.cm.get_cmap('viridis' if base_color=='steelblue' else 'plasma')(np.linspace(0.3, 0.8, N_TRIALS))
    #colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    
    trials_successful = 0
    attempts = 0
    while trials_successful < N_TRIALS and attempts < N_TRIALS * 10:
        attempts += 1
        points = get_all_eigenvalues_with_km(baseline_params, hopping, N_RING, CV)
        if points is None:
            continue
        
        # Plot all 90 points for this specific noisy trial
        ks = [p[0] for p in points]
        res = [p[1] for p in points]
        
        # ax.scatter(ks, res, alpha=0.35, s=20, color=colors, edgecolors='none', zorder=3)
        
        ax.scatter(ks, res, alpha=0.35, s=20, color=panel_colors[ax_idx], edgecolors='black', linewidths=0.3)
        
        trials_successful += 1
    
    # Visual anchors and layout styling
    ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
    ax.set_xlabel('Dominant $k_m$', fontsize=11)
    ax.set_title(f'CV = {CV:.1f} ({trials_successful} trials)', fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Clean X-axis tick display spacing
    ax.set_xticks(K_DISCRETE[::2])  
    ax.set_xticklabels([f'{k:.2f}' for k in K_DISCRETE[::2]], rotation=45, fontsize=8)
    ax.set_ylim(-3, 0.6)
    
    # Put a subtle legend entry on the first panel only to prevent clutter
    if ax_idx == 0:
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

axes[0].set_ylabel('Real Part of Eigenvalues: Re(λ)', fontsize=12)

fig.suptitle(
    f'All Eigenvalues Per Trial with Baseline Overlay — Heterogeneous Ring (N={N_RING}), Config {CONFIG_ID}\n'
    f'Black dots represent baseline control. Coloured spreads represent noisy structural eigenvalue splitting.',
    fontsize=14, y=1.02
)

plt.tight_layout()
plt.savefig('eigenvalue_scatter_3954_config43_overlaid.png', dpi=200, bbox_inches='tight')
print("Saved: eigenvalue_scatter_3954_config43_overlaid.png")
plt.close()
