import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from BiologicalRealism.heterogenous_ring_3954_OG import compute_jacobian, find_steady_state, build_ring_jacobian_heterogeneous

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
CONFIG_IDS = [49, 17]
N_RING = 10
N_TRIALS = 10
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEED = 42

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']

#### HOMOEGENOUS VERSION

# def compute_discrete_dispersion(params, ss, dU, dV, dW, k_discrete):
#     J = compute_jacobian(ss, params)
#     D = np.diag([dU, dV, dW])
#     max_reals = np.zeros(len(k_discrete))
#     for i, k in enumerate(k_discrete):
#         M = J - k**2 * D
#         max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
#     return max_reals

# noisy_cvs = [0.1, 0.2, 0.3, 0.4]
# panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
# fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey='row')

# for row_idx, config_id in enumerate(CONFIG_IDS):
    
#     # Extract row matching loop context safely
#     row_data = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]
    
#     baseline_params = np.array([
#         row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
#         row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
#         row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
#     ])
#     baseline_ss = np.array([row_data['u_star'], row_data['v_star'], row_data['w_star']])
#     dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
    
#     np.random.seed(SEED)
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

#                 # old version where we didint get rid of trials that arent in the negative in the beginning
#                 # dispersion_results[CV].append(disp)
#                 # successful += 1

#                 # NEW 1. k0 = 0 must be stable (negative)
#                 if disp[0] < 0:
#                     dispersion_results[CV].append(disp)
#                     successful += 1

#     baseline_disp = dispersion_results[0.0][0]
#     row_axes = axes_multi[row_idx]
    
#     for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
#         curves = dispersion_results[CV]
        
#         # Plot Baseline
#         ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8, label='Baseline (CV=0.0)' if (row_idx==0 and col_idx==0) else "", zorder=5)
        
#         # Plot Noisy Trials
#         for i, disp in enumerate(curves):
#             label = f'Noisy (CV={CV})' if (i == 0 and row_idx == 0 and col_idx == 0) else ""
#             ax.plot(K_DISCRETE, disp, 'o-', color=color, linewidth=1.2,
#                     markersize=6, alpha=0.4, zorder=3, label=label)
            
#         ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
#         ax.set_xticks(K_DISCRETE)
#         ax.grid(alpha=0.3, linestyle='--')
        
#         # Clean Clean-up: Only show titles over the top row of subplots
#         if row_idx == 0:
#             ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
            
#         # Clean Clean-up: Only show X axis text on the very bottom row panels
#         if row_idx == 1:
#             visible_ticks = K_DISCRETE[::1]
#             visible_labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES[::1], visible_ticks)]
#             ax.set_xticks(visible_ticks)
#             ax.set_xticklabels(visible_labels, rotation=20, ha='right', fontsize=9)
#             ax.set_xlabel('Wavenumber $k_m$', fontsize=11)
#             # tick_labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
#             # ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
#             # ax.set_xlabel('Discrete Wavenumbers ($k_m$)', fontsize=11)

#     # Put a clear bold row marker on the left-most panel of each row
#     row_axes[0].set_ylabel(f'Config {config_id}\nMax Re(λ)', fontsize=12)

# fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)

# fig_multi.suptitle(
#     f'Topology 3954 Discrete Ring Dispersion Comparison (N={N_RING} cells)\n'
#     f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]}',
#     fontsize=14, y=0.95
# )

# legend_handles = [
#     mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
#     mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
# ]

# fig_multi.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)

# plt.savefig('3954_homogeneous_dispersion_comparison.png', dpi=200, bbox_inches='tight')
# plt.close()




#### HETEROGENOUS VERSION

# def find_dominant_k(eigenvector, N):
#     #NEW
#     reshaped = eigenvector.reshape((N, 3))
#     # FFT each species separately, take magnitudes, sum across species
#     fft_per_species = np.abs(np.fft.fft(reshaped, axis=0))  # shape (N, 3)
#     fft_mag = np.sum(fft_per_species, axis=1)  # combined power per spatial frequency, length N

#     relevant_magnitudes = fft_mag[:N // 2 + 1]
#     return int(np.argmax(relevant_magnitudes))

def find_dominant_k(eigenvector, N):
    reshaped = eigenvector.reshape((N, 3))          # cell-major: confirmed correct
    power = np.abs(np.fft.fft(reshaped, axis=0))**2 # (N, 3), power per species
    total = power.sum(axis=1)                        # combine species, length N
    folded = np.zeros(N // 2 + 1)                    # bins m = 0 .. N/2
    for f in range(N):
        folded[min(f, N - f)] += total[f]
    return int(np.argmax(folded))

def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV, k_discrete):
    # Pass diffusion rates packaged as expected by your module (usually [dU, dV, dW])
    J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
    
    if J_ring is None:
        return None
    
    eigenvalues, eigenvectors = np.linalg.eig(J_ring)
    real_parts = np.real(eigenvalues)

    # NEW TO KINDA FILTER AND NOT GET THOSE WEIRD VALUES...
    # mask = real_parts > -5
    # eigenvalues = eigenvalues[mask]
    # eigenvectors = eigenvectors[:, mask]
    # real_parts = real_parts[mask]

    print(f"Trial: 90 eigenvalues sorted (top 20): {np.sort(real_parts)[::-1][:20]}")
    print(f"Median: {np.median(real_parts):.2f}, " f"min: {real_parts.min():.2f}, max: {real_parts.max():.2f}")
    
    # Initialize bins for each discrete wavenumber m
    max_re_per_km = np.full(len(k_discrete), -np.inf)
    
    for j in range(len(eigenvalues)):
        eigvec = eigenvectors[:, j]
        m = find_dominant_k(eigvec, N)
        if real_parts[j] > max_re_per_km[m]:
            max_re_per_km[m] = real_parts[j]
            
    # Replace unassigned structural modes (-inf) safely with NaN
    max_re_per_km = np.where(np.isinf(max_re_per_km), np.nan, max_re_per_km)
    return max_re_per_km


# Setup Plotting Canvas
noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey='row')

# MAIN EXECUTION LOOP
for row_idx, config_id in enumerate(CONFIG_IDS):
    
    row_data = type_i[(type_i['config_id'] == config_id) & (type_i['param_rank'] == 1)].iloc[0]
    
    baseline_params = np.array([
        row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
        row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
        row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
    ])
    
    # Bundle the diffusion coefficients into your 'hopping' parameter array
    dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
    hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}
    
    # Enforce random state seeding per configuration loop for reproducibility
    np.random.seed(SEED)
    dispersion_results = {CV: [] for CV in CV_VALUES}
    
    for CV in CV_VALUES:
        if CV == 0.0:
            # For baseline, cell-to-cell variability is 0
            disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, 0.0, K_DISCRETE)
            if disp is not None:
                dispersion_results[CV].append(disp)
        else:
            successful = 0
            attempts = 0
            max_attempts = N_TRIALS * 10  # Increased since heterogeneous systems fail to solve more often
            
            while successful < N_TRIALS and attempts < max_attempts:
                attempts += 1
                
                # Compute heterogeneous ring dispersion
                disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, CV, K_DISCRETE)
                
                #if disp is None or np.isnan(disp[0]):
                if disp is None or np.all(np.isnan(disp)):
                    continue
                
                # NEW version Check condition: k0 (index 0) must be stable (negative real parts)
                # if disp[0] < 0:
                #     dispersion_results[CV].append(disp)
                #     successful += 1
                
                # ORIGINAL VERSION No k=0 check — heterogeneous rings can have positive m=0 even without diffusion
                dispersion_results[CV].append(disp)
                successful += 1

    # Extract baseline dispersion profile
    baseline_disp = dispersion_results[0.0][0] if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE))
    row_axes = axes_multi[row_idx]
    
    # Generate Subplots per CV
    for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
        curves = dispersion_results[CV]
        
        # Plot Baseline Reference
        ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8, label='Baseline (CV=0.0)' if (row_idx == 0 and col_idx == 0) else "", zorder=5)  # changed from x_indices bc the spacing was not true 
        
        # Plot Noisy Heterogeneous Trials
        for i, disp in enumerate(curves):
            label = f'Heterogeneous Noisy (CV={CV})' if (i == 0 and row_idx == 0 and col_idx == 0) else ""
            ax.plot(K_DISCRETE, disp, 'o-', color=color, linewidth=1.2,
                    markersize=6, alpha=0.4, zorder=3, label=label)
            
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xticks(K_DISCRETE)
        ax.grid(alpha=0.3, linestyle='--')
        
        if row_idx == 0:
            ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
            
        if row_idx == 1:
            visible_ticks = K_DISCRETE[::1]
            visible_labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES[::1], visible_ticks)]
            ax.set_xticks(visible_ticks)
            ax.set_xticklabels(visible_labels, rotation=30, ha='right', fontsize=9)
            ax.set_xlabel('Wavenumber $k_m$', fontsize=11)

    row_axes[0].set_ylabel(f'Config {config_id}\nHeterogeneous Re(λ)', fontsize=12)

# Figure adjustments and saving
fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)
fig_multi.suptitle(
    f'Topology 3954 Heterogeneous Ring Fourier-projected growth spectrum (N={N_RING} cells)\n'
    f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]}',
    fontsize=14, y=0.95
)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
]

fig_multi.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)

plt.savefig('3954_heterogeneous_dispersion_comparison.png', dpi=200, bbox_inches='tight')
print("Saved as 3954_heterogeneous_dispersion_comparison.png")
plt.close()