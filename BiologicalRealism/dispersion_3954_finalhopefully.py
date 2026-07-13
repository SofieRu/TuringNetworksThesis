import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from heterogenous_ring_3954 import (compute_jacobian, find_steady_state,
                                    build_ring_jacobian_heterogeneous,
                                    _fourier_projectors, fourier_projected_dispersion,
                                    is_turing_ring)

# module load matplotlib/3.9.2-gfbf-2024a
# module load SciPy-bundle/2024.05-gfbf-2024a

CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
CONFIG_IDS = [49, 17]
N_RING = 10
N_TRIALS = 30                       # raised so thin panels fill in
CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
SEED = 42

M_VALUES = np.arange(0, N_RING // 2 + 1)
K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
PROJECTORS = _fourier_projectors(N_RING)   # shared with the sweep -> identical computation

df = pd.read_csv(CSV_PATH)
type_i = df[df['classification'] == 'Type-I']


def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV):
    """Projected dispersion of one noisy ring, or None if a cell has no
    positive isolated steady state."""
    J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(
        N, baseline_params, hopping, CV)
    if J_ring is None:                          # (None, reason, None) on failure
        return None
    return fourier_projected_dispersion(J_ring, PROJECTORS)


# ======================================================================
# PLOTTING
# ======================================================================

noisy_cvs = [0.1, 0.2, 0.3, 0.4]
panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey='row')

for row_idx, config_id in enumerate(CONFIG_IDS):

    row_data = type_i[(type_i['config_id'] == config_id) &
                      (type_i['param_rank'] == 1)].iloc[0]

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
            max_attempts = N_TRIALS * 100          # generous budget for high-discard configs
            while successful < N_TRIALS and attempts < max_attempts:
                attempts += 1
                disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, CV)
                if disp is None or np.isnan(disp[0]):
                    continue
                dispersion_results[CV].append(disp)
                turing_flags[CV].append(is_turing_ring(disp))
                successful += 1

    baseline_disp = (dispersion_results[0.0][0]
                     if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE)))
    row_axes = axes_multi[row_idx]

    for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
        curves = dispersion_results[CV]
        flags = turing_flags[CV]

        ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8,
                label='Baseline (CV=0.0)' if (row_idx == 0 and col_idx == 0) else "", zorder=5)

        # colour noisy trials by whether they are still a proper Turing instability
        for i, (disp, is_t) in enumerate(zip(curves, flags)):
            c = color if is_t else '0.6'
            ax.plot(K_DISCRETE, disp, 'o-', color=c, linewidth=1.2,
                    markersize=5, alpha=0.4, zorder=3)

        ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xticks(K_DISCRETE)
        ax.grid(alpha=0.3, linestyle='--')

        n_turing = int(np.sum(flags))
        if row_idx == 0:
            ax.set_title(f'CV = {CV:.2f}  ({n_turing}/{len(curves)} Turing)', fontsize=12)
        if row_idx == 1:
            labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
            ax.set_xticks(K_DISCRETE)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            ax.set_xlabel('Wavenumber $k_m$', fontsize=11)

    row_axes[0].set_ylabel(f'Config {config_id}\nHeterogeneous Re(λ)', fontsize=12)

fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)
fig_multi.suptitle(
    f'Topology 3954 Heterogeneous Ring Dispersion (Fourier-projected, N={N_RING} cells)\n'
    f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]}  '
    f'(coloured = proper Turing, grey = lost Turing)',
    fontsize=14, y=0.97)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='deeppink', linewidth=1.2, marker='o', linestyle='-', alpha=0.6, label='Noisy trial (Turing)'),
    mlines.Line2D([], [], color='0.6', linewidth=1.2, marker='o', linestyle='-', alpha=0.6, label='Noisy trial (not Turing)'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
]
fig_multi.legend(handles=legend_handles, loc='lower center',
                 bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=11)

plt.savefig('3954_heterogeneous_dispersion_comparison.png', dpi=200, bbox_inches='tight')
print("Saved as 3954_heterogeneous_dispersion_comparison.png")
plt.close()
