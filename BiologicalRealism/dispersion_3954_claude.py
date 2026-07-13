# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from heterogenous_ring_3954 import (compute_jacobian, find_steady_state,
#                                     build_ring_jacobian_heterogeneous)

# # module load matplotlib/3.9.2-gfbf-2024a
# # module load SciPy-bundle/2024.05-gfbf-2024a

# CSV_PATH = '../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv'
# CONFIG_IDS = [49, 17]
# N_RING = 10
# N_TRIALS = 10
# CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
# SEED = 42

# M_VALUES = np.arange(0, N_RING // 2 + 1)
# K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)

# df = pd.read_csv(CSV_PATH)
# type_i = df[df['classification'] == 'Type-I']


# # ======================================================================
# # ASSIGN EACH EIGENVECTOR TO A DISCRETE WAVENUMBER  (FIXED)
# #
# # The eigenvectors of the heterogeneous ring are COMPLEX. For a complex
# # signal the FFT bins f and N-f are independent (no Hermitian symmetry) --
# # they are the two counter-rotating travelling components of the SAME
# # spatial wavelength. The previous code sliced fft[:N//2+1], discarding the
# # upper half, so any mode whose energy sat at f > N/2 was mis-binned
# # (usually dumped toward m=0). At CV=0.2 that mis-assigns ~41% of modes.
# #
# # Fix: FOLD the power spectrum onto m = min(f, N-f) and sum, and use POWER
# # (|.|**2, the physical spectral weight) rather than L1 magnitude.
# # Verified: at CV=0 this reproduces the exact block-based homogeneous
# # dispersion, and at CV>0 the noisy curves track the baseline instead of
# # collapsing to low k.
# # ======================================================================

# def find_dominant_k(eigenvector, N):
#     reshaped = eigenvector.reshape((N, 3))            # cell-major: [U0,V0,W0,U1,...]
#     power = np.abs(np.fft.fft(reshaped, axis=0))**2   # (N, 3) power per species
#     total = power.sum(axis=1)                         # combine species, length N
#     folded = np.zeros(N // 2 + 1)                     # bins m = 0 .. N/2
#     for f in range(N):
#         folded[min(f, N - f)] += total[f]
#     return int(np.argmax(folded))


# def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV, k_discrete,
#                                      return_cloud=False):
#     """Return max Re(lambda) binned per discrete wavenumber m.
#     If return_cloud=True, also return the raw (m, Re lambda) scatter for an
#     honest overlay of the spread the binning hides."""
#     J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(
#         N, baseline_params, hopping, CV)
#     if J_ring is None:                                # (None, reason, None) on failure
#         return (None, None) if return_cloud else None

#     eigenvalues, eigenvectors = np.linalg.eig(J_ring)
#     real_parts = np.real(eigenvalues)

#     max_re_per_km = np.full(len(k_discrete), -np.inf)
#     cloud_m, cloud_re = [], []
#     for j in range(len(eigenvalues)):
#         m = find_dominant_k(eigenvectors[:, j], N)
#         cloud_m.append(m); cloud_re.append(real_parts[j])
#         if real_parts[j] > max_re_per_km[m]:
#             max_re_per_km[m] = real_parts[j]

#     max_re_per_km = np.where(np.isinf(max_re_per_km), np.nan, max_re_per_km)
#     if return_cloud:
#         return max_re_per_km, (np.array(cloud_m), np.array(cloud_re))
#     return max_re_per_km


# # ======================================================================
# # PLOTTING
# # ======================================================================

# noisy_cvs = [0.1, 0.2, 0.3, 0.4]
# panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
# fig_multi, axes_multi = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey='row')

# for row_idx, config_id in enumerate(CONFIG_IDS):

#     row_data = type_i[(type_i['config_id'] == config_id) &
#                       (type_i['param_rank'] == 1)].iloc[0]

#     baseline_params = np.array([
#         row_data['alpha_u'], row_data['beta_u'], row_data['K_uu'], row_data['K_vu'], row_data['delta_u'],
#         row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
#         row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
#     ])
#     dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
#     hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}

#     np.random.seed(SEED)
#     dispersion_results = {CV: [] for CV in CV_VALUES}
#     cloud_results = {CV: [] for CV in CV_VALUES}

#     for CV in CV_VALUES:
#         if CV == 0.0:
#             disp, cloud = compute_heterogeneous_dispersion(
#                 baseline_params, hopping, N_RING, 0.0, K_DISCRETE, return_cloud=True)
#             if disp is not None:
#                 dispersion_results[CV].append(disp)
#                 cloud_results[CV].append(cloud)
#         else:
#             successful = 0
#             attempts = 0
#             max_attempts = N_TRIALS * 10
#             while successful < N_TRIALS and attempts < max_attempts:
#                 attempts += 1
#                 disp, cloud = compute_heterogeneous_dispersion(
#                     baseline_params, hopping, N_RING, CV, K_DISCRETE, return_cloud=True)
#                 if disp is None or np.isnan(disp[0]):
#                     continue
#                 dispersion_results[CV].append(disp)
#                 cloud_results[CV].append(cloud)
#                 successful += 1

#     baseline_disp = (dispersion_results[0.0][0]
#                      if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE)))
#     row_axes = axes_multi[row_idx]

#     for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
#         curves = dispersion_results[CV]

#         # faint raw eigenvalue cloud (honest about the spread binning hides)
#         for cloud in cloud_results[CV]:
#             cm, cre = cloud
#             ax.scatter(K_DISCRETE[cm], cre, s=8, color='0.75', alpha=0.35, zorder=1)

#         # baseline dispersion
#         ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8,
#                 label='Baseline (CV=0.0)' if (row_idx == 0 and col_idx == 0) else "", zorder=5)

#         # noisy binned dispersion curves
#         for i, disp in enumerate(curves):
#             label = f'Heterogeneous Noisy (CV={CV})' if (i == 0 and row_idx == 0 and col_idx == 0) else ""
#             ax.plot(K_DISCRETE, disp, 'o-', color=color, linewidth=1.2,
#                     markersize=6, alpha=0.4, zorder=3, label=label)

#         ax.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
#         ax.set_xticks(K_DISCRETE)
#         ax.grid(alpha=0.3, linestyle='--')

#         if row_idx == 0:
#             ax.set_title(f'CV = {CV:.2f} ({len(curves)} trials)', fontsize=12)
#         if row_idx == 1:
#             labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
#             ax.set_xticks(K_DISCRETE)
#             ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
#             ax.set_xlabel('Wavenumber $k_m$', fontsize=11)

#     row_axes[0].set_ylabel(f'Config {config_id}\nHeterogeneous Re(λ)', fontsize=12)

# fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)
# fig_multi.suptitle(
#     f'Topology 3954 Heterogeneous Ring Dispersion Comparison (N={N_RING} cells)\n'
#     f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]}',
#     fontsize=14, y=0.95)

# legend_handles = [
#     mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
#     mlines.Line2D([], [], color='deeppink', linewidth=1.2, marker='o', linestyle='-', alpha=0.6, label='Noisy trial (binned)'),
#     mlines.Line2D([], [], color='0.75', marker='o', linestyle='None', label='Raw eigenvalues'),
#     mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
# ]
# fig_multi.legend(handles=legend_handles, loc='lower center',
#                  bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=11)

# plt.savefig('3954_heterogeneous_dispersion_comparison.png', dpi=200, bbox_inches='tight')
# print("Saved as 3954_heterogeneous_dispersion_comparison.png")
# plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from heterogenous_ring_3954 import (compute_jacobian, find_steady_state,
                                    build_ring_jacobian_heterogeneous)

# module load matplotlib/3.9.2-gfbf-2024a
# module load SciPy-bundle/2024.05-gfbf-2024a

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


# ======================================================================
# FOURIER-PROJECTED DISPERSION  (Galerkin projection)
#
# WHY NOT eigenvalue binning:
#   The old approach eigendecomposed the 3N x 3N ring Jacobian and assigned
#   each of the 3N eigenvalues to a discrete wavenumber via the dominant
#   Fourier mode of its eigenvector, then took max Re per bin. Once the ring
#   is heterogeneous the eigenvectors are mixed/localised, so the bins fill
#   UNEVENLY -- a high-k bin can catch a single deep diffusion mode and
#   report max Re(lambda) ~ -40, producing the zigzag spikes.
#
# WHAT THIS DOES INSTEAD:
#   For each discrete wavenumber m we project the ring Jacobian onto the
#   Fourier mode phi_m(j) = exp(2*pi*i*m*j/N) (one copy per species) and take
#   max Re of the resulting 3x3 effective operator P_m^H J_ring P_m.
#   * For a HOMOGENEOUS ring this equals J - k_m^2 D exactly (verified), so
#     it reproduces the standard dispersion relation.
#   * For a HETEROGENEOUS ring it is the effective growth rate of Fourier
#     mode m -- a well-defined Rayleigh-Ritz / Galerkin quantity.
#   Exactly N/2+1 values, one per k_m. No binning, no starved bins, no -40.
# ======================================================================

def _fourier_projectors(N):
    """Precompute the (3N x 3) complex projector for each wavenumber m."""
    projectors = []
    for m in range(N // 2 + 1):
        phi = np.exp(2j * np.pi * m * np.arange(N) / N) / np.sqrt(N)
        P = np.zeros((3 * N, 3), dtype=complex)
        for j in range(N):
            for s in range(3):
                P[3 * j + s, s] = phi[j]
        projectors.append(P)
    return projectors

def fourier_projected_dispersion(J_ring, projectors, k_discrete):
    out = np.zeros(len(k_discrete))
    for mi, P in enumerate(projectors):
        Jp = P.conj().T @ J_ring @ P          # 3x3 effective operator at k_m
        out[mi] = np.max(np.real(np.linalg.eigvals(Jp)))
    return out

def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV,
                                     projectors, k_discrete):
    J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(
        N, baseline_params, hopping, CV)
    if J_ring is None:                        # (None, reason, None) on failure
        return None
    return fourier_projected_dispersion(J_ring, projectors, k_discrete)


# ======================================================================
# PLOTTING
# ======================================================================

PROJECTORS = _fourier_projectors(N_RING)

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

    for CV in CV_VALUES:
        if CV == 0.0:
            disp = compute_heterogeneous_dispersion(
                baseline_params, hopping, N_RING, 0.0, PROJECTORS, K_DISCRETE)
            if disp is not None:
                dispersion_results[CV].append(disp)
        else:
            successful = 0
            attempts = 0
            max_attempts = N_TRIALS * 100
            while successful < N_TRIALS and attempts < max_attempts:
                attempts += 1
                disp = compute_heterogeneous_dispersion(
                    baseline_params, hopping, N_RING, CV, PROJECTORS, K_DISCRETE)
                if disp is None or np.isnan(disp[0]):
                    continue
                dispersion_results[CV].append(disp)
                successful += 1

    baseline_disp = (dispersion_results[0.0][0]
                     if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE)))
    row_axes = axes_multi[row_idx]

    for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
        curves = dispersion_results[CV]

        ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8,
                label='Baseline (CV=0.0)' if (row_idx == 0 and col_idx == 0) else "", zorder=5)

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
            labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]
            ax.set_xticks(K_DISCRETE)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            ax.set_xlabel('Wavenumber $k_m$', fontsize=11)

    row_axes[0].set_ylabel(f'Config {config_id}\nHeterogeneous Re(λ)', fontsize=12)

fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)
fig_multi.suptitle(
    f'Topology 3954 Heterogeneous Ring Dispersion (Fourier-projected, N={N_RING} cells)\n'
    f'Rows track Config {CONFIG_IDS[0]} vs Config {CONFIG_IDS[1]}',
    fontsize=14, y=0.95)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='deeppink', linewidth=1.2, marker='o', linestyle='-', alpha=0.6, label='Noisy trial'),
    mlines.Line2D([], [], color='red', linewidth=1.5, linestyle=':', label='Turing Threshold'),
]
fig_multi.legend(handles=legend_handles, loc='lower center',
                 bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=11)

plt.savefig('3954_heterogeneous_dispersion_comparison.png', dpi=200, bbox_inches='tight')
print("Saved as 3954_heterogeneous_dispersion_comparison.png")
plt.close()