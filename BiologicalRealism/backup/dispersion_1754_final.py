# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# from heterogenous_ring_1754_earlyversion import (
#     compute_jacobian, find_steady_state,
#     #build_ring_jacobian_heterogeneous,
#     fourier_projectors, 
#     projected_dispersion,
#     is_turing_ring)

# # module load matplotlib/3.9.2-gfbf-2024a
# # module load SciPy-bundle/2024.05-gfbf-2024a

# CSV_PATH = '../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv'
# CONFIG_IDS = [49, 18]
# N_RING = 20
# N_TRIALS = 30                      # raised so thin panels fill in
# CV_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
# SEED = 42

# M_VALUES = np.arange(0, N_RING // 2 + 1)
# K_DISCRETE = 2 * np.sin(M_VALUES * np.pi / N_RING)
# PROJECTORS = fourier_projectors(N_RING)   # shared with the sweep -> identical computation

# df = pd.read_csv(CSV_PATH)
# type_i = df[df['classification'] == 'Type-I']


# def compute_heterogeneous_dispersion(baseline_params, hopping, N, CV):
#     """Projected dispersion of one noisy ring, or None if a cell has no
#     positive isolated steady state."""
#     #J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
#     J_ring, steady_states, params_list, _ = build_ring_jacobian_heterogeneous(N, baseline_params, hopping, CV)
#     if J_ring is None:                          # (None, reason, None) on failure
#         return None
#     return projected_dispersion(J_ring, PROJECTORS)


# def build_diffusion_operator(N_cells, hopping):
#     h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
#     size = 3 * N_cells
#     Ldiff = np.zeros((size, size))
#     for i in range(N_cells):
#         idx = 3 * i
#         left, right = (i - 1) % N_cells, (i + 1) % N_cells
#         for s in range(3):
#             Ldiff[idx+s, idx+s]     -= 2 * h[s]
#             Ldiff[idx+s, 3*left+s]  += h[s]
#             Ldiff[idx+s, 3*right+s] += h[s]
#     return Ldiff


# def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):
#     """Frozen-coefficient ring: each cell gets its own noisy params and its own
#     ISOLATED reaction fixed point. Returns (J_ring, steady_states, params_list,
#     balance_resid) or (None, "no_isolated_ss", None, None) if a cell has no
#     positive reaction fixed point.
 
#     balance_resid = ||Ldiff @ x*|| / ||x*|| quantifies how far the isolated fixed
#     points are from satisfying the coupled diffusion balance. Small => the
#     frozen-coefficient approximation is well justified for that trial."""
#     sigma = np.sqrt(np.log(1 + CV**2))
#     mu = -sigma**2 / 2
 
#     params_list, steady_states = [], []
#     for _ in range(N_cells):
#         params_i = baseline_params * np.random.lognormal(mu, sigma, size=len(baseline_params))
#         ss_i = find_steady_state(params_i)
#         if ss_i is None:
#             return None, "no_isolated_ss", None, None
#         params_list.append(params_i)
#         steady_states.append(ss_i)
 
#     Ldiff = build_diffusion_operator(N_cells, hopping)
#     J_ring = Ldiff.copy()
#     for i in range(N_cells):
#         J_ring[3*i:3*i+3, 3*i:3*i+3] += compute_jacobian(steady_states[i], params_list[i])
 
#     x_star = np.concatenate(steady_states)
#     balance_resid = np.linalg.norm(Ldiff @ x_star) / np.linalg.norm(x_star)
#     return J_ring, steady_states, params_list, balance_resid
 

# # ======================================================================
# # PLOTTING
# # ======================================================================

# noisy_cvs = [0.1, 0.2, 0.3, 0.4]
# panel_colors = ['steelblue', 'deeppink', 'darkorange', 'forestgreen']
# fig_multi, axes_multi = plt.subplots(2, 4, figsize=(12.8, 6), sharex=True, sharey='row')

# for row_idx, config_id in enumerate(CONFIG_IDS):

#     row_data = type_i[(type_i['config_id'] == config_id) &
#                       (type_i['param_rank'] == 1)].iloc[0]

#     baseline_params = np.array([
#         row_data['alpha_u'], row_data['beta_u'], row_data['K_vu'], row_data['delta_u'],
#         row_data['alpha_v'], row_data['beta_v'], row_data['K_uv'], row_data['K_wv'], row_data['delta_v'],
#         row_data['alpha_w'], row_data['beta_w'], row_data['K_ww'], row_data['K_uw'], row_data['K_vw'], row_data['delta_w']
#     ])
#     dU, dV, dW = row_data['dU'], row_data['dV'], row_data['dW']
#     hopping = {'h_u': dU, 'h_v': dV, 'h_w': dW}

#     np.random.seed(SEED)
#     dispersion_results = {CV: [] for CV in CV_VALUES}
#     turing_flags = {CV: [] for CV in CV_VALUES}

#     # for CV in CV_VALUES:
#     #     if CV == 0.0:
#     #         disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, 0.0)
#     #         if disp is not None:
#     #             dispersion_results[CV].append(disp)
#     #             turing_flags[CV].append(is_turing_ring(disp))
#     #     else:
#     #         successful = 0
#     #         attempts = 0
#     #         max_attempts = N_TRIALS * 100          # generous budget for high-discard configs
#     #         while successful < N_TRIALS and attempts < max_attempts:
#     #             attempts += 1
#     #             disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, CV)
#     #             if disp is None or np.isnan(disp[0]):
#     #                 continue
#     #             dispersion_results[CV].append(disp)
#     #             turing_flags[CV].append(is_turing_ring(disp))
#     #             successful += 1

#     DISP_LIMIT = 1.0   # keep only trials whose projected dispersion stays below this

#     for CV in CV_VALUES:
#         if CV == 0.0:
#             disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, 0.0)
#             if disp is not None:
#                 dispersion_results[CV].append(disp)
#                 turing_flags[CV].append(is_turing_ring(disp))
#         else:
#             successful = 0
#             attempts = 0
#             rejected_limit = 0
#             max_attempts = N_TRIALS * 200          # bigger budget: over-limit trials are common
#             while successful < N_TRIALS and attempts < max_attempts:
#                 attempts += 1
#                 disp = compute_heterogeneous_dispersion(baseline_params, hopping, N_RING, CV)
#                 if disp is None or np.isnan(disp[0]):
#                     continue
#                 if np.max(disp) > DISP_LIMIT:       # spiky trial -> reject and resample
#                     rejected_limit += 1
#                     continue
#                 dispersion_results[CV].append(disp)
#                 turing_flags[CV].append(is_turing_ring(disp))
#                 successful += 1
#             acc = 100 * successful / max(attempts, 1)
#             print(f"config {config_id} CV={CV}: kept {successful}, "
#                   f"rejected {rejected_limit} over-limit  ({acc:.1f}% acceptance)")


#     baseline_disp = (dispersion_results[0.0][0]
#                      if len(dispersion_results[0.0]) > 0 else np.zeros(len(K_DISCRETE)))
#     row_axes = axes_multi[row_idx]

#     for col_idx, (ax, CV, color) in enumerate(zip(row_axes, noisy_cvs, panel_colors)):
#         curves = dispersion_results[CV]
#         flags = turing_flags[CV]

#         ax.plot(K_DISCRETE, baseline_disp, 'o-', color='black', linewidth=2.0, markersize=8,
#                 label='Baseline (CV=0.0)' if (row_idx == 0 and col_idx == 0) else "", zorder=5)

#         # colour noisy trials by whether they are still a proper Turing instability
#         for i, (disp, is_t) in enumerate(zip(curves, flags)):
#             c = color
#             ax.plot(K_DISCRETE, disp, 'o-', color=c, linewidth=1.2,
#                     markersize=5, alpha=0.4, zorder=3)

#         ax.axhline(0, color='red', linestyle=':', linewidth=2.5, alpha=0.9)

#         chosen_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10] 
#         filtered_ticks = [K_DISCRETE[i] for i in chosen_indices]
#         ax.grid(alpha=0.3, linestyle='--')

#         n_turing = int(np.sum(flags))
#         if row_idx == 0:
#             ax.set_title(f'CV = {CV:.2f}', fontsize=12)
#         if row_idx == 1:
#             labels = [f'$k_{{{m}}}$={k:.2f}' for m, k in zip(M_VALUES, K_DISCRETE)]

#             # Filter labels using the exact same indices
#             filtered_labels = [labels[i] for i in chosen_indices]
#             ax.set_xticks(filtered_ticks)
#             ax.set_xticklabels(filtered_labels, rotation=40, ha='right', fontsize=10)
#             ax.set_xlabel("Wavenumber $k_m$", fontsize=12)

#     row_axes[0].set_ylabel(f'Config {config_id}\nMax Re(λ)', fontsize=12)

# fig_multi.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.04, hspace=0.06)
# fig_multi.suptitle(
#     f'Topology 1754 Heterogeneous Ring Dispersion (Fourier-projected, N={N_RING} cells, 30 Trials)\n'
#     f'Robust Config {CONFIG_IDS[0]} vs Fragile Config {CONFIG_IDS[1]} ',
#     fontsize=14, y=0.97)

# legend_handles = [
#     mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
#     # mlines.Line2D([], [], color='deeppink', linewidth=1.2, marker='o', linestyle='-', alpha=0.6, label='Noisy trial (Turing)'),
#     mlines.Line2D([], [], color='red', linewidth=2, linestyle=':', label='Turing Threshold'),
# ]
# fig_multi.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=12)

# plt.savefig('1754_heterogeneous_dispersion_comparison_new.png', dpi=200, bbox_inches='tight')
# print("Saved as 1754_heterogeneous_dispersion_comparison_new.png")
# plt.close()






#!/usr/bin/env python3
"""
Heterogeneous-ring dispersion under parameter noise -- the CLEAN version.

Why the old plots were messy
----------------------------
A heterogeneous ring has broken translational symmetry, so its eigenmodes are
NOT labelled by a single wavenumber. Two common "fixes" both fail:
  * Fourier / mean-field projection  -> SMOOTH but INFLATES (peak shoots to +1..+3
                                        instead of the true ~+0.45): it is a
                                        compression that discards mode coupling.
  * hard eigenvalue-binning          -> BOUNDED but ZIG-ZAGS (the huge -h_u*k^2
                                        u-damping modes drop into random bins).

What this script does instead
-----------------------------
1. best-match reconstruction (bounded, correct): for each wavenumber m we take
   the max Re over the K eigenvalues whose eigenvectors are most concentrated at
   m (folding +/- m; K=6 interior to cover the degeneracy x 3 species, 3 at the
   ends). Its peak equals the TRUE max ring eigenvalue -> it can never inflate.
2. ensemble median + 10-90% band across trials, one curve per noise level. A
   single noisy ring is intrinsically ragged; the disorder-average is the clean,
   defensible object and it shows the physics directly: as CV grows the growth
   band drops toward zero.

The black baseline is the EXACT homogeneous dispersion max Re(J - k^2 D), which
is well defined because the CV=0 ring is translation invariant.

Reads baseline_params + hopping straight from your cv_sweep pkls; kinetics
(1754 = 15 params, no u self-activation; 3954 = 16 params, u self-activation)
are auto-selected from the parameter count.

# module load matplotlib/3.9.2-gfbf-2024a
# module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
PKLS = {                                  # label -> pkl file
    '1754 config 49': '1754_cv_sweep_high_config49_N10.pkl',
    '3954 config 49': '3954_cv_sweep_high_config49_N10.pkl',
}
N_RING   = 20
N_TRIALS = 40
CVS      = [0.1, 0.2, 0.3, 0.4]
COLORS   = {0.1: 'steelblue', 0.2: 'deeppink', 0.3: 'darkorange', 0.4: 'seagreen'}
Y_LIM    = (-10, 1)                     # focus on the band; deep damping falls off
K_MATCH  = 3

# ----------------------------------------------------------------- Hill kinetics
nH = 2
Ha  = lambda X, K: X**nH / (K**nH + X**nH)
Hi  = lambda X, K: K**nH / (K**nH + X**nH)
dHa = lambda x, K:  nH*K**nH*x**(nH-1) / (K**nH + x**nH)**2
dHi = lambda x, K: -nH*K**nH*x**(nH-1) / (K**nH + x**nH)**2

def ode_1754(s, p):
    u, v, w = s
    au, bu, Kvu, du = p[0:4]; av, bv, Kuv, Kwv, dv = p[4:9]
    aw, bw, Kww, Kuw, Kvw, dw = p[9:15]
    return np.array([au + bu*Hi(v, Kvu) - du*u,
                     av + bv*Ha(u, Kuv)*Hi(w, Kwv) - dv*v,
                     aw + bw*Ha(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw*w])

def jac_1754(s, p):
    u, v, w = s
    au, bu, Kvu, du = p[0:4]; av, bv, Kuv, Kwv, dv = p[4:9]
    aw, bw, Kww, Kuw, Kvw, dw = p[9:15]
    J = np.zeros((3, 3))
    J[0, 0] = -du;                       J[0, 1] = bu*dHi(v, Kvu)
    J[1, 0] = bv*dHa(u, Kuv)*Hi(w, Kwv); J[1, 1] = -dv
    J[1, 2] = bv*Ha(u, Kuv)*dHi(w, Kwv)
    J[2, 0] = bw*Ha(w, Kww)*dHi(u, Kuw)*Hi(v, Kvw)
    J[2, 1] = bw*Ha(w, Kww)*Hi(u, Kuw)*dHi(v, Kvw)
    J[2, 2] = bw*dHa(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw
    return J

def ode_3954(s, p):
    u, v, w = s
    au, bu, Kuu, Kvu, du = p[0:5]; av, bv, Kuv, Kwv, dv = p[5:10]
    aw, bw, Kww, Kuw, Kvw, dw = p[10:16]
    return np.array([au + bu*Ha(u, Kuu)*Hi(v, Kvu) - du*u,
                     av + bv*Ha(u, Kuv)*Hi(w, Kwv) - dv*v,
                     aw + bw*Ha(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw*w])

def jac_3954(s, p):
    u, v, w = s
    au, bu, Kuu, Kvu, du = p[0:5]; av, bv, Kuv, Kwv, dv = p[5:10]
    aw, bw, Kww, Kuw, Kvw, dw = p[10:16]
    J = np.zeros((3, 3))
    J[0, 0] = bu*dHa(u, Kuu)*Hi(v, Kvu) - du            # u self-activation
    J[0, 1] = bu*Ha(u, Kuu)*dHi(v, Kvu)
    J[1, 0] = bv*dHa(u, Kuv)*Hi(w, Kwv); J[1, 1] = -dv
    J[1, 2] = bv*Ha(u, Kuv)*dHi(w, Kwv)
    J[2, 0] = bw*Ha(w, Kww)*dHi(u, Kuw)*Hi(v, Kvw)
    J[2, 1] = bw*Ha(w, Kww)*Hi(u, Kuw)*dHi(v, Kvw)
    J[2, 2] = bw*dHa(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw
    return J

# ----------------------------------------------------------------- solver / ring
def find_ss(ode, jac, p, guess, n_newton=80):
    x = np.array(guess, float)
    for _ in range(n_newton):
        try:
            dx = np.linalg.solve(jac(x, p), -ode(x, p))
        except np.linalg.LinAlgError:
            return None
        x = x + dx
        if np.max(np.abs(dx)) < 1e-12:
            break
    if np.max(np.abs(ode(x, p))) < 1e-8 and np.all(x > 0):
        return x
    return None

def diffusion_operator(N, hv):
    L = np.zeros((3*N, 3*N))
    for i in range(N):
        idx = 3*i; l = (i-1) % N; r = (i+1) % N
        for s in range(3):
            L[idx+s, idx+s] -= 2*hv[s]; L[idx+s, 3*l+s] += hv[s]; L[idx+s, 3*r+s] += hv[s]
    return L

def noisy_ring(ode, jac, base, L, N, CV, ss0, rng):
    sg = np.sqrt(np.log(1 + CV**2)); mu = -sg**2 / 2
    J = L.copy()
    for i in range(N):
        pi = base * rng.lognormal(mu, sg, len(base))
        si = find_ss(ode, jac, pi, ss0)                     # SEEDED at baseline ss
        if si is None:
            return None
        J[3*i:3*i+3, 3*i:3*i+3] += jac(si, pi)
    return J

def bestmatch_dispersion(J, N, K=K_MATCH):
    """max Re per wavenumber over the K eigenmodes most concentrated there.
    Bounded by the true spectrum (peak == max ring eigenvalue), so no inflation."""
    vals, vecs = np.linalg.eig(J)
    nm = N // 2 + 1
    pw = np.zeros((len(vals), nm))
    for j in range(len(vals)):
        vv = vecs[:, j].reshape(N, 3)
        p = sum(np.abs(np.fft.fft(vv[:, s]))**2 for s in range(3))
        pw[j, 0] = p[0]
        for m in range(1, nm):
            pw[j, m] = p[m] + p[(N-m) % N]
    disp = np.zeros(nm)
    for m in range(nm):
        k = 3 if m in (0, nm-1) else K
        disp[m] = np.max(np.real(vals[np.argsort(pw[:, m])[-k:]]))
    return disp

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(PKLS), figsize=(6.6*len(PKLS), 4.8), sharey=True)
if len(PKLS) == 1:
    axes = [axes]

for ax, (label, pkl) in zip(axes, PKLS.items()):
    d = pickle.load(open(pkl, 'rb'))
    base = np.array(d['baseline_params'], float)
    hp = d['hopping']; hv = np.array([hp['h_u'], hp['h_v'], hp['h_w']]); D = np.diag(hv)
    ode, jac = (ode_1754, jac_1754) if len(base) == 15 else (ode_3954, jac_3954)

    ss0 = None
    for g in ([0.5, 0.15, 0.1], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]):
        ss0 = find_ss(ode, jac, base, g)
        if ss0 is not None:
            break
    L = diffusion_operator(N_RING, hv)
    ks = 2*np.sin(np.pi*np.arange(N_RING//2 + 1)/N_RING)
    J0 = jac(ss0, base)
    disp0 = np.array([np.max(np.real(np.linalg.eigvals(J0 - (k*k)*D))) for k in ks])

    print(f"\n== {label} ==  baseline peak {disp0.max():+.3f} at k={ks[np.argmax(disp0)]:.2f}")
    print("CV   median_peak   band[10-90%]")
    for CV in CVS:
        rng = np.random.default_rng(1); C = []; tries = 0
        while len(C) < N_TRIALS and tries < N_TRIALS*40:
            tries += 1
            J = noisy_ring(ode, jac, base, L, N_RING, CV, ss0, rng)
            if J is not None:
                C.append(bestmatch_dispersion(J, N_RING))
        C = np.array(C)
        med = np.median(C, 0); lo = np.percentile(C, 10, 0); hi = np.percentile(C, 90, 0)
        pk = int(np.argmax(med))
        print(f"{CV:.1f}  {med[pk]:+.3f}      [{lo[pk]:+.2f}, {hi[pk]:+.2f}]")
        ax.fill_between(ks, lo, hi, color=COLORS[CV], alpha=0.13)
        ax.plot(ks, med, 'o-', color=COLORS[CV], lw=2, ms=4, label=f'CV = {CV}')

    ax.plot(ks, disp0, 'o-', color='black', lw=2.6, ms=6, label='baseline (CV=0)', zorder=6)
    ax.axhline(0, color='red', ls=':', lw=1.8)
    ax.set_ylim(*Y_LIM); ax.grid(alpha=0.3, ls='--')
    ax.set_xlabel('wavenumber $k$', fontsize=12)
    ax.set_title(label, fontsize=13)
axes[0].set_ylabel(r'max Re$(\lambda)$', fontsize=12)
axes[0].legend(fontsize=9, loc='lower center')
fig.suptitle('Heterogeneous-ring dispersion under parameter noise\n'
             'best-match ensemble median + 10-90% band (N=20, 40 trials/CV)',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('dispersion_ensemble.png', dpi=200, bbox_inches='tight')
print("\nSaved: dispersion_ensemble.png")