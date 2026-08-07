# #!/usr/bin/env python3

# ### VERSION 2 

# """
# Heterogeneous ring of Topology-1754 cells (frozen-coefficient + Fourier).

# FIX vs the version that blew up to max Re ~ 10 for 1754:
#   find_steady_state now seeds fsolve at the BASELINE steady state, so each noisy
#   cell tracks the baseline branch instead of random-jumping to a far fixed point
#   (often one at a steep part of a Hill curve, whose local Jacobian has a huge
#   positive eigenvalue -- and max Re over 30 cells then picks that up). 1754 was
#   hit hard because it has no u self-activation to pin u; 3954 already stayed on
#   branch, so this does not change 3954's (good) results.

# Everything else (isolated-fixed-point ring, Fourier m=0/band test) is unchanged.

# # module load SciPy-bundle/2024.05-gfbf-2024a
# """

# import numpy as np
# from scipy.optimize import fsolve
# import pandas as pd
# import pickle

# # ======================================================================
# # REACTION KINETICS  (1754: u has no self-activation)
# # ======================================================================
# n = 2

# def hill_activation(X, K): return X**n / (K**n + X**n)
# def hill_inhibition(X, K): return K**n / (K**n + X**n)
# def dH_act(x, K):          return n * K**n * x**(n-1) / (K**n + x**n)**2
# def dH_inh(x, K):          return -n * K**n * x**(n-1) / (K**n + x**n)**2

# def ode_system(state, params):
#     u, v, w = state
#     alpha_u, beta_u, K_vu, delta_u = params[0:4]
#     alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
#     alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
#     du = alpha_u + beta_u * hill_inhibition(v, K_vu) - delta_u * u
#     dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
#     dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
#     return [du, dv, dw]

# def find_steady_state(params, n_attempts=100, guess=None):
#     """Positive reaction fixed point. If `guess` (the baseline steady state) is
#     given it is tried FIRST, so the perturbed cell stays on the baseline branch
#     instead of jumping to a far root. Random restarts remain as a fallback."""
#     guesses = ([np.asarray(guess, dtype=float)] if guess is not None else [])
#     guesses += [np.random.uniform(0.01, 10.0, 3) for _ in range(n_attempts)]
#     for g in guesses:
#         ss, info, ier, msg = fsolve(ode_system, g, args=(params,), full_output=True)
#         if ier == 1 and np.max(np.abs(ode_system(ss, params))) < 1e-8 and np.all(ss > 0):
#             return ss
#     return None

# def compute_jacobian(state, params):
#     u, v, w = state
#     alpha_u, beta_u, K_vu, delta_u = params[0:4]
#     alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
#     alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
#     J = np.zeros((3, 3))
#     J[0, 0] = -delta_u                                   # 1754: no self-activation
#     J[0, 1] = beta_u * dH_inh(v, K_vu)
#     J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
#     J[1, 1] = -delta_v
#     J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
#     J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
#     J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
#     J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w
#     return J

# # ======================================================================
# # single-cell continuous classification (sanity only)
# # ======================================================================
# def is_turing_shaberi(J, eigs_0, DU, DV, DW):
#     if np.max(np.real(eigs_0)) >= 0:
#         return None
#     D = np.diag([DU, DV, DW]); k_values = np.arange(0.01, 10.01, 0.01)
#     max_reals = np.zeros(len(k_values)); has_complex_unstable = False
#     for i, k in enumerate(k_values):
#         eigs_k = np.linalg.eigvals(J - (k**2) * D)
#         max_reals[i] = np.max(np.real(eigs_k))
#         if max_reals[i] > 0 and np.any(np.abs(np.imag(eigs_k[np.real(eigs_k) > 0])) > 1e-8):
#             has_complex_unstable = True
#     if np.max(max_reals) <= 0: return None
#     if has_complex_unstable:   return 'Hopf'
#     if max_reals[-1] < 0:      return 'Type-I'
#     return 'Filter' if np.argmax(max_reals) >= len(k_values) - 2 else 'Type-II'

# # ======================================================================
# # ring geometry + Fourier projection
# # ======================================================================
# def build_diffusion_operator(N_cells, hopping):
#     h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
#     Ldiff = np.zeros((3 * N_cells, 3 * N_cells))
#     for i in range(N_cells):
#         idx = 3 * i; left, right = (i - 1) % N_cells, (i + 1) % N_cells
#         for s in range(3):
#             Ldiff[idx+s, idx+s]     -= 2 * h[s]
#             Ldiff[idx+s, 3*left+s]  += h[s]
#             Ldiff[idx+s, 3*right+s] += h[s]
#     return Ldiff

# def fourier_projectors(N):
#     projs = []
#     for m in range(N // 2 + 1):
#         phi = np.exp(2j * np.pi * m * np.arange(N) / N) / np.sqrt(N)
#         P = np.zeros((3 * N, 3), dtype=complex)
#         for j in range(N):
#             for s in range(3):
#                 P[3*j+s, s] = phi[j]
#         projs.append(P)
#     return projs

# def projected_dispersion(J_ring, projectors):
#     return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P)))
#                      for P in projectors])

# def is_turing_ring(disp):
#     return (disp[0] < 0) and (np.max(disp[1:]) > 0)

# def k_eff(N):
#     return 2 * np.sin(np.pi * np.arange(N // 2 + 1) / N)

# # ======================================================================
# # ring Jacobians
# # ======================================================================
# def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
#     J_local = compute_jacobian(steady_state, params)
#     J_ring = build_diffusion_operator(N_cells, hopping)
#     for i in range(N_cells):
#         J_ring[3*i:3*i+3, 3*i:3*i+3] += J_local
#     return J_ring

# def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV, baseline_ss):
#     """Frozen-coefficient ring, but each cell's isolated fixed point is found by
#     seeding at baseline_ss (branch tracking) -> no rogue far fixed points."""
#     sigma = np.sqrt(np.log(1 + CV**2)); mu = -sigma**2 / 2
#     params_list, steady_states = [], []
#     for _ in range(N_cells):
#         params_i = baseline_params * np.random.lognormal(mu, sigma, size=len(baseline_params))
#         ss_i = find_steady_state(params_i, guess=baseline_ss)      # <-- the fix
#         if ss_i is None:
#             return None, "no_isolated_ss", None, None
#         params_list.append(params_i); steady_states.append(ss_i)
#     Ldiff = build_diffusion_operator(N_cells, hopping)
#     J_ring = Ldiff.copy()
#     for i in range(N_cells):
#         J_ring[3*i:3*i+3, 3*i:3*i+3] += compute_jacobian(steady_states[i], params_list[i])
#     x_star = np.concatenate(steady_states)
#     balance_resid = np.linalg.norm(Ldiff @ x_star) / np.linalg.norm(x_star)
#     return J_ring, steady_states, params_list, balance_resid

# # ======================================================================
# # MAIN
# # ======================================================================
# if __name__ == "__main__":
#     CONFIG_TO_TEST = 49
#     CONFIG_LABEL   = "high"
#     n_trials       = 500
#     N_cells        = 10

#     df = pd.read_csv('../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv')
#     df = df[df['classification'] == 'Type-I']
#     row = df[(df['config_id'] == CONFIG_TO_TEST) & (df['param_rank'] == 1)].iloc[0]

#     baseline_params = np.array([
#         row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
#         row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
#         row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']])
#     steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
#     hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

#     PROJECTORS = fourier_projectors(N_cells)

#     J = compute_jacobian(steady_state_expected, baseline_params)
#     turing = is_turing_shaberi(J, np.linalg.eigvals(J),
#                                hopping['h_u'], hopping['h_v'], hopping['h_w'])
#     disp0 = projected_dispersion(
#         build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping),
#         PROJECTORS)
#     print("=" * 70)
#     print(f"Continuous classification: {turing}")
#     print(f"Discrete N={N_cells} baseline: m=0 {disp0[0]:+.4f}, "
#           f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
#     print("=" * 70)

#     np.random.seed(42)
#     results_by_cv = []
#     for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
#         band_eig, m0_eig, full_eig = [], [], []
#         turing_count = discarded = fail_m0 = fail_band = 0
#         for _ in range(n_trials):
#             if CV == 0:
#                 J_ring = build_ring_jacobian_homogeneous(
#                     N_cells, steady_state_expected, baseline_params, hopping)
#             else:
#                 J_ring, _, _, _ = build_ring_jacobian_heterogeneous(
#                     N_cells, baseline_params, hopping, CV, steady_state_expected)
#                 if J_ring is None:
#                     discarded += 1; continue
#             disp = projected_dispersion(J_ring, PROJECTORS)
#             m0_eig.append(disp[0]); band_eig.append(np.max(disp[1:]))
#             full_eig.append(np.max(np.real(np.linalg.eigvals(J_ring))))
#             if disp[0] < 0 and np.max(disp[1:]) > 0: turing_count += 1
#             if disp[0] >= 0:        fail_m0 += 1
#             if np.max(disp[1:]) <= 0: fail_band += 1

#         n_valid = n_trials - discarded
#         full_arr = np.array(full_eig)
#         turing_full = int(np.sum(full_arr > 0))            # full-ring instability (original criterion)
#         rob_cond = 100 * turing_count / n_valid if n_valid > 0 else np.nan
#         rob_marg = 100 * turing_count / n_trials
#         rob_full = 100 * turing_full / n_valid if n_valid > 0 else np.nan
#         results_by_cv.append({
#             'CV': CV, 'n_valid': n_valid, 'n_discarded': discarded,
#             'discard_rate': 100 * discarded / n_trials,
#             'mean_m0': np.mean(m0_eig) if m0_eig else np.nan,
#             'mean_band': np.mean(band_eig) if band_eig else np.nan,
#             'mean_full': np.mean(full_eig) if full_eig else np.nan,
#             'turing_count': turing_count, 'fail_m0': fail_m0, 'fail_band': fail_band,
#             'robustness_conditional': rob_cond, 'robustness_marginal': rob_marg,
#             'robustness_full': rob_full,
#             'all_m0': np.array(m0_eig), 'all_band': np.array(band_eig),
#             'all_full': full_arr})                          # <-- plot THIS, not all_band
#         print(f"CV={CV:<5} valid={n_valid}/{n_trials} disc={discarded} "
#               f"({100*discarded/n_trials:.1f}%)")
#         if n_valid > 0:
#             print(f"    m=0 {np.mean(m0_eig):+.4f} | band {np.mean(band_eig):+.4f} | "
#                   f"full-ring max {np.mean(full_eig):+.4f}  (should be O(1), not ~10)")
#             print(f"    robustness {rob_cond:.1f}% cond / {rob_marg:.1f}% marg  "
#                   f"(m0fail {fail_m0}, bandfail {fail_band})")

#     print("\n" + "=" * 92)
#     print(f"{'CV':<6}{'m=0':<11}{'band':<11}{'full':<11}{'valid':<8}{'disc%':<8}"
#           f"{'robust(cond)':<14}{'robust(marg)'}")
#     print("-" * 92)
#     for r in results_by_cv:
#         print(f"{r['CV']:<6.2f}{r['mean_m0']:<+11.4f}{r['mean_band']:<+11.4f}"
#               f"{r['mean_full']:<+11.4f}{r['n_valid']:<8}{r['discard_rate']:<8.1f}"
#               f"{r['robustness_conditional']:<14.1f}{r['robustness_marginal']:.1f}")
#     print("=" * 92)

#     out = f'1754_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
#     with open(out, 'wb') as f:
#         pickle.dump({'results': results_by_cv, 'baseline_params': baseline_params,
#                      'hopping': hopping, 'n_trials': n_trials,
#                      'config_id': CONFIG_TO_TEST, 'config_name': row['config_name']}, f)
#     print(f"Saved -> {out}")






# VERSION 3

#!/usr/bin/env python3
"""
Heterogeneous ring of Topology-1754 cells (frozen-coefficient + Fourier).

FIX vs the version that blew up to max Re ~ 10 for 1754:
  find_steady_state now seeds fsolve at the BASELINE steady state, so each noisy
  cell tracks the baseline branch instead of random-jumping to a far fixed point
  (often one at a steep part of a Hill curve, whose local Jacobian has a huge
  positive eigenvalue -- and max Re over 30 cells then picks that up). 1754 was
  hit hard because it has no u self-activation to pin u; 3954 already stayed on
  branch, so this does not change 3954's (good) results.

Everything else (isolated-fixed-point ring, Fourier m=0/band test) is unchanged.

# module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# ======================================================================
# REACTION KINETICS  (1754: u has no self-activation)
# ======================================================================
n = 2

def hill_activation(X, K): return X**n / (K**n + X**n)
def hill_inhibition(X, K): return K**n / (K**n + X**n)
def dH_act(x, K):          return n * K**n * x**(n-1) / (K**n + x**n)**2
def dH_inh(x, K):          return -n * K**n * x**(n-1) / (K**n + x**n)**2

def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    du = alpha_u + beta_u * hill_inhibition(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    return [du, dv, dw]

def find_steady_state(params, n_attempts=100, guess=None):
    """Positive reaction fixed point. If `guess` (the baseline steady state) is
    given it is tried FIRST, so the perturbed cell stays on the baseline branch
    instead of jumping to a far root. Random restarts remain as a fallback."""
    guesses = ([np.asarray(guess, dtype=float)] if guess is not None else [])
    guesses += [np.random.uniform(0.01, 10.0, 3) for _ in range(n_attempts)]
    for g in guesses:
        ss, info, ier, msg = fsolve(ode_system, g, args=(params,), full_output=True)
        if ier == 1 and np.max(np.abs(ode_system(ss, params))) < 1e-8 and np.all(ss > 0):
            return ss
    return None

def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    J = np.zeros((3, 3))
    J[0, 0] = -delta_u                                   # 1754: no self-activation
    J[0, 1] = beta_u * dH_inh(v, K_vu)
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w
    return J

# ======================================================================
# single-cell continuous classification (sanity only)
# ======================================================================
def is_turing_shaberi(J, eigs_0, DU, DV, DW):
    if np.max(np.real(eigs_0)) >= 0:
        return None
    D = np.diag([DU, DV, DW]); k_values = np.arange(0.01, 10.01, 0.01)
    max_reals = np.zeros(len(k_values)); has_complex_unstable = False
    for i, k in enumerate(k_values):
        eigs_k = np.linalg.eigvals(J - (k**2) * D)
        max_reals[i] = np.max(np.real(eigs_k))
        if max_reals[i] > 0 and np.any(np.abs(np.imag(eigs_k[np.real(eigs_k) > 0])) > 1e-8):
            has_complex_unstable = True
    if np.max(max_reals) <= 0: return None
    if has_complex_unstable:   return 'Hopf'
    if max_reals[-1] < 0:      return 'Type-I'
    return 'Filter' if np.argmax(max_reals) >= len(k_values) - 2 else 'Type-II'

# ======================================================================
# ring geometry + Fourier projection
# ======================================================================
def build_diffusion_operator(N_cells, hopping):
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    Ldiff = np.zeros((3 * N_cells, 3 * N_cells))
    for i in range(N_cells):
        idx = 3 * i; left, right = (i - 1) % N_cells, (i + 1) % N_cells
        for s in range(3):
            Ldiff[idx+s, idx+s]     -= 2 * h[s]
            Ldiff[idx+s, 3*left+s]  += h[s]
            Ldiff[idx+s, 3*right+s] += h[s]
    return Ldiff

def fourier_projectors(N):
    projs = []
    for m in range(N // 2 + 1):
        phi = np.exp(2j * np.pi * m * np.arange(N) / N) / np.sqrt(N)
        P = np.zeros((3 * N, 3), dtype=complex)
        for j in range(N):
            for s in range(3):
                P[3*j+s, s] = phi[j]
        projs.append(P)
    return projs

def projected_dispersion(J_ring, projectors):
    return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P)))
                     for P in projectors])

def is_turing_ring(disp):
    return (disp[0] < 0) and (np.max(disp[1:]) > 0)

def k_eff(N):
    return 2 * np.sin(np.pi * np.arange(N // 2 + 1) / N)

# ======================================================================
# ring Jacobians
# ======================================================================
def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    J_local = compute_jacobian(steady_state, params)
    J_ring = build_diffusion_operator(N_cells, hopping)
    for i in range(N_cells):
        J_ring[3*i:3*i+3, 3*i:3*i+3] += J_local
    return J_ring

def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV, baseline_ss):
    """Frozen-coefficient ring, but each cell's isolated fixed point is found by
    seeding at baseline_ss (branch tracking) -> no rogue far fixed points."""
    sigma = np.sqrt(np.log(1 + CV**2)); mu = -sigma**2 / 2
    params_list, steady_states = [], []
    for _ in range(N_cells):
        params_i = baseline_params * np.random.lognormal(mu, sigma, size=len(baseline_params))
        ss_i = find_steady_state(params_i, guess=baseline_ss)      # <-- the fix
        if ss_i is None:
            return None, "no_isolated_ss", None, None
        params_list.append(params_i); steady_states.append(ss_i)
    Ldiff = build_diffusion_operator(N_cells, hopping)
    J_ring = Ldiff.copy()
    for i in range(N_cells):
        J_ring[3*i:3*i+3, 3*i:3*i+3] += compute_jacobian(steady_states[i], params_list[i])
    x_star = np.concatenate(steady_states)
    balance_resid = np.linalg.norm(Ldiff @ x_star) / np.linalg.norm(x_star)
    return J_ring, steady_states, params_list, balance_resid

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    CONFIG_TO_TEST = 49
    CONFIG_LABEL   = "high"
    n_trials       = 500
    N_cells        = 10

    df = pd.read_csv('../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv')
    df = df[df['classification'] == 'Type-I']
    row = df[(df['config_id'] == CONFIG_TO_TEST) & (df['param_rank'] == 1)].iloc[0]

    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']])
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

    PROJECTORS = fourier_projectors(N_cells)

    J = compute_jacobian(steady_state_expected, baseline_params)
    turing = is_turing_shaberi(J, np.linalg.eigvals(J),
                               hopping['h_u'], hopping['h_v'], hopping['h_w'])
    disp0 = projected_dispersion(
        build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping),
        PROJECTORS)
    print("=" * 70)
    print(f"Continuous classification: {turing}")
    print(f"Discrete N={N_cells} baseline: m=0 {disp0[0]:+.4f}, "
          f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
    print("=" * 70)

    # baseline reaction stability (diffusion off) = the exact "m=0" condition
    react0 = np.max(np.real(np.linalg.eigvals(
        compute_jacobian(steady_state_expected, baseline_params))))

    np.random.seed(42)
    results_by_cv = []
    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        full_eig, react_eig = [], []
        turing_count = discarded = fail_react = fail_stable = 0
        for _ in range(n_trials):
            if CV == 0:
                J_ring = build_ring_jacobian_homogeneous(
                    N_cells, steady_state_expected, baseline_params, hopping)
                reaction_max = react0
            else:
                J_ring, ss_list, p_list, _ = build_ring_jacobian_heterogeneous(
                    N_cells, baseline_params, hopping, CV, steady_state_expected)
                if J_ring is None:
                    discarded += 1; continue
                # reaction stability = block-diagonal (diffusion OFF), max over cells
                reaction_max = max(
                    np.max(np.real(np.linalg.eigvals(compute_jacobian(ss_list[i], p_list[i]))))
                    for i in range(N_cells))
            full_max = np.max(np.real(np.linalg.eigvals(J_ring)))   # diffusion ON
            react_eig.append(reaction_max); full_eig.append(full_max)

            # EXACT TURING CRITERION (no projection):
            #   stable without diffusion (reaction_max < 0) AND unstable with diffusion (full_max > 0)
            if reaction_max < 0 and full_max > 0:
                turing_count += 1
            if reaction_max >= 0: fail_react += 1     # a cell is locally unstable -> not Turing
            if full_max <= 0:     fail_stable += 1     # ring fully stable -> no instability

        n_valid = n_trials - discarded
        rob_cond = 100 * turing_count / n_valid if n_valid > 0 else np.nan
        rob_marg = 100 * turing_count / n_trials
        results_by_cv.append({
            'CV': CV, 'n_valid': n_valid, 'n_discarded': discarded,
            'discard_rate': 100 * discarded / n_trials,
            'mean_reaction': np.mean(react_eig) if react_eig else np.nan,
            'mean_full': np.mean(full_eig) if full_eig else np.nan,
            'turing_count': turing_count, 'fail_reaction': fail_react, 'fail_stable': fail_stable,
            'robustness_conditional': rob_cond, 'robustness_marginal': rob_marg,
            'all_reaction': np.array(react_eig), 'all_full': np.array(full_eig)})  # plot all_full
        print(f"CV={CV:<5} valid={n_valid}/{n_trials} disc={discarded} "
              f"({100*discarded/n_trials:.1f}%)")
        if n_valid > 0:
            print(f"    reaction max {np.mean(react_eig):+.4f} (want <0) | "
                  f"full-ring max {np.mean(full_eig):+.4f} (want >0)")
            print(f"    robustness {rob_cond:.1f}% cond / {rob_marg:.1f}% marg  "
                  f"(reactfail {fail_react}, fullstable {fail_stable})")

    print("\n" + "=" * 92)
    print(f"{'CV':<6}{'reaction':<12}{'full':<12}{'valid':<8}{'disc%':<8}"
          f"{'robust(cond)':<14}{'robust(marg)'}")
    print("-" * 92)
    for r in results_by_cv:
        print(f"{r['CV']:<6.2f}{r['mean_reaction']:<+12.4f}{r['mean_full']:<+12.4f}"
              f"{r['n_valid']:<8}{r['discard_rate']:<8.1f}"
              f"{r['robustness_conditional']:<14.1f}{r['robustness_marginal']:.1f}")
    print("=" * 92)

    out = f'1754_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    with open(out, 'wb') as f:
        pickle.dump({'results': results_by_cv, 'baseline_params': baseline_params,
                     'hopping': hopping, 'n_trials': n_trials,
                     'config_id': CONFIG_TO_TEST, 'config_name': row['config_name']}, f)
    print(f"Saved -> {out}")