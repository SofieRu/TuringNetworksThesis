#!/usr/bin/env python3
"""
Heterogeneous ring of Topology-1754 cells: robustness of the Turing instability
to parameter noise.
 
*** EARLY VERSION (Fourier projection + frozen-coefficient) ***
This is the first corrected 1754 script: it classifies with the two-condition
Fourier test (m=0 stable AND some m>0 unstable) and linearises each cell around
its ISOLATED reaction fixed point. It was later superseded because the
frozen-coefficient linearisation is invalid for strong diffusion (h_w=2.0) and
the Fourier projection is unreliable under strong heterogeneity. Kept for
reference / its conditional-vs-marginal robustness reporting. For final results
use the coupled-steady-state + exact-test version instead.
 
WHAT THIS TESTS (two-condition Turing test, per Fourier mode):
    Condition A (uniform mode m=0):     max Re(lambda) < 0  -> stable without structure
    Condition B (some finite mode m>0): max Re(lambda) > 0  -> diffusion-driven pattern
 
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
"""
 
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle
 
# ======================================================================
# REACTION KINETICS
# ======================================================================
 
n = 2
 
def hill_activation(X, K):
    return X**n / (K**n + X**n)
 
def hill_inhibition(X, K):
    return K**n / (K**n + X**n)
 
def dH_act(x, K):
    return n * K**n * x**(n-1) / (K**n + x**n)**2
 
def dH_inh(x, K):
    return -n * K**n * x**(n-1) / (K**n + x**n)**2
 
def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]               # K_uu removed
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    du = alpha_u + beta_u * hill_inhibition(v, K_vu) - delta_u * u          # no H_act(u)
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    return [du, dv, dw]

def find_steady_state(params, n_attempts=100):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        steady_state, info, ier, msg = fsolve(
            ode_system, initial_guess, args=(params,), full_output=True)
        residuals = ode_system(steady_state, params)
        if ier == 1 and np.max(np.abs(residuals)) < 1e-8 and np.all(steady_state > 0):
            return steady_state
    return None
 
def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    J = np.zeros((3, 3))
    # --- row u: CHANGED for 1754 (no self-activation) ---
    J[0, 0] = -delta_u
    J[0, 1] = beta_u * dH_inh(v, K_vu)
    J[0, 2] = 0
    # --- rows v, w: identical to 3954 ---
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w
    return J
 
# ======================================================================
# CONTINUOUS single-cell dispersion classification (sanity only)
# ======================================================================
 
def is_turing_shaberi(J, eigs_0, DU, DV, DW):
    if np.max(np.real(eigs_0)) >= 0:
        return None
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.01)
    max_reals = np.zeros(len(k_values))
    has_complex_unstable = False
    for i, k in enumerate(k_values):
        eigs_k = np.linalg.eigvals(J - (k**2) * D)
        max_reals[i] = np.max(np.real(eigs_k))
        if max_reals[i] > 0:
            unstable = eigs_k[np.real(eigs_k) > 0]
            if np.any(np.abs(np.imag(unstable)) > 1e-8):
                has_complex_unstable = True
    if np.max(max_reals) <= 0:
        return None
    if has_complex_unstable:
        return 'Hopf'
    if max_reals[-1] < 0:
        return 'Type-I'
    return 'Filter' if np.argmax(max_reals) >= len(k_values) - 2 else 'Type-II'
 
# ======================================================================
# RING GEOMETRY:  diffusion operator and Fourier-mode projectors
# ======================================================================
 
def build_diffusion_operator(N_cells, hopping):
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    size = 3 * N_cells
    Ldiff = np.zeros((size, size))
    for i in range(N_cells):
        idx = 3 * i
        left, right = (i - 1) % N_cells, (i + 1) % N_cells
        for s in range(3):
            Ldiff[idx+s, idx+s]     -= 2 * h[s]
            Ldiff[idx+s, 3*left+s]  += h[s]
            Ldiff[idx+s, 3*right+s] += h[s]
    return Ldiff
 
def fourier_projectors(N):
    """One (3N x 3) complex projector per discrete wavenumber m = 0..N/2."""
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
    """max Re(lambda) of P_m^H J_ring P_m for each mode m.
    Homogeneous ring: equals J - k_m^2 D exactly.
    Heterogeneous ring: leading-order growth rate of mode m (approximate)."""
    return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P)))
                     for P in projectors])
 
def is_turing_ring(disp):
    """Two-condition test: uniform mode stable AND some finite mode unstable."""
    return (disp[0] < 0) and (np.max(disp[1:]) > 0)
 
def k_eff(N):
    return 2 * np.sin(np.pi * np.arange(N // 2 + 1) / N)
 
# ======================================================================
# RING JACOBIANS
# ======================================================================
 
def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    J_local = compute_jacobian(steady_state, params)
    J_ring = build_diffusion_operator(N_cells, hopping)
    for i in range(N_cells):
        J_ring[3*i:3*i+3, 3*i:3*i+3] += J_local
    return J_ring
 
def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):
    """Frozen-coefficient ring: each cell gets its own noisy params and its own
    ISOLATED reaction fixed point. Returns (J_ring, steady_states, params_list,
    balance_resid) or (None, "no_isolated_ss", None, None) if a cell has no
    positive reaction fixed point.
 
    balance_resid = ||Ldiff @ x*|| / ||x*|| quantifies how far the isolated fixed
    points are from satisfying the coupled diffusion balance. Small => the
    frozen-coefficient approximation is well justified for that trial."""
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2
 
    params_list, steady_states = [], []
    for _ in range(N_cells):
        params_i = baseline_params * np.random.lognormal(mu, sigma, size=len(baseline_params))
        ss_i = find_steady_state(params_i)
        if ss_i is None:
            return None, "no_isolated_ss", None, None
        params_list.append(params_i)
        steady_states.append(ss_i)
 
    Ldiff = build_diffusion_operator(N_cells, hopping)
    J_ring = Ldiff.copy()
    for i in range(N_cells):
        J_ring[3*i:3*i+3, 3*i:3*i+3] += compute_jacobian(steady_states[i], params_list[i])
 
    x_star = np.concatenate(steady_states)
    balance_resid = np.linalg.norm(Ldiff @ x_star) / np.linalg.norm(x_star)
    return J_ring, steady_states, params_list, balance_resid
 
# ======================================================================
# MAIN: MONTE-CARLO CV SWEEP
# ======================================================================
 

CONFIG_TO_TEST = 49
CONFIG_LABEL   = "high"
n_trials       = 500
N_cells        = 10
 
df_file = pd.read_csv('../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv')
df_params = df_file[df_file['classification'] == 'Type-I']
row = df_params[(df_params['config_id'] == CONFIG_TO_TEST) &
                    (df_params['param_rank'] == 1)].iloc[0]
 
baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])
steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}
 
if __name__ == "__main__":
 
    CONFIG_TO_TEST = 49
    CONFIG_LABEL   = "high"
    n_trials       = 500
    N_cells        = 10
 
    df_file = pd.read_csv('../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv')
    df_params = df_file[df_file['classification'] == 'Type-I']
    row = df_params[(df_params['config_id'] == CONFIG_TO_TEST) &
                    (df_params['param_rank'] == 1)].iloc[0]
 
    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}
 
    PROJECTORS = fourier_projectors(N_cells)
 
    # ---- baseline sanity: is the CV=0 ring actually Turing on the DISCRETE lattice? ----
    J = compute_jacobian(steady_state_expected, baseline_params)
    turing = is_turing_shaberi(J, np.linalg.eigvals(J),
                               hopping['h_u'], hopping['h_v'], hopping['h_w'])
    J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,
                                              baseline_params, hopping)
    disp0 = projected_dispersion(J_ring0, PROJECTORS)
 
    print("=" * 70)
    print(f"Continuous single-cell classification: {turing}")
    print(f"Discrete N={N_cells} ring baseline: m=0 {disp0[0]:+.4f}, "
          f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
    for m, (k, g) in enumerate(zip(k_eff(N_cells), disp0)):
        print(f"  m={m}  k_eff={k:.4f}  Re(lambda)={g:+.5f}  {'UNSTABLE' if g > 0 else ''}")
    if not is_turing_ring(disp0):
        print("WARNING: discrete baseline is NOT Turing -> the sweep measures noise "
              "around a non-Turing point. Increase N_cells or pick another config.")
    print("=" * 70)
 
    # ---- CV sweep ----
    np.random.seed(42)
    results_by_cv = []
 
    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        band_eig, m0_eig, proj_err = [], [], []
        turing_count = discarded = fail_m0 = fail_band = 0
 
        for _ in range(n_trials):
            if CV == 0:
                J_ring = build_ring_jacobian_homogeneous(
                    N_cells, steady_state_expected, baseline_params, hopping)
                bres = 0.0
            else:
                J_ring, ss_hetero, params_hetero, bres = build_ring_jacobian_heterogeneous(
                    N_cells, baseline_params, hopping, CV)
                if J_ring is None:
                    discarded += 1
                    continue
 
            disp = projected_dispersion(J_ring, PROJECTORS)
            m0_eig.append(disp[0])
            band_eig.append(np.max(disp[1:]))
            full_max = np.max(np.real(np.linalg.eigvals(J_ring)))
            proj_err.append(full_max - np.max(disp))
 
            m0_unstable = disp[0] >= 0
            band_dead   = np.max(disp[1:]) <= 0
            if not m0_unstable and not band_dead:
                turing_count += 1
            if m0_unstable: fail_m0 += 1
            if band_dead:   fail_band += 1
 
        n_valid = n_trials - discarded
        # TWO robustness definitions -- report both in the thesis:
        #   conditional: among trials where every cell kept a fixed point
        #   marginal:    among ALL trials (a lost fixed point is itself a noise failure)
        rob_cond = 100 * turing_count / n_valid if n_valid > 0 else np.nan
        rob_marg = 100 * turing_count / n_trials
 
        results_by_cv.append({
            'CV': CV, 'n_valid': n_valid, 'n_discarded': discarded,
            'discard_rate': 100 * discarded / n_trials,
            'mean_m0': np.mean(m0_eig) if m0_eig else np.nan,
            'mean_band': np.mean(band_eig) if band_eig else np.nan,
            'turing_count': turing_count,
            'fail_m0': fail_m0, 'fail_band': fail_band,
            'robustness_conditional': rob_cond,
            'robustness_marginal': rob_marg,
            'max_projection_error': np.max(np.abs(proj_err)) if proj_err else np.nan,
            'all_m0': np.array(m0_eig), 'all_band': np.array(band_eig),
        })
 
        print(f"CV={CV:<5} valid={n_valid}/{n_trials}  discarded={discarded} "
              f"({100*discarded/n_trials:.1f}%)")
        if n_valid > 0:
            print(f"    m=0 mean {np.mean(m0_eig):+.5f} | band mean {np.mean(band_eig):+.5f} "
                  f"| robustness {rob_cond:.1f}% (cond) / {rob_marg:.1f}% (marg)")
            print(f"    failures: m0={fail_m0} band={fail_band} | "
                  f"max|projection error|={np.max(np.abs(proj_err)):.4f}")
 
    # ---- summary ----
    print("\n" + "=" * 92)
    print(f"{'CV':<6}{'m=0':<11}{'band':<11}{'valid':<8}{'disc%':<8}"
          f"{'m0fail':<8}{'bandfail':<10}{'robust(cond)':<14}{'robust(marg)'}")
    print("-" * 92)
    for r in results_by_cv:
        print(f"{r['CV']:<6.2f}{r['mean_m0']:<+11.5f}{r['mean_band']:<+11.5f}"
              f"{r['n_valid']:<8}{r['discard_rate']:<8.1f}{r['fail_m0']:<8}"
              f"{r['fail_band']:<10}{r['robustness_conditional']:<14.1f}"
              f"{r['robustness_marginal']:.1f}")
    print("=" * 92)
 
    output_file = f'1754_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'results': results_by_cv, 'baseline_params': baseline_params,
                     'hopping': hopping, 'n_trials': n_trials,
                     'config_id': CONFIG_TO_TEST, 'config_name': row['config_name']}, f)
    print(f"Saved -> {output_file}")