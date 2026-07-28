#!/usr/bin/env python3
"""
Heterogeneous ring of Topology-3954 cells: robustness of the Turing instability
to parameter noise.

TOPOLOGY 3954: node u DOES self-activate (16 params).  1754 = same file with the
u self-activation removed (15 params).

METHOD
------
1. COUPLED STEADY STATE via HOMOTOPY CONTINUATION. Start from the exact
   homogeneous base state (CV=0) and ramp the per-cell noise up in small steps
   (params_i(s) = baseline * noise_i**s, s: 0 -> 1), refining with Newton at each
   step. This tracks the near-uniform branch and is robust to the bistability
   that u self-activation creates (a plain line-search solve stalls there and
   reports spurious 'no base state' discards). A discard now means the branch
   genuinely folds away, not solver failure.
2. PROJECTION-FREE TURING TEST (exact for heterogeneous rings):
       Condition A (reaction stability): reaction_max = max_i maxRe(J_i(X_i*)) < 0
       Condition B (diffusion-driven):   full_max     = maxRe(J_ring)          > 0
       Turing  <=>  reaction_max < 0  AND  full_max > 0

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# ======================================================================
# REACTION KINETICS  --  TOPOLOGY 3954 (u self-activates)
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
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    du = alpha_u + beta_u * hill_activation(u, K_uu) * hill_inhibition(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    return [du, dv, dw]

def find_steady_state(params, n_attempts=100, guess=None):
    """Single-cell positive fixed point (CV=0 baseline + diagnostics only)."""
    guesses = ([np.asarray(guess, float)] if guess is not None else [])
    guesses += [np.random.uniform(0.01, 10.0, 3) for _ in range(n_attempts)]
    for g in guesses:
        steady_state, info, ier, msg = fsolve(
            ode_system, g, args=(params,), full_output=True)
        residuals = ode_system(steady_state, params)
        if ier == 1 and np.max(np.abs(residuals)) < 1e-8 and np.all(steady_state > 0):
            return steady_state
    return None

def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    J = np.zeros((3, 3))
    # --- row u: 3954 KEEPS self-activation (differs from 1754) ---
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[0, 2] = 0
    # --- rows v, w: identical to 1754 ---
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
# RING GEOMETRY + (secondary) Fourier projection
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

def k_eff(N):
    return 2 * np.sin(np.pi * np.arange(N // 2 + 1) / N)

# ======================================================================
# COUPLED RING STEADY STATE (homotopy) + EXACT TURING TEST
# ======================================================================

def ring_rhs(X, params_list, Ldiff, N_cells):
    F = Ldiff @ X
    for i in range(N_cells):
        F[3*i:3*i+3] += ode_system(X[3*i:3*i+3], params_list[i])
    return F

def full_ring_jacobian(X, params_list, Ldiff, N_cells):
    J = Ldiff.copy()
    for i in range(N_cells):
        J[3*i:3*i+3, 3*i:3*i+3] += compute_jacobian(X[3*i:3*i+3], params_list[i])
    return J

def reaction_max_re(X, params_list, N_cells):
    """maxRe over the block-diagonal (diffusion off). Condition A wants this < 0."""
    return max(np.max(np.real(np.linalg.eigvals(compute_jacobian(X[3*i:3*i+3], params_list[i]))))
               for i in range(N_cells))

def _newton(X, params_list, Ldiff, N_cells, it=40):
    X = np.array(X, dtype=float)
    for _ in range(it):
        F = ring_rhs(X, params_list, Ldiff, N_cells)
        if np.linalg.norm(F) < 1e-11:
            break
        try:
            dX = np.linalg.solve(full_ring_jacobian(X, params_list, Ldiff, N_cells), -F)
        except np.linalg.LinAlgError:
            return None
        X = X + dX
    if np.max(np.abs(ring_rhs(X, params_list, Ldiff, N_cells))) < 1e-7 and np.all(X > 0):
        return X
    return None

def solve_ring_steady_state(baseline_params, noise_list, Ldiff, N_cells, baseline_ss):
    """Adaptive homotopy: params_i(s) = baseline * noise_i**s, s from 0 to 1,
    seeded at the exact homogeneous base state. Step halves on failure and grows
    back on success. Returns the coupled steady state, or None if the near-uniform
    branch genuinely folds away."""
    X = np.tile(baseline_ss, N_cells).astype(float)
    s, ds = 0.0, 1.0 / 8
    while s < 1.0 - 1e-9:
        s_try = min(s + ds, 1.0)
        params_s = [baseline_params * (nz ** s_try) for nz in noise_list]
        Xn = _newton(X, params_s, Ldiff, N_cells)
        if Xn is None:
            ds *= 0.5
            if ds < 1e-4:
                return None                 # genuine fold: base state lost
            continue
        X, s = Xn, s_try
        ds = min(ds * 1.5, 1.0 / 8)
    return X

def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    J_local = compute_jacobian(steady_state, params)
    J_ring = build_diffusion_operator(N_cells, hopping)
    for i in range(N_cells):
        J_ring[3*i:3*i+3, 3*i:3*i+3] += J_local
    return J_ring

def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV, baseline_ss):
    """Draw per-cell noise, solve the coupled ring steady state by homotopy, and
    return (J_ring, X, params_list, ring_resid) or (None,'no_coupled_ss',None,None)."""
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2
    noise_list = [np.random.lognormal(mu, sigma, size=len(baseline_params))
                  for _ in range(N_cells)]
    Ldiff = build_diffusion_operator(N_cells, hopping)
    X = solve_ring_steady_state(baseline_params, noise_list, Ldiff, N_cells, baseline_ss)
    if X is None:
        return None, "no_coupled_ss", None, None
    params_list = [baseline_params * nz for nz in noise_list]
    J_ring = full_ring_jacobian(X, params_list, Ldiff, N_cells)
    ring_resid = np.max(np.abs(ring_rhs(X, params_list, Ldiff, N_cells)))
    return J_ring, X, params_list, ring_resid

# ======================================================================
# MAIN: MONTE-CARLO CV SWEEP
# ======================================================================

if __name__ == "__main__":

    CONFIG_TO_TEST = 49
    CONFIG_LABEL   = "high"
    n_trials       = 1000
    N_cells        = 10

    df_file = pd.read_csv('../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv')
    df_params = df_file[df_file['classification'] == 'Type-I']
    row = df_params[(df_params['config_id'] == CONFIG_TO_TEST) &
                    (df_params['param_rank'] == 1)].iloc[0]

    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],   # u: 5 (incl K_uu)
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],   # v: 5
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']  # w: 6
    ])                                                                             # total: 16
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

    PROJECTORS = fourier_projectors(N_cells)

    J = compute_jacobian(steady_state_expected, baseline_params)
    turing = is_turing_shaberi(J, np.linalg.eigvals(J),
                               hopping['h_u'], hopping['h_v'], hopping['h_w'])
    J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,
                                              baseline_params, hopping)
    react0 = np.max(np.real(np.linalg.eigvals(J)))
    full0  = np.max(np.real(np.linalg.eigvals(J_ring0)))

    print("=" * 70)
    print(f"Continuous single-cell classification: {turing}")
    print(f"Baseline steady state (u*,v*,w*): {np.round(steady_state_expected, 4)} "
          f"(tiny components => real base-state loss expected under noise)")
    print(f"Discrete N={N_cells} baseline (EXACT): reaction_max {react0:+.4f} (<0?), "
          f"full_max {full0:+.4f} (>0?), Turing={react0 < 0 and full0 > 0}")
    print("=" * 70)

    np.random.seed(42)
    results_by_cv = []

    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        full_vals, react_vals, resids, proj_err = [], [], [], []
        turing_count = discarded = fail_reaction = fail_stable = 0

        for _ in range(n_trials):
            if CV == 0:
                J_ring = build_ring_jacobian_homogeneous(
                    N_cells, steady_state_expected, baseline_params, hopping)
                react_max = react0
                rres = 0.0
            else:
                J_ring, X, params_list, rres = build_ring_jacobian_heterogeneous(
                    N_cells, baseline_params, hopping, CV, steady_state_expected)
                if J_ring is None:
                    discarded += 1
                    continue
                react_max = reaction_max_re(X, params_list, N_cells)

            full_max = np.max(np.real(np.linalg.eigvals(J_ring)))
            full_vals.append(full_max)
            react_vals.append(react_max)
            resids.append(rres)
            disp = projected_dispersion(J_ring, PROJECTORS)
            proj_err.append(full_max - np.max(disp))

            if (react_max < 0) and (full_max > 0):
                turing_count += 1
            if react_max >= 0:
                fail_reaction += 1
            if full_max <= 0:
                fail_stable += 1

        n_valid = n_trials - discarded
        rob_cond = 100 * turing_count / n_valid if n_valid > 0 else np.nan
        rob_marg = 100 * turing_count / n_trials

        results_by_cv.append({
            'CV': CV, 'n_valid': n_valid, 'n_discarded': discarded,
            'discard_rate': 100 * discarded / n_trials,
            'mean_reaction': np.mean(react_vals) if react_vals else np.nan,
            'mean_full': np.mean(full_vals) if full_vals else np.nan,
            'turing_count': turing_count,
            'fail_reaction': fail_reaction, 'fail_stable': fail_stable,
            'robustness_conditional': rob_cond,
            'robustness_marginal': rob_marg,
            'max_projection_error': np.max(np.abs(proj_err)) if proj_err else np.nan,
            'max_ring_residual': np.max(resids) if resids else np.nan,
            'all_reaction': np.array(react_vals), 'all_full': np.array(full_vals),
        })

        print(f"CV={CV:<5} valid={n_valid}/{n_trials}  discarded={discarded} "
              f"({100*discarded/n_trials:.1f}%, base-state fold)")
        if n_valid > 0:
            print(f"    reaction_max mean {np.mean(react_vals):+.5f} (want <0) | "
                  f"full_max mean {np.mean(full_vals):+.5f} (want >0)")
            print(f"    Turing {rob_cond:.1f}% (cond) / {rob_marg:.1f}% (marg) | "
                  f"reaction-unstable={fail_reaction} fully-stable={fail_stable}")
            print(f"    max ring residual={np.max(resids):.2e}")

    print("\n" + "=" * 96)
    print(f"{'CV':<6}{'react_max':<12}{'full_max':<12}{'valid':<8}{'disc%':<8}"
          f"{'reactUns':<10}{'fullStab':<10}{'robust(cond)':<14}{'robust(marg)'}")
    print("-" * 96)
    for r in results_by_cv:
        print(f"{r['CV']:<6.2f}{r['mean_reaction']:<+12.5f}{r['mean_full']:<+12.5f}"
              f"{r['n_valid']:<8}{r['discard_rate']:<8.1f}{r['fail_reaction']:<10}"
              f"{r['fail_stable']:<10}{r['robustness_conditional']:<14.1f}"
              f"{r['robustness_marginal']:.1f}")
    print("=" * 96)

    output_file = f'3954_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'results': results_by_cv, 'baseline_params': baseline_params,
                     'hopping': hopping, 'n_trials': n_trials,
                     'config_id': CONFIG_TO_TEST, 'config_name': row['config_name']}, f)
    print(f"Saved -> {output_file}")