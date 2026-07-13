#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

# ======================================================================
# FUNCTIONS FROM OBJECTIVE 1 (HILL FUNCTIONS, ODE SYSTEM, STEADY STATE,
# SINGLE-CELL JACOBIAN, TURING CLASSIFICATION) -- unchanged
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

def find_steady_state(params, n_attempts=100):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        sol = fsolve(ode_system, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system(steady_state, params)
        if (ier == 1 and np.max(np.abs(residuals)) < 1e-8 and np.all(steady_state > 0)):
            return steady_state
    return None

def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    J = np.zeros((3, 3))
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[0, 2] = 0
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w

    return J

def is_turing_shaberi(J, eigs_0, DU, DV, DW):
    # STEP 1: Homogeneous steady state must be stable
    if np.max(np.real(eigs_0)) >= 0:
        return None

    # STEP 2: Sweep k in [0, 10], step 0.01 (Shaberi 2025 methodology)
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.01)

    max_reals = np.zeros(len(k_values))
    has_complex_unstable = False

    for i, k in enumerate(k_values):
        M = J - (k**2) * D
        eigs_k = np.linalg.eigvals(M)
        max_reals[i] = np.max(np.real(eigs_k))

        if max_reals[i] > 0:
            unstable_eigs = eigs_k[np.real(eigs_k) > 0]
            if np.any(np.abs(np.imag(unstable_eigs)) > 1e-8):
                has_complex_unstable = True

    if np.max(max_reals) <= 0:
        return None

    if has_complex_unstable:
        return 'Hopf'

    # STEP 3: Type-I = restabilises (goes negative) by k=10
    if max_reals[-1] < 0:
        return 'Type-I'

    # STEP 4: Distinguish Filter from Type-II by peak location
    max_idx = np.argmax(max_reals)
    if max_idx >= len(k_values) - 2:
        return 'Filter'

    return 'Type-II'


# ======================================================================
# HOMOGENEOUS RING JACOBIAN (identical cells) -- unchanged, used for the
# CV=0 baseline and as the ground-truth for the equivalence check
# ======================================================================

def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    J_local = compute_jacobian(steady_state, params)

    h_u = hopping['h_u']
    h_v = hopping['h_v']
    h_w = hopping['h_w']

    size = 3 * N_cells
    J_ring = np.zeros((size, size))

    for i in range(N_cells):
        idx = 3 * i
        J_ring[idx:idx+3, idx:idx+3] = J_local.copy()

        J_ring[idx,   idx]   -= 2*h_u
        J_ring[idx+1, idx+1] -= 2*h_v
        J_ring[idx+2, idx+2] -= 2*h_w

        left  = (i - 1) % N_cells
        right = (i + 1) % N_cells

        J_ring[idx,   3*left]   += h_u
        J_ring[idx+1, 3*left+1] += h_v
        J_ring[idx+2, 3*left+2] += h_w

        J_ring[idx,   3*right]   += h_u
        J_ring[idx+1, 3*right+1] += h_v
        J_ring[idx+2, 3*right+2] += h_w

    return J_ring


# ======================================================================
# HETEROGENEOUS RING -- CORRECTED
#
# Key ideas:
#   * The base state of a heterogeneous ring is NOT the per-cell isolated
#     fixed points; it is the spatially-varying solution of the fully
#     coupled system  R_i(x_i) + (L (x) D) x = 0.  We solve for that.
#   * The initial guess is the TILED baseline steady state (smooth,
#     in-basin, tracks the pattern-forming branch). Building the guess from
#     random-start per-cell solves lands different cells on different
#     branches of a multistable GRN -> jagged guess -> fsolve stalls.
#   * fsolve is given the exact analytic Jacobian (fprime).
#   * A continuation (homotopy) fallback ramps params baseline -> target
#     for trials the direct solve cannot crack.
# ======================================================================

def build_diffusion_operator(N_cells, hopping):
    """L (x) D : the ring diffusion operator. Geometry only, so it is
    constant for a given N and hopping, and it is exactly the diffusion
    part of the ring Jacobian."""
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    size = 3 * N_cells
    Ldiff = np.zeros((size, size))
    for i in range(N_cells):
        idx = 3 * i
        left  = (i - 1) % N_cells
        right = (i + 1) % N_cells
        for s in range(3):
            Ldiff[idx+s, idx+s]      -= 2 * h[s]
            Ldiff[idx+s, 3*left+s]   += h[s]
            Ldiff[idx+s, 3*right+s]  += h[s]
    return Ldiff

def ring_residual(X, params_list, Ldiff, N_cells):
    """Full coupled RHS: per-cell reaction + whole-ring diffusion."""
    react = np.zeros(3 * N_cells)
    for i in range(N_cells):
        idx = 3 * i
        react[idx:idx+3] = ode_system(X[idx:idx+3], params_list[i])
    return react + Ldiff @ X

def ring_jacobian_full(X, params_list, Ldiff, N_cells):
    """Analytic Jacobian of ring_residual: block-diagonal reaction
    Jacobians + the (constant) diffusion operator."""
    J = Ldiff.copy()
    for i in range(N_cells):
        idx = 3 * i
        J[idx:idx+3, idx:idx+3] += compute_jacobian(X[idx:idx+3], params_list[i])
    return J

def solve_ring_steady_state(params_list, baseline_params, Ldiff, N_cells,
                            baseline_ss, tol=1e-6, n_cont_steps=8):
    """Solve the coupled ring steady state, tracking the pattern-forming
    branch. Returns X_star, or None if it cannot converge below `tol`.

    Strategy 1: Newton from the tiled baseline steady state (smooth guess).
    Strategy 2 (fallback): parameter continuation baseline -> target.
    """
    # --- Strategy 1: direct solve from a smooth, near-uniform guess ---
    X = np.tile(baseline_ss, N_cells)
    X, info, ier, msg = fsolve(ring_residual, X,
                               args=(params_list, Ldiff, N_cells),
                               fprime=ring_jacobian_full, full_output=True)
    if np.max(np.abs(ring_residual(X, params_list, Ldiff, N_cells))) < tol:
        return X

    # --- Strategy 2: continuation in noise amplitude ---
    X = np.tile(baseline_ss, N_cells)
    for t in np.linspace(0, 1, n_cont_steps + 1)[1:]:
        interp = [(1 - t) * baseline_params + t * p for p in params_list]
        X, info, ier, msg = fsolve(ring_residual, X,
                                   args=(interp, Ldiff, N_cells),
                                   fprime=ring_jacobian_full, full_output=True)
    if np.max(np.abs(ring_residual(X, params_list, Ldiff, N_cells))) < tol:
        return X

    return None

def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV,
                                      baseline_ss=None):
    """Build the 3N x 3N ring Jacobian for a heterogeneous (noisy) ring.

    Returns (J_ring, steady_states, params_list) on success.
    On failure returns (None, reason, None) where reason is:
        "no_converge"     - solver could not find the coupled steady state
        "no_positive_ss"  - converged, but some component <= 0 (no positive
                            coupled steady state exists: a genuine result)
    """
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2
    params_list = [baseline_params * np.random.lognormal(mu, sigma, size=16)
                   for _ in range(N_cells)]

    if baseline_ss is None:
        baseline_ss = find_steady_state(baseline_params)
        if baseline_ss is None:
            return None, "no_converge", None

    Ldiff = build_diffusion_operator(N_cells, hopping)

    X_star = solve_ring_steady_state(params_list, baseline_params,
                                     Ldiff, N_cells, baseline_ss)
    if X_star is None:
        return None, "no_converge", None
    if np.any(X_star <= 0):
        return None, "no_positive_ss", None

    steady_states = [X_star[3*i:3*i+3] for i in range(N_cells)]

    J_ring = Ldiff.copy()
    for i in range(N_cells):
        idx = 3 * i
        J_ring[idx:idx+3, idx:idx+3] += compute_jacobian(steady_states[i],
                                                         params_list[i])

    return J_ring, steady_states, params_list


# ======================================================================
# MAIN: MONTE-CARLO CV SWEEP
# (Everything that touches the CSV / does the sweep lives here so the file
#  is safe to `import` from the dispersion script.)
# ======================================================================

if __name__ == "__main__":

    # ---- config selection ----
    CONFIG_TO_TEST = 49
    CONFIG_LABEL = "high"   # "high" or "low" -- change when you switch configs
    n_trials = 100
    N_cells = 10

    # ---- load parameters ----
    df_file = pd.read_csv('../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv')
    df_params = df_file[df_file['classification'] == 'Type-I']

    config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) &
                            (df_params['param_rank'] == 1)]
    row = config_data.iloc[0]

    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

    # ---- single-cell Turing classification (sanity) ----
    J = compute_jacobian(steady_state_expected, baseline_params)
    eigs = np.linalg.eigvals(J)
    turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])

    print("\n" + "="*70)
    print("STEP 4: MONTE CARLO - CV SWEEP")
    print("="*70)

    if turing == 'Type-I':
        J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,
                                                  baseline_params, hopping)
        max_real_ring = np.max(np.real(np.linalg.eigvals(J_ring0)))
        print(f"Homogeneous ring baseline Re(lambda) = {max_real_ring:.6f}")
        if max_real_ring < 0:
            print("WARNING: Ring baseline is stable "
                  "(continuous Turing peak between discrete k_m values)")
    else:
        print(f"WARNING: This config is not Type-I in continuous analysis (got: {turing})")

    # ---- EQUIVALENCE CHECK: at CV=0 the heterogeneous builder must
    #      reproduce the homogeneous ring Jacobian exactly. This also
    #      validates build_diffusion_operator / ring_residual, which the
    #      CV=0 sweep branch below never exercises. ----
    Jh, _, _ = build_ring_jacobian_heterogeneous(N_cells, baseline_params,
                                                 hopping, 0.0, steady_state_expected)
    Jhom = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,
                                           baseline_params, hopping)
    assert Jh is not None and np.allclose(Jh, Jhom, atol=1e-8), \
        "CV=0 equivalence FAILED -- diffusion operator / residual is wrong"
    print("CV=0 equivalence check passed (heterogeneous == homogeneous).")

    # ---- CV sweep ----
    np.random.seed(42)
    results_by_cv = []

    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

        print(f"\n{'='*70}")
        print(f"CV = {CV}")
        print(f"{'='*70}")

        max_eigenvalues = []
        turing_count = 0
        discarded_count = 0
        noconv_count = 0   # solver could not converge
        neg_count = 0      # converged but no positive coupled steady state

        for trial in range(n_trials):
            if CV == 0:
                J_ring = build_ring_jacobian_homogeneous(
                    N_cells=N_cells,
                    steady_state=steady_state_expected,
                    params=baseline_params,
                    hopping=hopping
                )
            else:
                J_ring, reason, params_hetero = build_ring_jacobian_heterogeneous(
                    N_cells=N_cells,
                    baseline_params=baseline_params,
                    hopping=hopping,
                    CV=CV,
                    baseline_ss=steady_state_expected
                )
                if J_ring is None:
                    discarded_count += 1
                    if reason == "no_positive_ss":
                        neg_count += 1
                    else:
                        noconv_count += 1
                    continue

            eigs = np.linalg.eigvals(J_ring)
            max_real = np.max(np.real(eigs))
            max_eigenvalues.append(max_real)
            if max_real > 0:
                turing_count += 1

        max_eigenvalues = np.array(max_eigenvalues)
        n_valid = len(max_eigenvalues)
        discard_rate = 100 * discarded_count / n_trials

        if n_valid > 0:
            robustness = 100 * turing_count / n_valid
            result = {
                'CV': CV,
                'mean_eig': np.mean(max_eigenvalues),
                'std_eig': np.std(max_eigenvalues),
                'median_eig': np.median(max_eigenvalues),
                'min_eig': np.min(max_eigenvalues),
                'max_eig': np.max(max_eigenvalues),
                'turing_count': turing_count,
                'n_valid': n_valid,
                'n_discarded': discarded_count,
                'n_noconv': noconv_count,
                'n_no_positive_ss': neg_count,
                'discard_rate': discard_rate,
                'robustness': robustness,
                'all_eigenvalues': max_eigenvalues
            }
        else:
            result = {
                'CV': CV,
                'mean_eig': np.nan, 'std_eig': np.nan, 'median_eig': np.nan,
                'min_eig': np.nan, 'max_eig': np.nan,
                'turing_count': 0, 'n_valid': 0,
                'n_discarded': discarded_count,
                'n_noconv': noconv_count,
                'n_no_positive_ss': neg_count,
                'discard_rate': discard_rate, 'robustness': np.nan,
                'all_eigenvalues': np.array([])
            }

        results_by_cv.append(result)

        print(f"  Valid trials:  {n_valid}/{n_trials}")
        print(f"  Discarded:     {discarded_count} ({discard_rate:.1f}%)")
        print(f"    - no convergence:      {noconv_count}")
        print(f"    - no positive SS:      {neg_count}   (genuine, not a bug)")
        if n_valid > 0:
            print(f"  Mean Re(lambda): {result['mean_eig']:.6f} +/- {result['std_eig']:.6f}")
            print(f"  Robustness:      {robustness:.1f}% ({turing_count}/{n_valid})")

    # ---- summary table ----
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'CV':<6} {'Mean Re(l)':<14} {'Std':<12} {'Valid':<8} "
          f"{'Discard%':<10} {'NoConv':<8} {'NoSS':<8} {'Robustness'}")
    print("-"*70)
    for r in results_by_cv:
        if r['n_valid'] > 0:
            print(f"{r['CV']:<6.2f} {r['mean_eig']:<14.6f} {r['std_eig']:<12.6f} "
                  f"{r['n_valid']:<8} {r['discard_rate']:<10.1f} "
                  f"{r['n_noconv']:<8} {r['n_no_positive_ss']:<8} "
                  f"{r['robustness']:.1f}% ({r['turing_count']}/{r['n_valid']})")
        else:
            print(f"{r['CV']:<6.2f} {'all discarded':<14} {'-':<12} "
                  f"{0:<8} {r['discard_rate']:<10.1f} "
                  f"{r['n_noconv']:<8} {r['n_no_positive_ss']:<8} -")
    print("="*70)

    # ---- save ----
    output_data = {
        'results': results_by_cv,
        'baseline_params': baseline_params,
        'hopping': hopping,
        'n_trials': n_trials,
        'config_id': CONFIG_TO_TEST,
        'config_name': row['config_name']
    }
    output_file = f'3954_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\nSaved results to {output_file}")
