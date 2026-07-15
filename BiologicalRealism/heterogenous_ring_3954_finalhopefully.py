#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

# ======================================================================
# FUNCTIONS FROM OBJECTIVE 1
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

    if max_reals[-1] < 0:
        return 'Type-I'

    max_idx = np.argmax(max_reals)
    if max_idx >= len(k_values) - 2:
        return 'Filter'



# ======================================================================
# RING DIFFUSION OPERATOR  (L (x) D)  -- shared by both ring builders
# ======================================================================

def build_diffusion_operator(N_cells, hopping):
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


# ======================================================================
# FOURIER-PROJECTED DISPERSION  (Galerkin projection onto wavenumber modes)
#
# For each discrete wavenumber m, project the ring Jacobian onto the Fourier
# mode phi_m(j) = exp(2*pi*i*m*j/N) (one copy per species) and take max Re of
# the resulting 3x3 effective operator P_m^H J_ring P_m.
#   * Homogeneous ring -> equals J - k_m^2 D exactly (standard dispersion).
#   * Heterogeneous ring -> the effective growth rate of Fourier mode m.
# Returns exactly N/2+1 values, one per k_m. No eigenvalue binning, so no
# starved bins / spurious deep-negative spikes.
#
# THIS IS ALSO WHAT THE ROBUSTNESS METRIC USES: a proper Turing instability
# requires the uniform mode m=0 to be STABLE (disp[0] < 0) and at least one
# finite wavenumber to be UNSTABLE (max(disp[1:]) > 0). The old metric
# max Re(lambda) > 0 could not see this distinction and therefore counted
# non-Turing (m=0 unstable) instabilities as "robust".
# ======================================================================

def _fourier_projectors(N):
    projs = []
    for m in range(N // 2 + 1):
        phi = np.exp(2j * np.pi * m * np.arange(N) / N) / np.sqrt(N)
        P = np.zeros((3 * N, 3), dtype=complex)
        for j in range(N):
            for s in range(3):
                P[3*j+s, s] = phi[j]
        projs.append(P)
    return projs

def fourier_projected_dispersion(J_ring, projectors):
    return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P)))
                     for P in projectors])

def is_turing_ring(disp):
    """Proper Turing test on a projected dispersion vector:
    uniform mode stable AND some finite wavenumber unstable."""
    return (disp[0] < 0) and (np.max(disp[1:]) > 0)


# ======================================================================
# HOMOGENEOUS RING JACOBIAN (identical cells) -- used for CV = 0
# ======================================================================

def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    Ldiff = build_diffusion_operator(N_cells, hopping)
    J_local = compute_jacobian(steady_state, params)
    J_ring = Ldiff.copy()
    for i in range(N_cells):
        idx = 3 * i
        J_ring[idx:idx+3, idx:idx+3] += J_local
    return J_ring


# ======================================================================
# HETEROGENEOUS RING JACOBIAN  --  FROZEN-COEFFICIENT (LOCAL) APPROACH
#
# Each cell gets its own noisy parameter set and its own isolated reaction
# fixed point x_i*. The ring Jacobian is assembled from the per-cell reaction
# Jacobians at those fixed points plus the ring diffusion operator. This was
# validated against the full coupled steady-state analysis (100% agreement on
# the Turing verdict for convergent trials); the coupled solve is intractable
# here because the baseline ring is Turing-unstable (its coupled steady state
# is an unstable saddle).
#
# Returns (J_ring, steady_states, params_list) on success, or
# (None, "no_isolated_ss", None) if a cell has no positive reaction fixed
# point (a genuine, reportable outcome).
# ======================================================================

def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2

    params_list = []
    steady_states = []
    for i in range(N_cells):
        params_i = baseline_params * np.random.lognormal(mu, sigma, size=16)
        ss_i = find_steady_state(params_i)
        if ss_i is None:
            return None, "no_isolated_ss", None
        params_list.append(params_i)
        steady_states.append(ss_i)

    Ldiff = build_diffusion_operator(N_cells, hopping)
    J_ring = Ldiff.copy()
    for i in range(N_cells):
        idx = 3 * i
        J_ring[idx:idx+3, idx:idx+3] += compute_jacobian(steady_states[i], params_list[i])

    return J_ring, steady_states, params_list


# ======================================================================
# MAIN: MONTE-CARLO CV SWEEP
# ======================================================================

if __name__ == "__main__":

    CONFIG_TO_TEST = 21 #maybe 21 or 3 
    CONFIG_LABEL = "low"   # "high" or "low"
    n_trials = 1000
    N_cells = 20

    df_file = pd.read_csv('../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv')
    df_params = df_file[df_file['classification'] == 'Type-I']

    config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) & (df_params['param_rank'] == 1)]
    row = config_data.iloc[0]

    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

    # projectors built ONCE (geometry only, reused every trial)
    PROJECTORS = _fourier_projectors(N_cells)

    # ---- single-cell Turing classification (sanity) ----
    J = compute_jacobian(steady_state_expected, baseline_params)
    eigs = np.linalg.eigvals(J)
    turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])

    print("\n" + "="*70)
    print("STEP 4: MONTE CARLO - CV SWEEP  (proper Turing metric)")
    print("="*70)

    if turing == 'Type-I':
        J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,baseline_params, hopping)
        disp0 = fourier_projected_dispersion(J_ring0, PROJECTORS)
        print(f"Homogeneous ring baseline: m=0 {disp0[0]:+.4f}, "
              f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
    else:
        print(f"WARNING: This config is not Type-I in continuous analysis (got: {turing})")

    # ---- CV sweep ----
    np.random.seed(42)
    results_by_cv = []

    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

        print(f"\n{'='*70}")
        print(f"CV = {CV}")
        print(f"{'='*70}")

        max_eigenvalues = []
        turing_count = 0
        discarded_count = 0   # a cell had no positive isolated fixed point

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
                    CV=CV
                )
                if J_ring is None:
                    discarded_count += 1
                    continue

            # ---- PROPER TURING CLASSIFICATION via projected dispersion ----
            disp = fourier_projected_dispersion(J_ring, PROJECTORS)
            max_eigenvalues.append(np.max(disp))
            if is_turing_ring(disp):          # m=0 stable AND some m>0 unstable
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
                'discard_rate': discard_rate, 'robustness': np.nan,
                'all_eigenvalues': np.array([])
            }

        results_by_cv.append(result)

        print(f"  Valid trials:  {n_valid}/{n_trials}")
        print(f"  Discarded:     {discarded_count} ({discard_rate:.1f}%)  (no positive isolated SS)")
        if n_valid > 0:
            print(f"  max proj Re(lambda): {result['mean_eig']:.6f} +/- {result['std_eig']:.6f}")
            print(f"  Turing robustness:   {robustness:.1f}% ({turing_count}/{n_valid})")

    print("\n" + "="*70)
    print("SUMMARY TABLE  (robustness = fraction that are proper Turing)")
    print("="*70)
    print(f"{'CV':<6} {'Mean maxRe':<14} {'Std':<12} {'Valid':<8} {'Discard%':<10} {'Robustness'}")
    print("-"*70)
    for r in results_by_cv:
        if r['n_valid'] > 0:
            print(f"{r['CV']:<6.2f} {r['mean_eig']:<14.6f} {r['std_eig']:<12.6f} "
                  f"{r['n_valid']:<8} {r['discard_rate']:<10.1f} "
                  f"{r['robustness']:.1f}% ({r['turing_count']}/{r['n_valid']})")
        else:
            print(f"{r['CV']:<6.2f} {'all discarded':<14} {'-':<12} "
                  f"{0:<8} {r['discard_rate']:<10.1f} -")
    print("="*70)

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
