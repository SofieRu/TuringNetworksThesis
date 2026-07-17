#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

n = 2

def hill_activation(X, K):
    return X**n / (K**n + X**n)

def hill_inhibition(X, K):
    return K**n / (K**n + X**n)

def dH_act(x, K):
    return n * K**n * x**(n-1) / (K**n + x**n)**2

def dH_inh(x, K):
    return -n * K**n * x**(n-1) / (K**n + x**n)**2

# K_uu removed, reindexed
def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]               
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]  
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    du = alpha_u + beta_u * hill_inhibition(v, K_vu) - delta_u * u  # no hill_activation(u) term
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
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[9:15]
    J = np.zeros((3, 3))
    J[0, 0] = -delta_u                    # NO self-activation term
    J[0, 1] = beta_u * dH_inh(v, K_vu)
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
    return 'Type-II'


# RING DIFFUSION OPERATOR  (L (x) D)  -- shared by both ring builders

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
    return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P))) for P in projectors])

def is_turing_ring(disp):
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


def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2

    params_list = []
    steady_states = []
    for i in range(N_cells):
        params_i = baseline_params * np.random.lognormal(mu, sigma, size=15)
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


# MAIN: MONTE-CARLO CV SWEEP

if __name__ == "__main__":

    CONFIG_TO_TEST = 49 #maybe 21 or 3 
    CONFIG_LABEL = "high"   # "high" or "low" or "lab"
    n_trials = 1000
    N_cells = 30

    df_file = pd.read_csv('../TopologyRanking/Topology1754/1754_FINAL_lhs_results_parameters.csv')
    df_params = df_file[df_file['classification'] == 'Type-I']

    config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) & (df_params['param_rank'] == 1)]
    row = config_data.iloc[0]

    baseline_params = np.array([
        row['alpha_u'], row['beta_u'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']])
    
    steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}

    # ================= DIAGNOSTIC =================
    print("\n" + "="*70)
    print("DIAGNOSTIC: is the CSV steady state actually a root of ode_system?")
    print("="*70)
    res = ode_system(steady_state_expected, baseline_params)
    print("CSV steady state:", steady_state_expected)
    print("residual of CSV steady state:", np.max(np.abs(res)))

    print("\nfind_steady_state on UNPERTURBED params, 10 draws:")
    for _ in range(10):
        print("   ", find_steady_state(baseline_params))
    print("="*70)
    # ============== END DIAGNOSTIC ================

    print("\n" + "="*70)
    print("baseline_params:", np.array2string(baseline_params, precision=4))
    print("="*70)

    np.random.seed(123)
    CV_test = 0.05
    sigma = np.sqrt(np.log(1 + CV_test**2))
    mu = -sigma**2 / 2
    NP = len(baseline_params)          # 15 for 1754, 16 for 3954

    print(f"\nONE TRIAL AT CV={CV_test}: per-cell steady states and local eigenvalues")
    print(f"{'cell':<5} {'u*':<10} {'v*':<10} {'w*':<10} {'maxRe J_i':<12} {'||J_i||'}")
    print("-"*70)
    J_list = []
    for i in range(N_cells):
        params_i = baseline_params * np.random.lognormal(mu, sigma, size=NP)
        ss_i = find_steady_state(params_i)
        if ss_i is None:
            print(f"{i:<5} NO STEADY STATE")
            continue
        J_i = compute_jacobian(ss_i, params_i)
        J_list.append(J_i)
        print(f"{i:<5} {ss_i[0]:<10.5f} {ss_i[1]:<10.5f} {ss_i[2]:<10.5f} "
              f"{np.max(np.real(np.linalg.eigvals(J_i))):<+12.5f} "
              f"{np.linalg.norm(J_i):.3f}")

    print("-"*70)
    print("baseline  ", np.array2string(steady_state_expected, precision=5))
    J_base = compute_jacobian(steady_state_expected, baseline_params)
    print(f"baseline maxRe J = {np.max(np.real(np.linalg.eigvals(J_base))):+.5f}   "
          f"||J|| = {np.linalg.norm(J_base):.3f}")
    if J_list:
        J_mean = np.mean(J_list, axis=0)
        print(f"mean(J_i) maxRe   = {np.max(np.real(np.linalg.eigvals(J_mean))):+.5f}")
    print("="*70)

    # projectors built ONCE (geometry only, reused every trial)
    PROJECTORS = _fourier_projectors(N_cells)

    # ---- single-cell Turing classification (sanity) ----
    J = compute_jacobian(steady_state_expected, baseline_params)
    eigs = np.linalg.eigvals(J)
    turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])

    # print("\n" + "="*70)
    # print("STEP 4: MONTE CARLO - CV SWEEP  (proper Turing metric)")
    # print("="*70)

    # if turing == 'Type-I':
    #     J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected,baseline_params, hopping)
    #     disp0 = fourier_projected_dispersion(J_ring0, PROJECTORS)
    #     print(f"Homogeneous ring baseline: m=0 {disp0[0]:+.4f}, "
    #           f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
    # else:
    #     print(f"WARNING: This config is not Type-I in continuous analysis (got: {turing})")

    print("\n" + "="*70)
    print("STEP 4: MONTE CARLO - CV SWEEP  (proper Turing metric)")
    print("="*70)

    J_ring0 = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
    disp0 = fourier_projected_dispersion(J_ring0, PROJECTORS)

    print(f"Continuous classification: {turing}")
    print(f"Homogeneous ring baseline: m=0 {disp0[0]:+.4f}, " f"max(m>0) {np.max(disp0[1:]):+.4f}, Turing={is_turing_ring(disp0)}")
    print("disp0 per mode:", np.round(disp0, 5))
    print("unstable modes:", np.where(disp0[1:] > 0)[0] + 1)

    k_eff = 2 * np.sin(np.pi * np.arange(N_cells // 2 + 1) / N_cells)
    for m, (k, g) in enumerate(zip(k_eff, disp0)):
        print(f"  m={m}  k_eff={k:.4f}  Re(lambda)={g:+.5f}  {'UNSTABLE' if g > 0 else ''}")

    if turing != 'Type-I':
        print(f"WARNING: not Type-I in continuous analysis (got: {turing})")
    
    np.random.seed(42)
    results_by_cv = []

    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

            print(f"\n{'='*70}")
            print(f"CV = {CV}")
            print(f"{'='*70}")

            max_eigenvalues = []      # max over ALL modes (kept for continuity)
            band_eigenvalues = []     # max over m>0 only  (the Turing band itself)
            m0_eigenvalues = []       # the uniform mode
            turing_count = 0
            discarded_count = 0       # a cell had no positive isolated fixed point
            fail_m0 = 0               # uniform mode went unstable
            fail_band = 0             # band collapsed

            for trial in range(n_trials):
                if CV == 0:
                    J_ring = build_ring_jacobian_homogeneous(N_cells=N_cells, steady_state=steady_state_expected,params=baseline_params, hopping=hopping)
                else:
                    # J_ring, reason, params_hetero = build_ring_jacobian_heterogeneous(N_cells=N_cells, baseline_params=baseline_params, hopping=hopping, CV=CV)

                    if J_ring is None:
                        discarded_count += 1
                        continue

                # ---- TURING CLASSIFICATION via projected dispersion ----
                disp = fourier_projected_dispersion(J_ring, PROJECTORS)

                max_eigenvalues.append(np.max(disp))
                band_eigenvalues.append(np.max(disp[1:]))
                m0_eigenvalues.append(disp[0])

                # counted independently so the two failure modes can overlap
                m0_unstable = disp[0] >= 0
                band_dead = np.max(disp[1:]) <= 0

                if not m0_unstable and not band_dead:
                    turing_count += 1
                if m0_unstable:
                    fail_m0 += 1
                if band_dead:
                    fail_band += 1

            max_eigenvalues = np.array(max_eigenvalues)
            band_eigenvalues = np.array(band_eigenvalues)
            m0_eigenvalues = np.array(m0_eigenvalues)
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
                    'mean_band': np.mean(band_eigenvalues),
                    'std_band': np.std(band_eigenvalues),
                    'mean_m0': np.mean(m0_eigenvalues),
                    'std_m0': np.std(m0_eigenvalues),
                    'turing_count': turing_count,
                    'n_valid': n_valid,
                    'n_discarded': discarded_count,
                    'discard_rate': discard_rate,
                    'fail_m0': fail_m0,
                    'fail_band': fail_band,
                    'robustness': robustness,
                    'all_eigenvalues': max_eigenvalues,
                    'all_band_eigenvalues': band_eigenvalues,
                    'all_m0_eigenvalues': m0_eigenvalues
                }
            else:
                result = {
                    'CV': CV,
                    'mean_eig': np.nan, 'std_eig': np.nan, 'median_eig': np.nan,
                    'min_eig': np.nan, 'max_eig': np.nan,
                    'mean_band': np.nan, 'std_band': np.nan,
                    'mean_m0': np.nan, 'std_m0': np.nan,
                    'turing_count': 0, 'n_valid': 0,
                    'n_discarded': discarded_count,
                    'discard_rate': discard_rate,
                    'fail_m0': fail_m0,
                    'fail_band': fail_band,
                    'robustness': np.nan,
                    'all_eigenvalues': np.array([]),
                    'all_band_eigenvalues': np.array([]),
                    'all_m0_eigenvalues': np.array([])
                }

            results_by_cv.append(result)

            print(f"  Valid trials:  {n_valid}/{n_trials}")
            print(f"  Discarded:     {discarded_count} ({discard_rate:.1f}%)  (no positive isolated SS)")
            if n_valid > 0:
                print(f"  max proj Re(lambda): {result['mean_eig']:.6f} +/- {result['std_eig']:.6f}")
                print(f"  band  max Re(m>0):   {result['mean_band']:+.6f} +/- {result['std_band']:.6f}")
                print(f"  uniform mode Re(m=0):{result['mean_m0']:+.6f} +/- {result['std_m0']:.6f}")
                print(f"  Turing robustness:   {robustness:.1f}% ({turing_count}/{n_valid})")
                print(f"  Failures: m0={fail_m0} ({100*fail_m0/n_valid:.1f}%)  "
                    f"band={fail_band} ({100*fail_band/n_valid:.1f}%)")

    # print("\n" + "="*70)
    print("\n" + "="*95)
    print("SUMMARY TABLE  (robustness = m=0 stable AND some m>0 unstable)")
    print("="*95)
    print(f"{'CV':<6} {'Mean m=0':<12} {'Mean band':<12} {'Valid':<8} {'Discard%':<10} "f"{'m0 fail':<9} {'band fail':<11} {'Robustness'}")
    print("-"*95)
        
    for r in results_by_cv:
        if r['n_valid'] > 0:
            print(f"{r['CV']:<6.2f} {r['mean_m0']:<+12.6f} {r['mean_band']:<+12.6f} "
                f"{r['n_valid']:<8} {r['discard_rate']:<10.1f} "
                f"{r['fail_m0']:<9} {r['fail_band']:<11} "
                f"{r['robustness']:.1f}% ({r['turing_count']}/{r['n_valid']})")
        else:
            print(f"{r['CV']:<6.2f} {'all discarded':<12} {'-':<12} "f"{0:<8} {r['discard_rate']:<10.1f} {'-':<9} {'-':<11} -")
    print("="*95)

    output_data = {
        'results': results_by_cv,
        'baseline_params': baseline_params,
        'hopping': hopping,
        'n_trials': n_trials,
        'config_id': CONFIG_TO_TEST,
        'config_name': row['config_name']
    }
    output_file = f'1754_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\nSaved results to {output_file}")






    # for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

    #     print(f"\n{'='*70}")
    #     print(f"CV = {CV}")
    #     print(f"{'='*70}")

    #     max_eigenvalues = []
    #     turing_count = 0
    #     discarded_count = 0   # a cell had no positive isolated fixed point

    #     for trial in range(n_trials):
    #         if CV == 0:
    #             J_ring = build_ring_jacobian_homogeneous(N_cells=N_cells, steady_state=steady_state_expected, params=baseline_params, hopping=hopping)
    #         else:
    #             J_ring, reason, params_hetero = build_ring_jacobian_heterogeneous(N_cells=N_cells, baseline_params=baseline_params, hopping=hopping, CV=CV)
    #             if J_ring is None:
    #                 discarded_count += 1
    #                 continue

    #         # ---- PROPER TURING CLASSIFICATION via projected dispersion ----
    #         disp = fourier_projected_dispersion(J_ring, PROJECTORS)
    #         max_eigenvalues.append(np.max(disp))
    #         if is_turing_ring(disp):          # m=0 stable AND some m>0 unstable
    #             turing_count += 1

    #     max_eigenvalues = np.array(max_eigenvalues)
    #     n_valid = len(max_eigenvalues)
    #     discard_rate = 100 * discarded_count / n_trials

    #     if n_valid > 0:
    #         robustness = 100 * turing_count / n_valid
    #         result = {
    #             'CV': CV,
    #             'mean_eig': np.mean(max_eigenvalues),
    #             'std_eig': np.std(max_eigenvalues),
    #             'median_eig': np.median(max_eigenvalues),
    #             'min_eig': np.min(max_eigenvalues),
    #             'max_eig': np.max(max_eigenvalues),
    #             'turing_count': turing_count,
    #             'n_valid': n_valid,
    #             'n_discarded': discarded_count,
    #             'discard_rate': discard_rate,
    #             'robustness': robustness,
    #             'all_eigenvalues': max_eigenvalues
    #         }
    #     else:
    #         result = {
    #             'CV': CV,
    #             'mean_eig': np.nan, 'std_eig': np.nan, 'median_eig': np.nan,
    #             'min_eig': np.nan, 'max_eig': np.nan,
    #             'turing_count': 0, 'n_valid': 0,
    #             'n_discarded': discarded_count,
    #             'discard_rate': discard_rate, 'robustness': np.nan,
    #             'all_eigenvalues': np.array([])
    #         }

    #     results_by_cv.append(result)

    #     print(f"  Valid trials:  {n_valid}/{n_trials}")
    #     print(f"  Discarded:     {discarded_count} ({discard_rate:.1f}%)  (no positive isolated SS)")
    #     if n_valid > 0:
    #         print(f"  max proj Re(lambda): {result['mean_eig']:.6f} +/- {result['std_eig']:.6f}")
    #         print(f"  Turing robustness:   {robustness:.1f}% ({turing_count}/{n_valid})")

    # print("\n" + "="*70)
    # print("SUMMARY TABLE  (robustness = fraction that are proper Turing)")
    # print("="*70)
    # print(f"{'CV':<6} {'Mean maxRe':<14} {'Std':<12} {'Valid':<8} {'Discard%':<10} {'Robustness'}")
    # print("-"*70)
    # for r in results_by_cv:
    #     if r['n_valid'] > 0:
    #         print(f"{r['CV']:<6.2f} {r['mean_eig']:<14.6f} {r['std_eig']:<12.6f} "
    #               f"{r['n_valid']:<8} {r['discard_rate']:<10.1f} "
    #               f"{r['robustness']:.1f}% ({r['turing_count']}/{r['n_valid']})")
    #     else:
    #         print(f"{r['CV']:<6.2f} {'all discarded':<14} {'-':<12} "
    #               f"{0:<8} {r['discard_rate']:<10.1f} -")
    # print("="*70)

    # output_data = {
    #     'results': results_by_cv,
    #     'baseline_params': baseline_params,
    #     'hopping': hopping,
    #     'n_trials': n_trials,
    #     'config_id': CONFIG_TO_TEST,
    #     'config_name': row['config_name']
    # }
    # output_file = f'1754_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
    # with open(output_file, 'wb') as f:
    #     pickle.dump(output_data, f)
    # print(f"\nSaved results to {output_file}")