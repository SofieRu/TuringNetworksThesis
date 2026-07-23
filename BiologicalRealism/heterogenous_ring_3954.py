#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# ORIGINAL VERSION WHERE I AM NOT SURE IF ITS OC
# FUNCTIONS FROM OBJECTIVE 1 (HILL FUNCTIONS, ODE SYSTEM, STEADY STATE FINDING, JACOBIAN)

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

def find_steady_state(params, n_attempts=100): # was 10 
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3) # np.random.uniform(0.001, 50.0, 3) bc broader range
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
    
    # STEP 2: Sweep k ∈ [0, 10] with step 0.01 (Shaberi 2025 methodology)
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.01) # change later back to 0.01
    
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
    # Filter (Diego 2018): monotonic — max sits at the END of the range
    # Type-II: has an interior peak — max is somewhere in the middle
    max_idx = np.argmax(max_reals)
    
    # Allow a tiny buffer for floating-point noise (last 0.2% of range)
    if max_idx >= len(k_values) - 2:
        return 'Filter'
    
    return 'Type-II'


# PARAMETRS FOR RING (EXAMPLE)

# THINGS TO CHECK: WHY ARE THERE SOO MANY DISCARDED...??!!
CONFIG_TO_TEST = 21
CONFIG_LABEL = "low"  # " high" or "low" — change this once when you switch configs
n_trials = 1000
N_cells = 10

# Load parameters from CSV
df_file = pd.read_csv('../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv')
df_params = df_file[df_file['classification'] == 'Type-I'] #RECENT CHANGE BC WE MODIFY THE PARAMETERS WE SAVE

# Get the best parameter set for this config and extract data
config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) & (df_params['param_rank'] == 1)]
row = config_data.iloc[0]

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])

steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])

hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW'],}

###########################################

# FUNCTION TO BUILD JACOBIAN FOR RING OF IDENTICAL CELLS

# Parameters: 
# N_cells: number of cells (e.g., 10)
# steady_state: [u*, v*, w*] from single cell
# params: your parameter array
# hopping: dict with h_u, h_v, h_w

def build_ring_jacobian_homogeneous(N_cells, steady_state, params, hopping):
    # Get local 3×3 Jacobian (same for all cells)
    J_local = compute_jacobian(steady_state, params)
    
    # Extract hopping rates
    h_u = hopping['h_u']
    h_v = hopping['h_v']
    h_w = hopping['h_w']
    
    # Build big matrix
    size = 3 * N_cells  # 30 for N=10
    J_ring = np.zeros((size, size))
    
    for i in range(N_cells):
        # Starting row/col for cell i
        idx = 3 * i
        
        # Put local Jacobian in diagonal block
        J_ring[idx:idx+3, idx:idx+3] = J_local.copy()
        
        # Add hopping terms (molecules leaving this cell)
        J_ring[idx,   idx]   -= 2*h_u  # u leaves to both neighbors
        J_ring[idx+1, idx+1] -= 2*h_v  # v leaves
        J_ring[idx+2, idx+2] -= 2*h_w  # w leaves
        
        # Neighbors in a ring
        left = (i - 1) % N_cells
        right = (i + 1) % N_cells
        
        # Coupling FROM left neighbor
        J_ring[idx,   3*left]   += h_u
        J_ring[idx+1, 3*left+1] += h_v
        J_ring[idx+2, 3*left+2] += h_w
        
        # Coupling FROM right neighbor
        J_ring[idx,   3*right]   += h_u
        J_ring[idx+1, 3*right+1] += h_v
        J_ring[idx+2, 3*right+2] += h_w
    
    return J_ring


# OLD VERSION THAT IS APPARNETLY NOT CORRECT :((
def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):
    
    # Generate perturbed parameters AND find steady states
    params_list = []
    steady_states = []
    
    # Lognormal parameters
    sigma = np.sqrt(np.log(1 + CV**2))
    mu = -sigma**2 / 2
    
    for i in range(N_cells):
        # Generate noise
        noise_factors = np.random.lognormal(mu, sigma, size=16)
        params_i = baseline_params * noise_factors
        
        # Try to find steady state
        ss_i = find_steady_state(params_i)
        
        if ss_i is None:
            # Can't find steady state - revert to wild-type
            # params_i = baseline_params.copy()  # Use baseline params
            # ss_i = steady_state_expected.copy()          # Use baseline steady state
            return None, None, None # NEW: be honest about how many didnt find steady state 
        
        # Append MATCHED pair
        params_list.append(params_i)
        steady_states.append(ss_i)
    
    # Build Jacobian (rest stays the same)
    h_u = hopping['h_u']
    h_v = hopping['h_v']
    h_w = hopping['h_w']
    
    size = 3 * N_cells
    J_ring = np.zeros((size, size))
    
    for i in range(N_cells):
        idx = 3 * i
        
        # Local Jacobian with MATCHED params and steady state
        J_local = compute_jacobian(steady_states[i], params_list[i])
        J_ring[idx:idx+3, idx:idx+3] = J_local
        
        # Hopping
        J_ring[idx, idx] -= 2*h_u
        J_ring[idx+1, idx+1] -= 2*h_v
        J_ring[idx+2, idx+2] -= 2*h_w
        
        # Coupling
        left = (i - 1) % N_cells
        right = (i + 1) % N_cells
        
        J_ring[idx, 3*left] += h_u
        J_ring[idx+1, 3*left+1] += h_v
        J_ring[idx+2, 3*left+2] += h_w
        
        J_ring[idx, 3*right] += h_u
        J_ring[idx+1, 3*right+1] += h_v
        J_ring[idx+2, 3*right+2] += h_w
    
    return J_ring, steady_states, params_list


###########################################

# TESTING THE FUNCTIONS
residuals = ode_system(steady_state_expected, baseline_params)
# print("\nSTEP 1: Check if we get Turing instability from single cell Jacobian")

# Compute Jacobian at THIS steady state
J = compute_jacobian(steady_state_expected, baseline_params)

# Check stability
eigs = np.linalg.eigvals(J)

# Check Turing
turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 4: MONTE CARLO - CV SWEEP")
    print("="*70)

    # Settings, CHANGE HERE FOR VARIATION
    # n_trials = 1000
    # N_cells = 10 # for sanity check run with N = 5, 10 and 20, 30??

    if turing == 'Type-I':
        # N_cells = 10
        J_ring = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
        eigs_ring = np.linalg.eigvals(J_ring)
        max_real_ring = np.max(np.real(eigs_ring))
        print(f"Homogeneous ring baseline Re(λ) = {max_real_ring:.6f}")
        if max_real_ring < 0:
            print("WARNING: Ring baseline is stable (continuous Turing peak between discrete k_m values)")
    else:
        print(f"WARNING: This config is not Type-I in continuous analysis (got: {turing})")

    np.random.seed(42)
    results_by_cv = []

    for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        
        print(f"\n{'='*70}")
        print(f"CV = {CV}")
        print(f"{'='*70}")
        
        max_eigenvalues = []
        turing_count = 0
        discarded_count = 0
        cellfail_count = 0
        resid_count = 0
        neg_count = 0
        
        for trial in range(n_trials):
            if CV == 0:
                J_ring = build_ring_jacobian_homogeneous(
                    N_cells=N_cells,
                    steady_state=steady_state_expected,
                    params=baseline_params,
                    hopping=hopping
                )
            else:
                J_ring, ss_hetero, params_hetero = build_ring_jacobian_heterogeneous(
                    N_cells=N_cells,
                    baseline_params=baseline_params,
                    hopping=hopping,
                    CV=CV
                )
                
                # NEW CHANGES
                if J_ring is None:
                    discarded_count += 1
                    cellfail_count += 1 # per-cell loop couldn't seed a guess
                    continue  # Skip this trial entirely
                if isinstance(J_ring, str):
                    discarded_count += 1
                    if J_ring == "FAIL_RESIDUAL":
                        resid_count += 1
                    else:
                        neg_count += 1
                    continue  # Skip this trial entirely

            
            eigs = np.linalg.eigvals(J_ring)
            max_real = np.max(np.real(eigs))
            
            max_eigenvalues.append(max_real)
            
            if max_real > 0:
                turing_count += 1
        
        # Statistics on VALID trials only
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
        
        print(f"  Valid trials: {n_valid}/{n_trials}")
        print(f"  Discarded:    {discarded_count} ({discard_rate:.1f}%)")
        if n_valid > 0:
            print(f"  Mean Re(λ):   {result['mean_eig']:.6f} ± {result['std_eig']:.6f}")
            print(f"  Robustness:   {robustness:.1f}% ({turing_count}/{n_valid})")


    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'CV':<6} {'Mean Re(λ)':<14} {'Std':<12} {'Valid':<8} {'Discard%':<10} {'Robustness'}")
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


    # Save results to file
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




# print("\n" + "="*70)
# print("STEP 4: MONTE CARLO - CV SWEEP")
# print("="*70)

# # Settings, CHANGE HERE FOR VARIATION
# # n_trials = 1000
# # N_cells = 10 # for sanity check run with N = 5, 10 and 20, 30??

# if turing == 'Type-I':
#     # N_cells = 10
#     J_ring = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
#     eigs_ring = np.linalg.eigvals(J_ring)
#     max_real_ring = np.max(np.real(eigs_ring))
#     print(f"Homogeneous ring baseline Re(λ) = {max_real_ring:.6f}")
#     if max_real_ring < 0:
#         print("WARNING: Ring baseline is stable (continuous Turing peak between discrete k_m values)")
# else:
#     print(f"WARNING: This config is not Type-I in continuous analysis (got: {turing})")

# np.random.seed(42)
# results_by_cv = []

# for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
#     print(f"\n{'='*70}")
#     print(f"CV = {CV}")
#     print(f"{'='*70}")
    
#     max_eigenvalues = []
#     turing_count = 0
#     discarded_count = 0
    
#     for trial in range(n_trials):
#         if CV == 0:
#             J_ring = build_ring_jacobian_homogeneous(
#                 N_cells=N_cells,
#                 steady_state=steady_state_expected,
#                 params=baseline_params,
#                 hopping=hopping
#             )
#         else:
#             J_ring, ss_hetero, params_hetero = build_ring_jacobian_heterogeneous(
#                 N_cells=N_cells,
#                 baseline_params=baseline_params,
#                 hopping=hopping,
#                 CV=CV
#             )
            
#             if J_ring is None:
#                 discarded_count += 1
#                 continue  # Skip this trial entirely
        
#         eigs = np.linalg.eigvals(J_ring)
#         max_real = np.max(np.real(eigs))
        
#         max_eigenvalues.append(max_real)
        
#         if max_real > 0:
#             turing_count += 1
    
#     # Statistics on VALID trials only
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
    
#     print(f"  Valid trials: {n_valid}/{n_trials}")
#     print(f"  Discarded:    {discarded_count} ({discard_rate:.1f}%)")
#     if n_valid > 0:
#         print(f"  Mean Re(λ):   {result['mean_eig']:.6f} ± {result['std_eig']:.6f}")
#         print(f"  Robustness:   {robustness:.1f}% ({turing_count}/{n_valid})")


# # Print summary table
# print("\n" + "="*70)
# print("SUMMARY TABLE")
# print("="*70)
# print(f"{'CV':<6} {'Mean Re(λ)':<14} {'Std':<12} {'Valid':<8} {'Discard%':<10} {'Robustness'}")
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

# # Save results to file
# output_data = {
#     'results': results_by_cv,
#     'baseline_params': baseline_params,
#     'hopping': hopping,
#     'n_trials': n_trials,
#     'config_id': CONFIG_TO_TEST,
#     'config_name': row['config_name']
# }

# output_file = f'3954_cv_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'

# with open(output_file, 'wb') as f:
#     pickle.dump(output_data, f)
