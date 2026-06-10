#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import pickle

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

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
    # STEP 1: Stability at k=0
    if np.max(np.real(eigs_0)) >= 0:
        return None
    
    # STEP 2: Check for instability with diffusion, # SUPPOSED TO BE 0.01 STEP, BUT INCREASED TO 0.1 FOR SPEED, CHANGE BACK LATER 
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.01)  # Δk = 0.01 per Shaberi et al., before i had np.arange(0.01, 10.01, 0.1)
    
    has_instability = False
    is_oscillatory = False
    
    for k in k_values:
        M = J - k**2 * D
        eigs_k = np.linalg.eigvals(M)
        
        if np.max(np.real(eigs_k)) > 0:
            has_instability = True
            
            unstable_eigs = eigs_k[np.real(eigs_k) > 0]
            if np.any(np.abs(np.imag(unstable_eigs)) > 1e-8):
                is_oscillatory = True
                break
    
    if not has_instability:
        return None
    
    if is_oscillatory:
        return 'Hopf'
    
    # STEP 3: Check RESTABILIZATION (Shaberi's method)
    k_high_values = np.linspace(10, 50, 20)
    for k in k_high_values:
        M = J - k**2 * D
        eigs_k = np.linalg.eigvals(M)
        if np.max(np.real(eigs_k)) < 0:
            return 'Type-I'  # Restabilizes
    
    return 'Type-II'  # Doesn't restabilize


# PARAMETRS FOR HOMOGENOUS RING (EXAMPLE)

# config_name,config_id,dU,dV,dW,param_rank,max_growth_rate,alpha_u,beta_u,K_uu,K_vu,delta_u,alpha_v,beta_v,K_uv,K_wv,delta_v,alpha_w,beta_w,K_ww,K_uw,K_vw,delta_w,u_star,v_star,w_star
# NEW_LHS_3954_Type2_V2_Unequal3,13,1.0,0.1,0.0,1,0.142536425011953,0.007077433094880293,1.4462479436287767,0.09245163265235273,0.26723942500123793,0.2436592755887972,0.005889422851713758,8.605716327777886,0.09207924851486214,0.12961324370483618,0.7777816957281338,0.0038604827469243463,1.6570791267318754,0.030767002915064585,0.9136856426323211,0.2649648697111866,0.18465525821355344,0.11797954202781268,1.697547444477265,0.22705338297596256

# baseline_params = np.array([
#     # u parameters
#     0.007077433094880293,   # alpha_u
#     1.4462479436287767,     # beta_u
#     0.09245163265235273,    # K_uu
#     0.26723942500123793,    # K_vu
#     0.2436592755887972,     # delta_u
#     # v parameters
#     0.005889422851713758,   # alpha_v
#     8.605716327777886,      # beta_v
#     0.09207924851486214,    # K_uv
#     0.12961324370483618,    # K_wv
#     0.7777816957281338,     # delta_v
#     # w parameters
#     0.0038604827469243463,  # alpha_w
#     1.6570791267318754,     # beta_w
#     0.030767002915064585,   # K_ww
#     0.9136856426323211,     # K_uw
#     0.2649648697111866,     # K_vw
#     0.18465525821355344     # delta_w
# ])

# # hopping rates (diffusion combination that leads to highest robustness), can we also take unequal diffusion or do we jsut use 1,1,0 instead of 1,0.1,0???
# hopping = {
#     'h_u': 1.0,
#     'h_v': 0.1,
#     'h_w': 0.0,
# }

# # Known steady state (for verification)
# steady_state_expected = np.array([
#     0.11797954202781268,    # u*
#     1.697547444477265,      # v*
#     0.22705338297596256     # w*
# ])


# CHANGE THIS TO TEST DIFFERENT CONFIGS:
CONFIG_TO_TEST = 4 #13 is the highest for 3954 and 10 is a lot lower with 0.0359 max Type I --> COMPARE 13 and 2 or 4 !!! for thesis i think?
CONFIG_LABEL = "low"  # or "low" — change this once when you switch configs


# Load parameters from CSV
df_params = pd.read_csv('../TopologyRanking/Topology3954/3954_PREFINAL_lhs_results_parameters.csv')

# Get the best parameter set for this config
config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) & (df_params['param_rank'] == 1)]

if len(config_data) == 0:
    print(f"ERROR: No parameters found for config {CONFIG_TO_TEST}!")
    exit(1)

# Extract parameters
row = config_data.iloc[0]

baseline_params = np.array([
    row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
    row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
    row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
])

steady_state_expected = np.array([row['u_star'], row['v_star'], row['w_star']])

hopping = {
    'h_u': row['dU'],
    'h_v': row['dV'],
    'h_w': row['dW'],
}

print(f"Testing config {CONFIG_TO_TEST}: {row['config_name']}")
print(f"Max growth rate from Obj 1: {row['max_growth_rate']:.6f}")



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

print("\n" + "="*70)
print("STEP 4: MONTE CARLO - CV SWEEP")
print("="*70)

# Settings, CHANGE HERE FOR VARIATION
n_trials = 1000
N_cells = 10 # for sanity check run with N = 5, 10 and 20, 30??

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

# # OLD VERSION: Loop over CV values
# for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
#     print(f"\n{'='*70}")
#     print(f"CV = {CV}")
#     print(f"{'='*70}")
    
#     max_eigenvalues = []
#     turing_count = 0
    
#     # Monte Carlo loop for this CV
#     for trial in range(n_trials):
#         if CV == 0:
#             # Homogeneous case - all cells identical
#             J_ring = build_ring_jacobian_homogeneous(
#                 N_cells=N_cells,
#                 steady_state=steady_state_expected,
#                 params=baseline_params,
#                 hopping=hopping
#             )
#         else:
#             # Heterogeneous case
#             J_ring, steady_states_hetero, params_hetero = build_ring_jacobian_heterogeneous(
#                 N_cells=N_cells,
#                 baseline_params=baseline_params,
#                 hopping=hopping,
#                 CV=CV
#             )
        
#         # Get eigenvalues
#         eigs = np.linalg.eigvals(J_ring)
#         max_real = np.max(np.real(eigs))
        
#         max_eigenvalues.append(max_real)
        
#         if max_real > 0:
#             turing_count += 1
    
#     # Calculate statistics
#     max_eigenvalues = np.array(max_eigenvalues)
#     robustness = 100 * turing_count / n_trials
    
#     result = {
#         'CV': CV,
#         'mean_eig': np.mean(max_eigenvalues),
#         'std_eig': np.std(max_eigenvalues),
#         'median_eig': np.median(max_eigenvalues),
#         'min_eig': np.min(max_eigenvalues),
#         'max_eig': np.max(max_eigenvalues),
#         'turing_count': turing_count,
#         'robustness': robustness,
#         'all_eigenvalues': max_eigenvalues
#     }
#     results_by_cv.append(result)
    
#     print(f"  Mean Re(λ): {result['mean_eig']:.6f} ± {result['std_eig']:.6f}")
#     print(f"  Robustness: {robustness:.1f}% ({turing_count}/{n_trials})")


for CV in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    print(f"\n{'='*70}")
    print(f"CV = {CV}")
    print(f"{'='*70}")
    
    max_eigenvalues = []
    turing_count = 0
    discarded_count = 0
    
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
            
            if J_ring is None:
                discarded_count += 1
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

print(f"\nResults saved to: {output_file}")
print(f"Config: {row['config_name']}")


















# FUNCTION TO BUILD JACOBIAN FOR RING OF DIFFERENT CELLS (PARAMETER HETEROGENEITY)

# Parameters: 

# Build (3N)×(3N) Jacobian for ring with parameter heterogeneity
# - N_cells: number of cells (10)
# - baseline_params: mean parameter values (16-element array)
# - hopping: dict with h_u, h_v, h_w
# - CV: coefficient of variation (e.g., 0.1 = 10% variation)

# Returns:
# - J_ring: 30×30 Jacobian matrix
# - steady_states: list of 10 steady states (one per cell)
# - params_list: list of 10 parameter arrays (one per cell)

# OLD VERSION 
# def build_ring_jacobian_heterogeneous(N_cells, baseline_params, hopping, CV):

#     # Generate perturbed parameters for each cell
#     params_list = []
#     steady_states = []

#     # for i in range(N_cells):
#     #     # Multiplicative noise: param_i = baseline × (1 + ε)
#     #     # where ε ~ N(0, CV²)
#     #     noise = np.random.normal(0, CV, size=16)
#     #     params_i = baseline_params * (1 + noise)
        
#     #     # Make sure all parameters stay positive
#     #     params_i = np.maximum(params_i, 1e-6)
        
#     #     params_list.append(params_i)

#     for i in range(N_cells):
#         sigma = np.sqrt(np.log(1 + CV**2))
#         mu = -sigma**2 / 2  # Keeps mean at 1
#         noise_factors = np.random.lognormal(mu, sigma, size=16)
#         params_i = baseline_params * noise_factors
#         params_list.append(params_i)
    
#     # Find steady state for each cell
#     for i, params_i in enumerate(params_list):
#         ss_i = find_steady_state(params_i)  # NOW we use this!
        
#         if ss_i is None:
#             # If can't find steady state, use baseline as fallback -> HAVE TO FIX LATER NOT SURE WHICH WAY IS CORRECT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#             #ss_i = find_steady_state(baseline_params)

#             params_i = baseline_params.copy()
#             ss_i = steady_state_expected.copy()
        
#         steady_states.append(ss_i)
    
#     # Build the big Jacobian
#     h_u = hopping['h_u']
#     h_v = hopping['h_v']
#     h_w = hopping['h_w']
    
#     size = 3 * N_cells
#     J_ring = np.zeros((size, size))
    
#     for i in range(N_cells):
#         idx = 3 * i
        
#         # Local Jacobian (DIFFERENT for each cell now!)
#         J_local = compute_jacobian(steady_states[i], params_list[i])
#         J_ring[idx:idx+3, idx:idx+3] = J_local
        
#         # Hopping (same as before)
#         J_ring[idx, idx] -= 2*h_u
#         J_ring[idx+1, idx+1] -= 2*h_v
#         J_ring[idx+2, idx+2] -= 2*h_w
        
#         # Coupling
#         left = (i - 1) % N_cells
#         right = (i + 1) % N_cells
        
#         J_ring[idx, 3*left] += h_u
#         J_ring[idx+1, 3*left+1] += h_v
#         J_ring[idx+2, 3*left+2] += h_w
        
#         J_ring[idx, 3*right] += h_u
#         J_ring[idx+1, 3*right+1] += h_v
#         J_ring[idx+2, 3*right+2] += h_w
    
#     return J_ring, steady_states, params_list


# BACKUP FOR THE VERSION WHERE WE CAN DIRECTLY COMPARE THE DIFFERENT VARIATIONS (LINE PLOT)

# print("\n" + "="*70)
# print("STEP 4: MONTE CARLO - CV SWEEP (TRAJECTORY VERSION)")
# print("="*70)

# # Settings
# CV_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# n_trials = 1000
# N_cells = 10

# # Storage
# results_by_cv = {cv: {'eigenvalues': [], 'turing_count': 0} for cv in CV_values}
# trajectories = []  # NEW: Store each realization's trajectory across all CVs

# print(f"Running {n_trials} realizations across {len(CV_values)} CV values...")
# print("This will take ~10-15 minutes...\n")

# # OUTER LOOP: Each realization (same base noise pattern)
# for trial in range(n_trials):
    
#     # Generate ONE base noise pattern for this realization
#     base_noise = np.random.normal(0, 1, size=16)
    
#     trajectory = {'trial': trial, 'cv_values': [], 'eigenvalues': []}
    
#     # INNER LOOP: Scale this noise by each CV value
#     for CV in CV_values:
        
#         if CV == 0:
#             # Homogeneous - no noise
#             J_ring = build_ring_jacobian_homogeneous(
#                 N_cells=N_cells,
#                 steady_state=steady_state,
#                 params=baseline_params,
#                 hopping=hopping
#             )
#         else:
#             # Heterogeneous - scale base noise by CV
#             noise = base_noise * CV
            
#             # Apply noise to parameters for each cell
#             params_list = []
#             steady_states = []
            
#             for i in range(N_cells):
#                 params_i = baseline_params * (1 + noise)
#                 params_i = np.maximum(params_i, 1e-6)  # Keep positive
#                 params_list.append(params_i)
                
#                 ss_i = find_steady_state(params_i)
#                 if ss_i is None:
#                     ss_i = find_steady_state(baseline_params)
#                 steady_states.append(ss_i)
            
#             # Build heterogeneous Jacobian manually
#             h_u = hopping['h_u']
#             h_v = hopping['h_v']
#             h_w = hopping['h_w']
            
#             size = 3 * N_cells
#             J_ring = np.zeros((size, size))
            
#             for i in range(N_cells):
#                 idx = 3 * i
#                 J_local = compute_jacobian(steady_states[i], params_list[i])
#                 J_ring[idx:idx+3, idx:idx+3] = J_local
                
#                 J_ring[idx, idx] -= 2*h_u
#                 J_ring[idx+1, idx+1] -= 2*h_v
#                 J_ring[idx+2, idx+2] -= 2*h_w
                
#                 left = (i - 1) % N_cells
#                 right = (i + 1) % N_cells
                
#                 J_ring[idx, 3*left] += h_u
#                 J_ring[idx+1, 3*left+1] += h_v
#                 J_ring[idx+2, 3*left+2] += h_w
                
#                 J_ring[idx, 3*right] += h_u
#                 J_ring[idx+1, 3*right+1] += h_v
#                 J_ring[idx+2, 3*right+2] += h_w
        
#         # Get eigenvalues
#         eigs = np.linalg.eigvals(J_ring)
#         max_real = np.max(np.real(eigs))
        
#         # Store in results
#         results_by_cv[CV]['eigenvalues'].append(max_real)
#         if max_real > 0:
#             results_by_cv[CV]['turing_count'] += 1
        
#         # Store in trajectory
#         trajectory['cv_values'].append(CV)
#         trajectory['eigenvalues'].append(max_real)
    
#     trajectories.append(trajectory)
    
#     # Progress
#     if (trial + 1) % 100 == 0:
#         print(f"  Completed {trial+1}/{n_trials} realizations...")

# # Calculate statistics for each CV
# results_summary = []
# for CV in CV_values:
#     eigenvalues = np.array(results_by_cv[CV]['eigenvalues'])
#     turing_count = results_by_cv[CV]['turing_count']
    
#     result = {
#         'CV': CV,
#         'mean_eig': np.mean(eigenvalues),
#         'std_eig': np.std(eigenvalues),
#         'median_eig': np.median(eigenvalues),
#         'min_eig': np.min(eigenvalues),
#         'max_eig': np.max(eigenvalues),
#         'turing_count': turing_count,
#         'robustness': 100 * turing_count / n_trials,
#         'all_eigenvalues': eigenvalues
#     }
#     results_summary.append(result)
    
#     print(f"\nCV = {CV}:")
#     print(f"  Mean Re(λ): {result['mean_eig']:.6f} ± {result['std_eig']:.6f}")
#     print(f"  Robustness: {result['robustness']:.1f}% ({turing_count}/{n_trials})")

# # Print summary table
# print("\n" + "="*70)
# print("SUMMARY TABLE")
# print("="*70)
# print(f"{'CV':<6} {'Mean Re(λ)':<15} {'Std':<12} {'Robustness':<12} {'Turing Count'}")
# print("-"*70)

# for r in results_summary:
#     print(f"{r['CV']:<6.2f} {r['mean_eig']:<15.6f} {r['std_eig']:<12.6f} "
#           f"{r['robustness']:<12.1f} {r['turing_count']}/{n_trials}")

# print("="*70)

# # Save results
# import pickle
# output_data = {
#     'results': results_summary,
#     'trajectories': trajectories,  # NEW: Individual trajectories!
#     'baseline_params': baseline_params,
#     'hopping': hopping,
#     'n_trials': n_trials,
#     'config_id': 13,
#     'config_name': 'NEW_LHS_3954_Type2_V2_Unequal3'
# }

# with open('objective2_cv_sweep_results.pkl', 'wb') as f:
#     pickle.dump(output_data, f)

# print(f"\nResults saved to: objective2_cv_sweep_results.pkl")
# print("="*70)