#!/usr/bin/env python3
import numpy as np
import sys
import pickle
import os
from scipy.linalg import eig
from scipy.optimize import fsolve
from scipy.stats import qmc
import pandas as pd

# Create directories (for local testing)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

################ LHS ANALYSIS FOR TOPOLOGY #3954 ################

# HILL FUNCTIONS AND ODE SYSTEM

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

def find_steady_state(params, n_attempts=10):
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


# TURING DETECTION METHODS

def is_turing_diego(J, DU, DV, DW):
    a1_0 = -np.trace(J)
    a2_0 = (J[0,0]*J[1,1] - J[0,1]*J[1,0] +
            J[0,0]*J[2,2] - J[0,2]*J[2,0] +
            J[1,1]*J[2,2] - J[1,2]*J[2,1])
    a3_0 = -np.linalg.det(J)
    
    if not (a1_0 > 0 and a3_0 > 0 and a1_0*a2_0 - a3_0 > 0):
        return False
    
    # SUPPOSED TO BE 0.01 STEP, BUT INCREASED TO 0.1 FOR SPEED, CHANGE BACK LATER 
    D = np.diag([DU, DV, DW])
    for k in np.arange(0.01, 10.01, 0.01):   
        M = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        
        if a1 > 0 and a2 > 0 and a3 < 0:
            return True
    return False



# SHABERI 

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


# Filtering as Type I a and Ib???

# def is_turing_shaberi(J, eigs_0, DU, DV, DW):
#     # STEP 1: Homogeneous steady state must be stable
#     if np.max(np.real(eigs_0)) >= 0:
#         return None
    
#     # STEP 2: Sweep a wider range of k to ensure high-k behavior captures the asymptote
#     D = np.diag([DU, DV, DW])
#     k_values = np.arange(0.01, 10.01, 0.01)
    
#     max_reals = np.zeros(len(k_values))
#     has_complex_unstable = False
    
#     for i, k in enumerate(k_values):
#         M = J - (k**2) * D
#         eigs_k = np.linalg.eigvals(M)
#         max_reals[i] = np.max(np.real(eigs_k))
        
#         if max_reals[i] > 0:
#             unstable_eigs = eigs_k[np.real(eigs_k) > 0]
#             if np.any(np.abs(np.imag(unstable_eigs)) > 1e-8):
#                 has_complex_unstable = True
    
#     # If no instabilities are found across the spectrum, it's not a Turing network
#     if np.max(max_reals) <= 0:
#         return None
    
#     if has_complex_unstable:
#         return 'Hopf'
    
#     # --- FIXED SEGMENTATION LOGIC ---
#     max_idx = np.argmax(max_reals)
#     is_interior_peak = max_idx < (len(k_values) - 5) # Buffer for floating point noise
    
#     # If it has an interior peak (Type I behavior)
#     if is_interior_peak:
#         if max_reals[-1] < 0:
#             return 'Type-Ia'  # Re-stabilises below 0
#         else:
#             return 'Type-Ib'  # Fails to re-stabilise but has a definitive macro-peak
            
#     # If the maximum value is at the end of the range (k -> infinity)
#     else:
#         # To separate Type-II (blow up) from Filter (asymptotic saturation), 
#         # evaluate the derivative/slope at the end of the evaluated spectrum.
#         slope_at_end = max_reals[-1] - max_reals[-5]
        
#         if slope_at_end > 1e-3: 
#             return 'Type-II' # Growth rate continues skyrocketing towards infinity
#         else:
#             return 'Filter'   # Growth rate flatlines into a stable positive asymptote


##############################################

# DIFFUSION CONFIGURATIONS

# DIFFUSION_CONFIGS = {
#     0:  {"name": "NEW_LHS_3954_Type1_V1_Equal",     "dU": 10.0, "dV": 1.0,  "dW": 1.0},
#     1:  {"name": "NEW_LHS_3954_Type1_V1_Control",   "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
#     2:  {"name": "NEW_LHS_3954_Type1_V2_Equal",     "dU": 1.0,  "dV": 10.0, "dW": 1.0},
#     3:  {"name": "NEW_LHS_3954_Type1_V3_Equal",     "dU": 1.0,  "dV": 1.0,  "dW": 10.0},
#     4:  {"name": "NEW_LHS_3954_Type1_V4_Equal",     "dU": 1.0,  "dV": 10.0, "dW": 10.0},

#     5:  {"name": "NEW_LHS_3954_Type2_V1_Equal",     "dU": 1.0,  "dV": 0.0,  "dW": 1.0},
#     6:  {"name": "NEW_LHS_3954_Type2_V1_Unequal1",  "dU": 1.0,  "dV": 0.0,  "dW": 5.0},
#     7:  {"name": "NEW_LHS_3954_Type2_V1_Unequal2",  "dU": 5.0,  "dV": 0.0,  "dW": 1.0},
#     8:  {"name": "NEW_LHS_3954_Type2_V1_Unequal3",  "dU": 1.0,  "dV": 0.0,  "dW": 0.1},
#     9:  {"name": "NEW_LHS_3954_Type2_V1_Unequal4",  "dU": 0.1,  "dV": 0.0,  "dW": 1.0},

#     10: {"name": "NEW_LHS_3954_Type2_V2_Equal",     "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
#     11: {"name": "NEW_LHS_3954_Type2_V2_Unequal1",  "dU": 5.0,  "dV": 1.0,  "dW": 0.0},
#     12: {"name": "NEW_LHS_3954_Type2_V2_Unequal2",  "dU": 1.0,  "dV": 5.0,  "dW": 0.0},
#     13: {"name": "NEW_LHS_3954_Type2_V2_Unequal3",  "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
#     14: {"name": "NEW_LHS_3954_Type2_V2_Unequal4",  "dU": 0.1,  "dV": 1.0,  "dW": 0.0},

#     15: {"name": "NEW_LHS_3954_Type2_V3_Equal",     "dU": 0.0,  "dV": 1.0,  "dW": 1.0},

#     16: {"name": "NEW_LHS_3954_Type3_V1_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
#     17: {"name": "NEW_LHS_3954_Type3_V2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
#     18: {"name": "NEW_LHS_3954_Type3_V3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},

#     19: {"name": "NEW_LHS_3954_Type3_V4_Equal",     "dU": 1.0,  "dV": 0.0,  "dW": 0.0},
#     20: {"name": "NEW_LHS_3954_Type3_V4_Unequal1",  "dU": 0.1,  "dV": 0.0,  "dW": 0.0},
#     21: {"name": "NEW_LHS_3954_Type3_V4_Unequal2",  "dU": 10.0, "dV": 0.0,  "dW": 0.0},
# }


DIFFUSION_CONFIGS = {
    # TYPE 1
    0:  {"name": "FINAL_LHS_3954_Type1_Control",              "dU": 1.0,  "dV": 1.0,  "dW": 1.0},

    # node u diffuses faster than v and w
    1:  {"name": "FINAL_LHS_3954_Type1_UFast_Unequal1",       "dU": 10.0, "dV": 1.0,  "dW": 1.0},
    2:  {"name": "FINAL_LHS_3954_Type1_UFast_Unequal2",       "dU": 1.0,  "dV": 0.1,  "dW": 0.1},
    3:  {"name": "FINAL_LHS_3954_Type1_UFast_Unequal3",       "dU": 10.0, "dV": 0.1,  "dW": 1.0},
    4:  {"name": "FINAL_LHS_3954_Type1_UFast_Unequal4",       "dU": 10.0, "dV": 1.0,  "dW": 0.1},
    5:  {"name": "FINAL_LHS_3954_Type1_UFast_Unequal5",       "dU": 10.0, "dV": 0.1,  "dW": 0.1},

    # node v diffuses faster than u and w
    6:  {"name": "FINAL_LHS_3954_Type1_VFast_Unequal1",       "dU": 1.0,  "dV": 10.0, "dW": 1.0},
    7:  {"name": "FINAL_LHS_3954_Type1_VFast_Unequal2",       "dU": 0.1,  "dV": 1.0,  "dW": 0.1},
    8:  {"name": "FINAL_LHS_3954_Type1_VFast_Unequal3",       "dU": 0.1,  "dV": 10.0, "dW": 1.0},
    9:  {"name": "FINAL_LHS_3954_Type1_VFast_Unequal4",       "dU": 1.0,  "dV": 10.0, "dW": 0.1},
    10: {"name": "FINAL_LHS_3954_Type1_VFast_Unequal5",       "dU": 0.1,  "dV": 10.0, "dW": 0.1},
    
    # nodes v and w diffuse faster than u
    11: {"name": "FINAL_LHS_3954_Type1_VWFast_Unequal1",      "dU": 1.0,  "dV": 10.0, "dW": 10.0},
    12: {"name": "FINAL_LHS_3954_Type1_VWFast_Unequal2",      "dU": 0.1,  "dV": 1.0,  "dW": 1.0},
    13: {"name": "FINAL_LHS_3954_Type1_VWFast_Unequal3",      "dU": 0.1,  "dV": 10.0, "dW": 10.0},

    # nodes u and v diffuse faster than w
    14: {"name": "FINAL_LHS_3954_Type1_UVFast_Unequal1",      "dU": 10.0, "dV": 10.0, "dW": 1.0},
    15: {"name": "FINAL_LHS_3954_Type1_UVFast_Unequal2",      "dU": 1.0,  "dV": 1.0,  "dW": 0.1},
    16: {"name": "FINAL_LHS_3954_Type1_UVFast_Unequal3",      "dU": 10.0, "dV": 10.0, "dW": 0.1},

    # TYPE 2 
    # (node v is immobile)
    17: {"name": "FINAL_LHS_3954_Type2_VFreeze_Equal1",       "dU": 1.0,  "dV": 0.0,  "dW": 1.0},
    18: {"name": "FINAL_LHS_3954_Type2_VFreeze_Equal2",       "dU": 0.1,  "dV": 0.0,  "dW": 0.1},
    19: {"name": "FINAL_LHS_3954_Type2_VFreeze_Equal3",       "dU": 10.0, "dV": 0.0,  "dW": 10.0},
    20: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal1",     "dU": 0.1,  "dV": 0.0,  "dW": 1.0},
    21: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal2",     "dU": 1.0,  "dV": 0.0,  "dW": 0.1},
    22: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal3",     "dU": 0.1,  "dV": 0.0,  "dW": 10.0},
    23: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal4",     "dU": 10.0, "dV": 0.0,  "dW": 0.1},
    24: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal5",     "dU": 1.0,  "dV": 0.0,  "dW": 10.0},
    25: {"name": "FINAL_LHS_3954_Type2_VFreeze_Unequal6",     "dU": 10.0, "dV": 0.0,  "dW": 1.0},

    # TYPE 3 (freeze core destabilising nodes u and w)
    # node u immobile
    26: {"name": "FINAL_LHS_3954_Type3_UFreeze_Equal1",       "dU": 0.0, "dV": 1.0,  "dW": 1.0},
    27: {"name": "FINAL_LHS_3954_Type3_UFreeze_Equal2",       "dU": 0.0, "dV": 0.1,  "dW": 0.1},
    28: {"name": "FINAL_LHS_3954_Type3_UFreeze_Equal3",       "dU": 0.0, "dV": 10.0, "dW": 10.0},
    29: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal1",     "dU": 0.0, "dV": 1.0,  "dW": 0.1},
    30: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal2",     "dU": 0.0, "dV": 0.1,  "dW": 1.0},
    31: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal3",     "dU": 0.0, "dV": 10.0, "dW": 1.0},
    32: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal4",     "dU": 0.0, "dV": 1.0,  "dW": 10.0},
    33: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal5",     "dU": 0.0, "dV": 0.1,  "dW": 10.0},
    34: {"name": "FINAL_LHS_3954_Type3_UFreeze_Unequal6",     "dU": 0.0, "dV": 10.0, "dW": 0.1},

    # node w immobile
    35: {"name": "FINAL_LHS_3954_Type3_WFreeze_Equal1",       "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    36: {"name": "FINAL_LHS_3954_Type3_WFreeze_Equal2",       "dU": 0.1,  "dV": 0.1,  "dW": 0.0},
    37: {"name": "FINAL_LHS_3954_Type3_WFreeze_Equal3",       "dU": 10.0, "dV": 10.0, "dW": 0.0},

    38: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 2.0,  "dV": 1.0,  "dW": 0.0},
    39: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 3.0,  "dV": 1.0,  "dW": 0.0},
    40: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 4.0,  "dV": 1.0,  "dW": 0.0},
    41: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 1.0,  "dV": 2.0,  "dW": 0.0},
    42: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 1.0,  "dV": 3.0,  "dW": 0.0},
    43: {"name": "FINAL_LHS_3954_Type3_WFreeze_Lab1",         "dU": 1.0,  "dV": 4.0,  "dW": 0.0},

    44: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal1",     "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
    45: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal2",     "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
    46: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal3",     "dU": 10.0, "dV": 1.0,  "dW": 0.0},
    47: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal4",     "dU": 1.0,  "dV": 10.0, "dW": 0.0},
    48: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal5",     "dU": 0.1,  "dV": 10.0, "dW": 0.0},
    49: {"name": "FINAL_LHS_3954_Type3_WFreeze_Unequal6",     "dU": 10.0, "dV": 0.1,  "dW": 0.0},

    # node u + w immobile
    50: {"name": "FINAL_LHS_3954_Type3_UWFreeze_Equal1",      "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
    51: {"name": "FINAL_LHS_3954_Type3_UWFreeze_Equal2",      "dU": 0.0,  "dV": 0.1,  "dW": 0.0},
    52: {"name": "FINAL_LHS_3954_Type3_UWFreeze_Equal3",      "dU": 0.0,  "dV": 10.0, "dW": 0.0},

    # node v + w immobile
    53: {"name": "FINAL_LHS_3954_Type3_VWFreeze_Equal1",      "dU": 1.0,  "dV": 0.0, "dW": 0.0},
    54: {"name": "FINAL_LHS_3954_Type3_VWFreeze_Equal2",      "dU": 0.1,  "dV": 0.0, "dW": 0.0},
    55: {"name": "FINAL_LHS_3954_Type3_VWFreeze_Equal3",      "dU": 10.0, "dV": 0.0, "dW": 0.0},
}

# add one version where node W is always immobile so like DCI, also 100 i think is type 2 when node w is immobile but not sure could also be type 3 when we freeze the entire destabilising cycle so VW??

# MAIN ANALYSIS FUNCTION

def run_analysis(config_id, n_samples, save_successful_params=False, max_successful=100):
    
    config = DIFFUSION_CONFIGS[config_id]
    DU, DV, DW = config["dU"], config["dV"], config["dW"]
    config_name = config["name"]
    
    print(f"Starting {config_name}: dU={DU}, dV={DV}, dW={DW}, n_samples={n_samples:,}")
    
    # Parameter ranges
    param_ranges = [
        (0.001, 0.1), (0.1, 10), (0.01, 1), (0.01, 1), (0.01, 1),
        (0.001, 0.1), (0.1, 10), (0.01, 1), (0.01, 1), (0.01, 1),
        (0.001, 0.1), (0.1, 10), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1)
    ]
    
    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=16, seed=42)
    samples = sampler.random(n=n_samples)
    params_log = np.zeros((n_samples, 16))
    
    for i in range(16):
        log_min = np.log10(param_ranges[i][0])
        log_max = np.log10(param_ranges[i][1])
        params_log[:, i] = 10**(log_min + samples[:, i] * (log_max - log_min))
    
    # Initialize counters
    steady_states = 0
    stable_without_diffusion = 0
    diego_turing = 0
    shaberi_total = 0
    shaberi_type_I = 0
    shaberi_type_II = 0
    shaberi_hopf = 0
    filter_count = 0

    # NEW: List to collect ALL successful parameters
    successful_params = [] if save_successful_params else None
    
    # Main loop
    np.random.seed(42)
    for i in range(n_samples):
        params = params_log[i]
        steady = find_steady_state(params)
        
        if steady is not None:
            steady_states += 1
            J = compute_jacobian(steady, params)
            
            # Compute eigenvalues ONCE
            eigs_0 = np.linalg.eigvals(J)
            
            if np.max(np.real(eigs_0)) < 0:
                stable_without_diffusion += 1
                
                # Diego (uses trace/det only)
                if is_turing_diego(J, DU, DV, DW):
                    diego_turing += 1
                
                # Shaberi (reuses eigs_0)
                turing_type = is_turing_shaberi(J, eigs_0, DU, DV, DW)
                
                if turing_type is not None: # NEW, do not count Hopf as Turing for Shaberi
                    if turing_type == 'Hopf':
                        shaberi_hopf += 1
                    else:
                        shaberi_total += 1
                        if turing_type == 'Type-I':
                            shaberi_type_I += 1
                        elif turing_type == 'Type-II':
                            shaberi_type_II += 1
                        elif turing_type == 'Filter':
                            filter_count += 1
                    
                    # SAVE ALL classified parameters (regardless of type)
                    if save_successful_params:
                        D = np.diag([DU, DV, DW])
                        max_growth_rate = -np.inf

                        for k in np.arange(0.01, 10.01, 0.01):  # finer step for accuracy
                            M = J - k**2 * D
                            eigs_k = np.linalg.eigvals(M)
                            max_real_k = np.max(np.real(eigs_k))
                            if max_real_k > max_growth_rate:
                                max_growth_rate = max_real_k

                        successful_params.append({'params_array': params.copy(),'steady_state': steady.copy(),'max_growth_rate': float(max_growth_rate),'classification': turing_type,})

        # Progress tracking
        if (i + 1) % 100000 == 0:
            print(f"[{config_name}] {i+1:,}/{n_samples:,} | Stable: {stable_without_diffusion} | "
                  f"Diego: {diego_turing} | Shaberi: {shaberi_total}")
    
    # NEW: After loop, select BEST parameter sets (most stable)
    if save_successful_params and len(successful_params) > 0:
        successful_params.sort(key=lambda x: x['max_growth_rate'], reverse=True)  # Sort by most negative max growth rate
        #successful_params = successful_params[:max_successful]  # CHANGE Keep only top N

    # Calculate robustness
    rob_diego = 100 * diego_turing / n_samples
    rob_shaberi_total = 100 * shaberi_total / n_samples
    rob_shaberi_type_I = 100 * shaberi_type_I / n_samples

    results = {
        "config_name": config_name,
        "config_id": config_id,
        "diffusion": {"dU": DU, "dV": DV, "dW": DW},
        "n_samples": n_samples,
        "steady_states": steady_states,
        "stable_without_diffusion": stable_without_diffusion,
        "diego_turing": diego_turing,
        "shaberi_total": shaberi_total,
        "shaberi_type_I": shaberi_type_I,
        "shaberi_type_II": shaberi_type_II,
        "shaberi_hopf": shaberi_hopf,
        "filter_count": filter_count,
        "rob_diego": rob_diego,
        "rob_shaberi_total": rob_shaberi_total,
        "rob_shaberi_type_I": rob_shaberi_type_I,
    }

    # NEW: Add successful parameters if they were saved
    if save_successful_params and successful_params:
        results['successful_params'] = successful_params
        results['n_successful_saved'] = len(successful_params)

    return results

# ACTUAL HPC CODE TO RUN ALL CONFIGURATIONS

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 3954-lhs-3node-hpc.py <config_id> [--save-params] [--n-to-save=N]")
        sys.exit(1)
    
    config_id = int(sys.argv[1])
    n_samples = 1_000_000  # 1 million samples
    
    # NEW: Check for parameter saving flags
    save_successful_params = '--save-params' in sys.argv
    max_successful = 100  # Default to saving top 2 parameter sets

    for arg in sys.argv:
        if arg.startswith('--n-to-save='):
            max_successful = int(arg.split('=')[1])
    
    results = run_analysis(config_id, n_samples, save_successful_params, max_successful)
    
    # Save as pickle (for Python)
    output_pkl = f"results/{results['config_name']}_1mio_with_params.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    # Save as CSV (for Excel)
    results_flat = {
        'config_name': results['config_name'],
        'config_id': results['config_id'],
        'dU': results['diffusion']['dU'],
        'dV': results['diffusion']['dV'],
        'dW': results['diffusion']['dW'],
        'n_samples': results['n_samples'],
        'steady_states': results['steady_states'],
        'stable_without_diffusion': results['stable_without_diffusion'],
        'diego_turing': results['diego_turing'],
        'shaberi_total': results['shaberi_total'],
        'shaberi_type_I': results['shaberi_type_I'],
        'shaberi_type_II': results['shaberi_type_II'],
        'shaberi_hopf': results['shaberi_hopf'],
        'filter_count': results['filter_count'],
        'rob_diego': results['rob_diego'],
        'rob_shaberi_total': results['rob_shaberi_total'],
        'rob_shaberi_type_I': results['rob_shaberi_type_I'],
    }
    output_csv = f"results/{results['config_name']}_1mio_with_params.csv"
    pd.DataFrame([results_flat]).to_csv(output_csv, index=False)

    # Print summary
    print(f"Diego Turing:    {results['diego_turing']} ({results['rob_diego']:.4f}%)")
    print(f"Shaberi Total:   {results['shaberi_total']} ({results['rob_shaberi_total']:.4f}%)")
    print(f"  Type-I:        {results['shaberi_type_I']}")
    print(f"  Type-II:       {results['shaberi_type_II']}")
    print(f"  Hopf:          {results['shaberi_hopf']}")
    print(f"  Filter:        {results['filter_count']}")
    print(f"\nSaved to:")
    print(f"  {output_pkl}")
    print(f"  {output_csv}")