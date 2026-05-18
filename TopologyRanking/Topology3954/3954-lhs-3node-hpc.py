#!/usr/bin/env python3
import numpy as np
import sys
import pickle
import os
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import qmc
import heapq

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
    for k in np.arange(0.01, 10.01, 0.1):   
        M = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        
        if a1 > 0 and a2 > 0 and a3 < 0:
            return True
    return False



def is_turing_shaberi(J, eigs_0, DU, DV, DW):
    # STEP 1: Stability at k=0
    if np.max(np.real(eigs_0)) >= 0:
        return None
    
    # STEP 2: Check for instability with diffusion, # SUPPOSED TO BE 0.01 STEP, BUT INCREASED TO 0.1 FOR SPEED, CHANGE BACK LATER 
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.1)
    
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


# ######### NEW VERSIONS, which is not shaberi accurate so we use the old and accurate one! #########

# def is_turing_shaberi(J, eigs_0, DU, DV, DW, tol=1e-9):
# 	# STEP 1: Stability at k=0
# 	if np.max(np.real(eigs_0)) >= -tol:
# 		return None

# 	# STEP 2: Scan k, record dispersion
# 	D = np.diag([DU, DV, DW])
# 	k_values = np.arange(0.01, 10.01, 0.1)

# 	max_real = np.empty(len(k_values))
# 	imag_at_max = np.empty(len(k_values))

# 	for i, k in enumerate(k_values):
# 		eigs_k = np.linalg.eigvals(J - k**2 * D)
# 		idx = np.argmax(np.real(eigs_k))
# 		max_real[i] = np.real(eigs_k[idx])
# 		imag_at_max[i] = np.imag(eigs_k[idx])

# 	# No instability with diffusion
# 	if np.max(max_real) <= tol:
# 		return None

# 	peak_idx = int(np.argmax(max_real))

# 	# Exclude oscillatory instability
# 	if np.abs(imag_at_max[peak_idx]) > 1e-8:
# 		return "Hopf"

# 	# Type II if the maximum is at the high-k boundary
# 	if peak_idx >= len(k_values) - 2:
# 		return "Type-II"

# 	# Scholes-style Type I = Ia + Ib
# 	return "Type-I"



##############################################

# DIFFUSION CONFIGURATIONS
# EXTENDED VERSION

# DIFFUSION_CONFIGS = {
#     # DCC: A=Destable, B=Complementary, C=Complementary
#     0:  {"name": "LHS_3954_DCC_Type1",          "dU": 1.0,  "dV": 10.0, "dW": 10.0},
#     1:  {"name": "LHS_3954_DCC_Type1_Var1",     "dU": 1.0,  "dV": 0.0,  "dW": 10.0},
#     2:  {"name": "LHS_3954_DCC_Type1_Var2",     "dU": 1.0,  "dV": 10.0, "dW": 0.0},
#     3:  {"name": "LHS_3954_DCC_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    
#     4:  {"name": "LHS_3954_DCC_Type2_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
#     5:  {"name": "LHS_3954_DCC_Type2_Unequal1", "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
#     6:  {"name": "LHS_3954_DCC_Type2_Unequal2", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
#     7:  {"name": "LHS_3954_DCC_Type2_Unequal3", "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
#     8:  {"name": "LHS_3954_DCC_Type2_Unequal4", "dU": 1.0,  "dV": 10.0, "dW": 0.0},
#     # 9:  {"name": "LHS_3954_DCC_Type2_Var1",     "dU": 1.0,  "dV": 0.0,  "dW": 0.0}, # no 1,0,0 if not necessary bc freezing both leads to salt and pepper/turing filters 

#     9:  {"name": "LHS_3954_DCC_Type3_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
#     10: {"name": "LHS_3954_DCC_Type3_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
#     11: {"name": "LHS_3954_DCC_Type3_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 0.1},
#     12: {"name": "LHS_3954_DCC_Type3_Unequal3", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
#     13: {"name": "LHS_3954_DCC_Type3_Unequal4", "dU": 0.0,  "dV": 10.0, "dW": 1.0},
    
#     # CDD: A=Complementary, B=Destable, C=Destable
#     14: {"name": "LHS_3954_CDD_Type1",          "dU": 10.0, "dV": 1.0,  "dW": 1.0},
#     #15: {"name": "LHS_3954_CDD_Type1_Var1",     "dU": 10.0, "dV": 1.0,  "dW": 0.0}, # for type I all destabilising nodes have to be mobile
#     #16: {"name": "LHS_3954_CDD_Type1_Var2",     "dU": 10.0, "dV": 0.0,  "dW": 1.0}, # for type I all destabilising nodes have to be mobile
#     15: {"name": "LHS_3954_CDD_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},   
    
#     16: {"name": "LHS_3954_CDD_Type2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
#     17: {"name": "LHS_3954_CDD_Type2_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
#     18: {"name": "LHS_3954_CDD_Type2_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 0.1},
#     19: {"name": "LHS_3954_CDD_Type2_Unequal3", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
#     20: {"name": "LHS_3954_CDD_Type2_Unequal4", "dU": 0.0,  "dV": 10.0, "dW": 1.0},
    
#     21: {"name": "LHS_3954_CDD_Type3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
#     22: {"name": "LHS_3954_CDD_Type3_Unequal1", "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
#     23: {"name": "LHS_3954_CDD_Type3_Unequal2", "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
#     24: {"name": "LHS_3954_CDD_Type3_Unequal3", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
#     25: {"name": "LHS_3954_CDD_Type3_Unequal4", "dU": 1.0,  "dV": 10.0, "dW": 0.0},
#     # 29: {"name": "LHS_3954_CDD_Type3_Var1",     "dU": 1.0,  "dV": 0.0,  "dW": 0.0}, # this one is weirdly high, soo i took it out...leads to turing filters
#     26: {"name": "LHS_3954_CDD_Type3_Var1",     "dU": 1.0,  "dV": 0.0,  "dW": 1.0},
    
#     # CCD: A=Compl., B=Compl., C=Destable
#     27: {"name": "LHS_3954_CCD_Type1",          "dU": 10.0, "dV": 10.0, "dW": 1.0},
#     28: {"name": "LHS_3954_CCD_Type1_Var1",     "dU": 10.0, "dV": 0.0,  "dW": 1.0},
#     29: {"name": "LHS_3954_CCD_Type1_Var2",     "dU": 0.0,  "dV": 10.0, "dW": 1.0}, # this is really high so maybe take it out, makes type I higher -> LHS_3954_CCD_Type1_Var2,29,"{'dU': 0.0, 'dV': 10.0, 'dW': 1.0}",1000000,969615,951296,6758,6758,1,6746,11,0.6758,0.6758,0.0001
#     30: {"name": "LHS_3954_CCD_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    
#     31: {"name": "LHS_3954_CCD_Type2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
#     32: {"name": "LHS_3954_CCD_Type2_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
#     33: {"name": "LHS_3954_CCD_Type2_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 0.1},
#     34: {"name": "LHS_3954_CCD_Type2_Unequal3", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
#     35: {"name": "LHS_3954_CCD_Type2_Unequal4", "dU": 0.0,  "dV": 10.0, "dW": 1.0},
    
#     36: {"name": "LHS_3954_CCD_Type3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
#     37: {"name": "LHS_3954_CCD_Type3_Unequal1", "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
#     38: {"name": "LHS_3954_CCD_Type3_Unequal2", "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
#     39: {"name": "LHS_3954_CCD_Type3_Unequal3", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
#     40: {"name": "LHS_3954_CCD_Type3_Unequal4", "dU": 1.0,  "dV": 10.0, "dW": 0.0},

#     # DCI: A=Destable, B=Compl., C=Immobile
#     41: {"name": "LHS_3954_DCI_Type1",          "dU": 1.0,  "dV": 10.0, "dW": 0.0},
#     42: {"name": "LHS_3954_DCI_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 0.0},

#     43: {"name": "LHS_3954_DCI_Type2_Equal",    "dU": 1.0,  "dV": 0.0,  "dW": 0.0},
#     44: {"name": "LHS_3954_DCI_Type2_Unequal1", "dU": 0.1,  "dV": 0.0,  "dW": 0.0},
#     45: {"name": "LHS_3954_DCI_Type2_Unequal2", "dU": 10.0, "dV": 0.0,  "dW": 0.0},

#     46: {"name": "LHS_3954_DCI_Type3_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
#     47: {"name": "LHS_3954_DCI_Type3_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 0.0},
#     48: {"name": "LHS_3954_DCI_Type3_Unequal2", "dU": 0.0,  "dV": 10.0, "dW": 0.0},
# }



DIFFUSION_CONFIGS = {
    0:  {"name": "NEW_LHS_3954_Type1_V1_Equal",     "dU": 10.0, "dV": 1.0,  "dW": 1.0},
    1:  {"name": "NEW_LHS_3954_Type1_V1_Control",   "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    2:  {"name": "NEW_LHS_3954_Type1_V2_Equal",     "dU": 1.0,  "dV": 10.0, "dW": 1.0},
    3:  {"name": "NEW_LHS_3954_Type1_V3_Equal",     "dU": 1.0,  "dV": 1.0,  "dW": 10.0},
    4:  {"name": "NEW_LHS_3954_Type1_V4_Equal",     "dU": 1.0,  "dV": 10.0, "dW": 10.0},

    5:  {"name": "NEW_LHS_3954_Type2_V1_Equal",     "dU": 1.0,  "dV": 0.0,  "dW": 1.0},
    6:  {"name": "NEW_LHS_3954_Type2_V1_Unequal1",  "dU": 1.0,  "dV": 0.0,  "dW": 5.0},
    7:  {"name": "NEW_LHS_3954_Type2_V1_Unequal2",  "dU": 5.0,  "dV": 0.0,  "dW": 1.0},
    8:  {"name": "NEW_LHS_3954_Type2_V1_Unequal3",  "dU": 1.0,  "dV": 0.0,  "dW": 0.1},
    9:  {"name": "NEW_LHS_3954_Type2_V1_Unequal4",  "dU": 0.1,  "dV": 0.0,  "dW": 1.0},

    10: {"name": "NEW_LHS_3954_Type2_V2_Equal",     "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    11: {"name": "NEW_LHS_3954_Type2_V2_Unequal1",  "dU": 5.0,  "dV": 1.0,  "dW": 0.0},
    12: {"name": "NEW_LHS_3954_Type2_V2_Unequal2",  "dU": 1.0,  "dV": 5.0,  "dW": 0.0},
    13: {"name": "NEW_LHS_3954_Type2_V2_Unequal3",  "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
    14: {"name": "NEW_LHS_3954_Type2_V2_Unequal4",  "dU": 0.1,  "dV": 1.0,  "dW": 0.0},

    15: {"name": "NEW_LHS_3954_Type2_V3_Equal",     "dU": 0.0,  "dV": 1.0,  "dW": 1.0},

    16: {"name": "NEW_LHS_3954_Type3_V1_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
    17: {"name": "NEW_LHS_3954_Type3_V2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
    18: {"name": "NEW_LHS_3954_Type3_V3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},

    19: {"name": "NEW_LHS_3954_Type3_V4_Equal",     "dU": 1.0,  "dV": 0.0,  "dW": 0.0},
    20: {"name": "NEW_LHS_3954_Type3_V4_Unequal1",  "dU": 0.1,  "dV": 0.0,  "dW": 0.0},
    21: {"name": "NEW_LHS_3954_Type3_V4_Unequal2",  "dU": 10.0, "dV": 0.0,  "dW": 0.0},
}


# add one version where node W is always immobile so like DCI, also 100 i think is type 2 when node w is immobile but not sure could also be type 3 when we freeze the entire destabilising cycle so VW??




# MAIN ANALYSIS FUNCTION

def run_analysis(config_id, n_samples):
    
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
    
    # Heap to keep the N strongest Type-I hits (min-heap on max_re_eig)
    n_save = 5
    type_I_heap = []  # entries: (max_re_eig, sample_idx, params, steady_state)

    # Initialize counters
    steady_states = 0
    stable_without_diffusion = 0
    diego_turing = 0
    shaberi_total = 0
    shaberi_type_I = 0
    shaberi_type_II = 0
    shaberi_hopf = 0
    
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
                
                if turing_type is not None:
                    shaberi_total += 1
                    if turing_type == 'Type-I':
                        shaberi_type_I += 1

                        # Compute max Re(lambda) over dispersion for ranking
                        D = np.diag([DU, DV, DW])
                        max_re_eig = -np.inf
                        for k in np.arange(0.01, 10.01, 0.01):
                            eigs_k = np.linalg.eigvals(J - k**2 * D)
                            max_re = max(max_re, np.max(np.real(eigs_k)))
                        
                        # Keep best n_save
                        entry = (max_re, i, params.copy(), steady.copy())
                        if len(type_I_heap) < n_save:
                            heapq.heappush(type_I_heap, entry)
                        else:
                            heapq.heappushpop(type_I_heap, entry)

                    elif turing_type == 'Type-II':
                        shaberi_type_II += 1
                    elif turing_type == 'Hopf':
                        shaberi_hopf += 1
        
        # Progress tracking
        if (i + 1) % 100000 == 0:
            print(f"[{config_name}] {i+1:,}/{n_samples:,} | Stable: {stable_without_diffusion} | "
                  f"Diego: {diego_turing} | Shaberi: {shaberi_total}")
    
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
        "rob_diego": rob_diego,
        "rob_shaberi_total": rob_shaberi_total,
        "rob_shaberi_type_I": rob_shaberi_type_I,
    }

    # Sort heap descending (strongest first) and store
    type_I_hits = sorted(type_I_heap, key=lambda x: x[0], reverse=True)
    results["type_I_hits"] = [{"rank": rank, "max_re_eig": entry[0], "sample_idx": entry[1], "params": entry[2], "steady_state": entry[3]} for rank, entry in enumerate(type_I_hits)]
    
    return results

# ACTUAL HPC CODE TO RUN ALL CONFIGURATIONS

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_lhs_3node.py <config_id>")
        sys.exit(1)
    
    config_id = int(sys.argv[1])
    n_samples = 1_000_000  # 500K samples for Phase 1 and then increase to 1 million
    
    results = run_analysis(config_id, n_samples)
    
    # Save as pickle (for Python)
    output_pkl = f"results/{results['config_name']}_1mio.pkl"
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
        'rob_diego': results['rob_diego'],
        'rob_shaberi_total': results['rob_shaberi_total'],
        'rob_shaberi_type_I': results['rob_shaberi_type_I'],
    }
    output_csv = f"results/{results['config_name']}_1mio.csv"
    pd.DataFrame([results_flat]).to_csv(output_csv, index=False)
    

    # Save Type-I hits as CSV (parameter sets for Obj 2 spatial analysis)
    if results["type_I_hits"]:
        hits_df = pd.DataFrame([{
            'config_name': results['config_name'],
            'rank': h['rank'],
            'max_re_eig': h['max_re_eig'],
            'sample_idx': h['sample_idx'],
            **{f"p{i}": v for i, v in enumerate(h["params"])},
            "u_ss": h["steady_state"][0],
            "v_ss": h["steady_state"][1],
            "w_ss": h["steady_state"][2],
        } for h in results["type_I_hits"]])
        hits_df.to_csv(f"results/{results['config_name']}_type_I_hits.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPLETED: {results['config_name']}")
    print(f"{'='*70}")
    print(f"Diego Turing:    {results['diego_turing']} ({results['rob_diego']:.4f}%)")
    print(f"Shaberi Total:   {results['shaberi_total']} ({results['rob_shaberi_total']:.4f}%)")
    print(f"  Type-I:        {results['shaberi_type_I']}")
    print(f"  Type-II:       {results['shaberi_type_II']}")
    print(f"  Hopf:          {results['shaberi_hopf']}")
    print(f"\nSaved to:")
    print(f"  {output_pkl}")
    print(f"  {output_csv}")
    print(f"{'='*70}")


