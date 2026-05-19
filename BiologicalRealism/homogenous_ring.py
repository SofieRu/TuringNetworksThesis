#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve

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


# PARAMETRS FOR HOMOGENOUS RING

# config_name,config_id,dU,dV,dW,param_rank,max_growth_rate,alpha_u,beta_u,K_uu,K_vu,delta_u,alpha_v,beta_v,K_uv,K_wv,delta_v,alpha_w,beta_w,K_ww,K_uw,K_vw,delta_w,u_star,v_star,w_star
# NEW_LHS_3954_Type2_V2_Unequal3,13,1.0,0.1,0.0,1,0.142536425011953,0.007077433094880293,1.4462479436287767,0.09245163265235273,0.26723942500123793,0.2436592755887972,0.005889422851713758,8.605716327777886,0.09207924851486214,0.12961324370483618,0.7777816957281338,0.0038604827469243463,1.6570791267318754,0.030767002915064585,0.9136856426323211,0.2649648697111866,0.18465525821355344,0.11797954202781268,1.697547444477265,0.22705338297596256

baseline_params = np.array([
    # u parameters
    0.007077433094880293,   # alpha_u
    1.4462479436287767,     # beta_u
    0.09245163265235273,    # K_uu
    0.26723942500123793,    # K_vu
    0.2436592755887972,     # delta_u
    
    # v parameters
    0.005889422851713758,   # alpha_v
    8.605716327777886,      # beta_v
    0.09207924851486214,    # K_uv
    0.12961324370483618,    # K_wv
    0.7777816957281338,     # delta_v
    
    # w parameters
    0.0038604827469243463,  # alpha_w
    1.6570791267318754,     # beta_w
    0.030767002915064585,   # K_ww
    0.9136856426323211,     # K_uw
    0.2649648697111866,     # K_vw
    0.18465525821355344     # delta_w
])


# hopping rates (diffusion combination that leads to highest robustness), can we also take unequal diffusion or do we jsut use 1,1,0 instead of 1,0.1,0???
hopping = {
    'h_u': 1.0,
    'h_v': 0.1,
    'h_w': 0.0,
}

# Known steady state (for verification)
steady_state_expected = np.array([
    0.11797954202781268,    # u*
    1.697547444477265,      # v*
    0.22705338297596256     # w*
])



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



# TESTING THE FUNCTIONS

residuals = ode_system(steady_state_expected, baseline_params)
print("\nSTEP 1: Check if we get Turing instability from single cell Jacobian")
print(f"Residuals: {np.max(np.abs(residuals)):.2e} (should be ~0)")

# Compute Jacobian at THIS steady state
J = compute_jacobian(steady_state_expected, baseline_params)

# Check stability
eigs = np.linalg.eigvals(J)
print(f"Max eigenvalue: {np.max(np.real(eigs)):.6f}")
print(f"Stable? {np.max(np.real(eigs)) < 0}")

# Check Turing
turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])
print(f"Turing? {turing}")


if turing == 'Type-I':
    print("\nSTEP 2: Building homogeneous ring")
    
    N_cells = 10
    
    # Build ring Jacobian (use known steady state)
    J_ring = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
    
    print(f"Ring Jacobian size: {J_ring.shape}")
    
    # Check eigenvalues
    eigs_ring = np.linalg.eigvals(J_ring)
    max_real_ring = np.max(np.real(eigs_ring))
    
    print(f"Max eigenvalue (ring): {max_real_ring:.6f}")
    print(f"Ring shows instability? {max_real_ring > 0}")
    
    if max_real_ring > 0:
        print("\nSUCCESS! Ring shows Turing instability!")
    else:
        print("\nRing is stable (no instability)")