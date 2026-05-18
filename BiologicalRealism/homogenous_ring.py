#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve


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















baseline_params = np.array([
    0.1,   # alpha_u
    1.0,   # beta_u
    0.5,   # K_uu (u self-activation)
    0.5,   # K_vu (v inhibits u)
    0.8,   # delta_u
    
    0.1,   # alpha_v
    1.2,   # beta_v
    0.5,   # K_uv (u activates v)
    0.5,   # K_wv (w inhibits v)
    1.0,   # delta_v
    
    0.1,   # alpha_w
    1.0,   # beta_w
    0.5,   # K_ww (w self-activation)
    0.5,   # K_uw (u inhibits w)
    0.5,   # K_vw (v inhibits w)
    0.9    # delta_w
])


# hopping rates (diffusion combination that leads to highest robustness)
hopping = {
    'h_u': 1.0,
    'h_v': 1.0,
    'h_w': 0.0,
}





# TESTING THE FUNCTIONS

if __name__ == "__main__":

    # Find steady state
    steady_state = find_steady_state(baseline_params)
    
    if steady_state is None:
        print("\nERROR: Could not find steady state!")
    else:
        print(f"\nSteady state: u={steady_state[0]:.3f}, v={steady_state[1]:.3f}, w={steady_state[2]:.3f}")
        
        # Compute Jacobian
        J = compute_jacobian(steady_state, baseline_params)
        
        # Check stability
        eigs = np.linalg.eigvals(J)
        max_real = np.max(np.real(eigs))
        print(f"Max eigenvalue (no diffusion): {max_real:.6f}")
        print(f"Stable? {max_real < 0}")
        
        # Check Turing
        turing = is_turing_shaberi(J, eigs, hopping['h_u'], hopping['h_v'], hopping['h_w'])
        print(f"Turing type: {turing}")
        
        if turing == 'Type-I':
            print("\nSUCCESS! Baseline parameters work.")
        else:
            print("\nParameters don't work. Need to adjust.")