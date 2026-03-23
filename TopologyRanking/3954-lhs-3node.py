#!/usr/bin/env python
# coding: utf-8

# ## Type I - III diffusable nodes for 3954 topology using Latin Hypercube Sampling (LHS)

# **Diffusion Rates:**
# 
# **Type I**   
# DU, DV, DW = 
# 
# **Type II**  
# DU, DV, DW =  
# 
# **Type III**  
# DU, DV, DW = 
# 
# Then for the LHS/robustness comparison, scan d from 0.1 to 10 and measure what fraction of our stable samples still gives Turing instability at each d value.
# 
# We should see Type I collapse to zero as d → 1, while Type II and III stay robustly non-zero...

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from scipy.optimize import fsolve
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')


# In[40]:


# hill coefficient and hill functions

n = 2

def hill_activation(X, K):
    return X**n / (K**n + X**n)

def hill_inhibition(X, K):
    return K**n / (K**n + X**n)

def dH_act(x, K):
    return n * K**n * x**(n-1) / (K**n + x**n)**2

def dH_inh(x, K):
    return -n * K**n * x**(n-1) / (K**n + x**n)**2


# In[41]:


def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    du = alpha_u + beta_u * hill_activation(u, K_uu) * hill_inhibition(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w

    return [du, dv, dw]


# In[42]:


def find_steady_state(params, n_attempts=10):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        sol = fsolve(ode_system, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system(steady_state, params)
        if (ier == 1 and
            np.max(np.abs(residuals)) < 1e-8 and
            np.all(steady_state > 0)):
            return steady_state
    return None


# In[43]:


def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    J = np.zeros((3, 3))

    # Row 0: d(du)/d(u,v,w)
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[0, 2] = 0

    # Row 1: d(dv)/d(u,v,w)
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)

    # Row 2: d(dw)/d(u,v,w)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w

    return J


# In[44]:


def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)

def is_turing_diego(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.logspace(-1, 2, 100):          # 500 to 100 bc otherwise long runtimes, change to 500 once we use HPC
        M  = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        if a3 < 0 and a1 > 0 and a2 > 0:
            return True
    return False

# CHECK AGAIN LATER: WHAT IS THE DIFFERNECE EBTWEEN THOSE TWO???

# def is_turing_shaberi(J, DU, DV, DW):
#     D = np.diag([DU, DV, DW])
#     any_unstable = False
#     for k in np.linspace(0.1, 10, 100):       # extended to 100, check this again bc not sure this is correct!
#         M = J - k**2 * D
#         if np.max(np.real(eigvals(M))) > 0:
#             any_unstable = True
#             break
#     if not any_unstable:
#         return False
#     M_large = J - 100**2 * D                   # check at k=100, why dont we just say 10**2 like in the RMT code??
#     return np.all(np.real(eigvals(M_large)) < 0)


def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.linspace(0.1, 10, 100):
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            return True
    return False


# In[47]:


param_ranges = [
    (0.001, 0.1),  # alpha_U
    (0.1, 10),     # beta_U
    (0.01, 1),     # K_UU  
    (0.01, 1),     # K_VU 
    (0.01, 1),     # delta_U
    (0.001, 0.1),  # alpha_V
    (0.1, 10),     # beta_V
    (0.01, 1),     # K_UV
    (0.01, 1),     # K_WV
    (0.01, 1),     # delta_V
    (0.001, 0.1),  # alpha_W
    (0.1, 10),     # beta_W
    (0.01, 1),     # K_WW
    (0.01, 1),     # K_UW
    (0.01, 1),     # K_VW
    (0.01, 1),     # delta_W
]

n_samples = 100_000
DU, DV, DW = 10.0, 0.0, 1.0

sampler = qmc.LatinHypercube(d=16, seed=42)   
samples = sampler.random(n=n_samples)
params_log = np.zeros((n_samples, 16))  

for i in range(16):  
    log_min = np.log10(param_ranges[i][0])
    log_max = np.log10(param_ranges[i][1])
    params_log[:, i] = 10**(log_min + samples[:, i] * (log_max - log_min))


# In[48]:


steady_found   = 0
stable_count   = 0
turing_diego   = 0
turing_shaberi = 0

np.random.seed(42)
for i in range(n_samples):
    params = params_log[i]
    steady = find_steady_state(params)

    if steady is not None:
        steady_found += 1
        J = compute_jacobian(steady, params)

        if is_stable(J):
            stable_count += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

# print results
rob_diego   = 100 * turing_diego   / stable_count if stable_count > 0 else 0.0
rob_shaberi = 100 * turing_shaberi / stable_count if stable_count > 0 else 0.0

results_3954_lhs = {
    "n_samples":      n_samples,
    "steady_found":   steady_found,
    "stable":         stable_count,
    "diego":          turing_diego,
    "shaberi":        turing_shaberi,
    "rob_diego":      rob_diego,
    "rob_shaberi":    rob_shaberi,
}

print(f"{'Type':<10} {'Tested':<10} {'Steady':<10} {'Stable':<10} {'Diego_Tu':<10} {'Shaberi_Tu':<12} {'Diego_Ro':<12} {'Shaberi_Ro':<12}")
print("-" * 95)

print(f"{'Type X':<10} "
      f"{results_3954_lhs['n_samples']:<10,} "
      f"{results_3954_lhs['steady_found']:<10,} "
      f"{results_3954_lhs['stable']:<10,} "
      f"{results_3954_lhs['diego']:<10,} "
      f"{results_3954_lhs['shaberi']:<10,} "
      f"{results_3954_lhs['rob_diego']:>11.7f}% "
      f"{results_3954_lhs['rob_shaberi']:>11.7f}%")


# In[ ]:


print(results_3954_lhs)

# DU, DV, DW = 0.0, 1.0, 1.0
results_3954_lhs_backup_typex = {'n_samples': 100000, 'steady_found': 96942, 'stable': 95145, 'diego': 668, 'shaberi': 656, 'rob_diego': 0.7020862893478376, 'rob_shaberi': 0.6894739607966788}

# DU, DV, DW = 1.0, 1.0, 0.0
results_3954_lhs_backup_typexx = {'n_samples': 100000, 'steady_found': 96942, 'stable': 95145, 'diego': 75, 'shaberi': 76, 'rob_diego': 0.07882705344474224, 'rob_shaberi': 0.07987808082400547}

#DU, DV, DW = 1.0, 0.0, 1.0
results_3954_lhs_backup_typexxx = {'n_samples': 100000, 'steady_found': 96942, 'stable': 95145, 'diego': 2, 'shaberi': 2, 'rob_diego': 0.0021020547585264594, 'rob_shaberi': 0.0021020547585264594}

#DU, DV, DW = 1.0, 0.0, 10.0
results_3954_lhs_backup_typexxxx = {'n_samples': 100000, 'steady_found': 96942, 'stable': 95145, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}

#DU, DV, DW = 10.0, 0.0, 1.0
results_3954_lhs_backup_typexxxxx = {'n_samples': 100000, 'steady_found': 96942, 'stable': 95145, 'diego': 49, 'shaberi': 45, 'rob_diego': 0.05150034158389826, 'rob_shaberi': 0.04729623206684534}

# DU, DV, DW = 0.1, 0.0, 0.1
# results_3954_lhs_backup_typexxx =

# results_type1_3954_lhs_d10_backup =

#results_type1_3954_lhs_d5_backup = 

#results_type1_3954_lhs_d1_backup = 

#results_type2_3954_lhs_backup =

#results_type3_3954_lhs_backup =


# In[ ]:




