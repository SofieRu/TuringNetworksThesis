#!/usr/bin/env python
# coding: utf-8

# ## Type I - III diffusable nodes based on the Diego et al. 2018 using and Latin Hypercube Sampling (LHS)

# **Diffusion Rates:**
# 
# **Type I**   
# DU, DV, DW = 1.0, 0.0, 10.0   (v immobile → this gives much higher values so lets do 1.0, 0, 1.0 ok so apparently we cant do that it has to be higher!!!!)  
# 
# **Type II**  
# DU, DV, DW = 1.0, 1.0, 0.0    (w immobile)  
# 
# **Type III**  
# DU, DV, DW = 0.0, 1.0, 1.0    (u immobile)
# 
# Then for the LHS/robustness comparison, scan d from 0.1 to 10 and measure what fraction of our stable samples still gives Turing instability at each d value.
# 
# We should see Type I collapse to zero as d → 1, while Type II and III stay robustly non-zero...

# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from scipy.optimize import fsolve
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')


# In[58]:


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


# ### Type I
# 
# _v is immobile, w and u > 1_

# In[39]:


def ode_system_type1(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    du = alpha_u + beta_u * hill_activation(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_activation(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    return [du, dv, dw]


# In[40]:


def find_steady_state(params, n_attempts=10):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        sol = fsolve(ode_system_type1, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system_type1(steady_state, params)
        if (ier == 1 and
            np.max(np.abs(residuals)) < 1e-8 and
            np.all(steady_state > 0)):
            return steady_state
    return None


# In[41]:


def compute_jacobian_type1(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    J = np.zeros((3, 3))
    J[0, 0] = -delta_u
    J[0, 1] = beta_u * dH_act(v, K_vu)
    J[0, 2] = 0
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_activation(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_act(w, K_wv)
    J[2, 0] = beta_w * dH_act(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = -delta_w
    return J


# In[42]:


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


# In[ ]:


param_ranges = [
    (0.001, 0.1),  # alpha_u
    (0.1,   10),   # beta_u
    (0.01,   1),   # K_vu
    (0.01,   1),   # delta_u
    (0.001, 0.1),  # alpha_v
    (0.1,   10),   # beta_v
    (0.01,   1),   # K_uv
    (0.01,   1),   # K_wv
    (0.01,   1),   # delta_v
    (0.001, 0.1),  # alpha_w
    (0.1,   10),   # beta_w
    (0.01,   1),   # K_uw
    (0.01,   1),   # K_vw
    (0.01,   1),   # delta_w
]

# FIND out wihch paper we got those values from! 
n_samples = 100_000
DU, DV, DW = 1.0, 0.0, 10.0

sampler = qmc.LatinHypercube(d=14, seed=42)
samples = sampler.random(n=n_samples)
params_log = np.zeros((n_samples, 14))

for i in range(14):
    log_min = np.log10(param_ranges[i][0])
    log_max = np.log10(param_ranges[i][1])
    params_log[:, i] = 10**(log_min + samples[:, i] * (log_max - log_min))


# In[44]:


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
        J = compute_jacobian_type1(steady, params)

        if is_stable(J):
            stable_count += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

# print results
rob_diego   = 100 * turing_diego   / stable_count if stable_count > 0 else 0.0
rob_shaberi = 100 * turing_shaberi / stable_count if stable_count > 0 else 0.0

results_type1_diego_lhs = {
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

print(f"{'Type 1':<10} "
      f"{results_type1_diego_lhs['n_samples']:<10,} "
      f"{results_type1_diego_lhs['steady_found']:<10,} "
      f"{results_type1_diego_lhs['stable']:<10,} "
      f"{results_type1_diego_lhs['diego']:<10,} "
      f"{results_type1_diego_lhs['shaberi']:<10,} "
      f"{results_type1_diego_lhs['rob_diego']:>11.7f}% "
      f"{results_type1_diego_lhs['rob_shaberi']:>11.7f}%")


# ### Type II
# 
# _w is immobile, u and v > 1_

# In[45]:


# Type II: v-w destabilising → v activates w (J[2,1] > 0)
#          u-v stabilising   → u inhibits v (J[1,0] < 0)

def ode_system_type2(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    du = alpha_u + beta_u * hill_activation(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_inhibition(u, K_uv) * hill_activation(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(u, K_uw) * hill_activation(v, K_vw) - delta_w * w
    return [du, dv, dw]


# In[46]:


def find_steady_state(params, n_attempts=10):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        sol = fsolve(ode_system_type2, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system_type2(steady_state, params)
        if (ier == 1 and
            np.max(np.abs(residuals)) < 1e-8 and
            np.all(steady_state > 0)):
            return steady_state
    return None


# In[47]:


def compute_jacobian_type2(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    J = np.zeros((3, 3))
    # Row 0: du/dt
    J[0, 0] = -delta_u
    J[0, 1] = beta_u * dH_act(v, K_vu)                              # v activates u
    J[0, 2] = 0
    # Row 1: dv/dt
    J[1, 0] = beta_v * dH_inh(u, K_uv) * hill_activation(w, K_wv)  # u inhibits v → J[1,0] < 0
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_inhibition(u, K_uv) * dH_act(w, K_wv)  # w activates v
    # Row 2: dw/dt
    J[2, 0] = beta_w * dH_act(u, K_uw) * hill_activation(v, K_vw)  # u activates w
    J[2, 1] = beta_w * hill_activation(u, K_uw) * dH_act(v, K_vw)  # v activates w → J[2,1] > 0
    J[2, 2] = -delta_w
    return J


# In[48]:


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

def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.linspace(0.1, 10, 100):
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            return True
    return False


# In[49]:


param_ranges = [
    (0.001, 0.1),  # alpha_u
    (0.1,   10),   # beta_u
    (0.01,   1),   # K_vu
    (0.01,   1),   # delta_u
    (0.001, 0.1),  # alpha_v
    (0.1,   10),   # beta_v
    (0.01,   1),   # K_uv
    (0.01,   1),   # K_wv
    (0.01,   1),   # delta_v
    (0.001, 0.1),  # alpha_w
    (0.1,   10),   # beta_w
    (0.01,   1),   # K_uw
    (0.01,   1),   # K_vw
    (0.01,   1),   # delta_w
]

n_samples = 100_000
DU, DV, DW = 1.0, 1.0, 0.0   # w immobile

sampler = qmc.LatinHypercube(d=14, seed=42)
samples = sampler.random(n=n_samples)
params_log = np.zeros((n_samples, 14))

for i in range(14):
    log_min = np.log10(param_ranges[i][0])
    log_max = np.log10(param_ranges[i][1])
    params_log[:, i] = 10**(log_min + samples[:, i] * (log_max - log_min))


# In[50]:


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
        J = compute_jacobian_type2(steady, params)

        if is_stable(J):
            stable_count += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

# print results
rob_diego   = 100 * turing_diego   / stable_count if stable_count > 0 else 0.0
rob_shaberi = 100 * turing_shaberi / stable_count if stable_count > 0 else 0.0

results_type2_diego_lhs = {
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

print(f"{'Type 2':<10} "
      f"{results_type2_diego_lhs['n_samples']:<10,} "
      f"{results_type2_diego_lhs['steady_found']:<10,} "
      f"{results_type2_diego_lhs['stable']:<10,} "
      f"{results_type2_diego_lhs['diego']:<10,} "
      f"{results_type2_diego_lhs['shaberi']:<10,} "
      f"{results_type2_diego_lhs['rob_diego']:>11.7f}% "
      f"{results_type2_diego_lhs['rob_shaberi']:>11.7f}%")


# ### Type III
# 
# _u is immobile, w and v are mobile, w > 0 !!_

# In[59]:


def ode_system_type3(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    du = alpha_u + beta_u * hill_activation(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_activation(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    return [du, dv, dw]


# In[60]:


def find_steady_state(params, n_attempts=10):
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3)
        sol = fsolve(ode_system_type3, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system_type3(steady_state, params)
        if (ier == 1 and
            np.max(np.abs(residuals)) < 1e-8 and
            np.all(steady_state > 0)):
            return steady_state
    return None


# In[61]:


def compute_jacobian_type3(state, params):
    u, v, w = state
    alpha_u, beta_u, K_vu, delta_u = params[0:4]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[4:9]
    alpha_w, beta_w, K_uw, K_vw, delta_w = params[9:14]
    J = np.zeros((3, 3))
    J[0, 0] = -delta_u
    J[0, 1] = beta_u * dH_act(v, K_vu)
    J[0, 2] = 0
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_activation(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_act(w, K_wv)
    J[2, 0] = beta_w * dH_act(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = -delta_w
    return J


# In[62]:


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

def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.linspace(0.1, 10, 100):
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            return True
    return False


# In[63]:


param_ranges = [
    (0.001, 0.1),  # alpha_u
    (0.1,   10),   # beta_u
    (0.01,   1),   # K_vu
    (0.01,   1),   # delta_u
    (0.001, 0.1),  # alpha_v
    (0.1,   10),   # beta_v
    (0.01,   1),   # K_uv
    (0.01,   1),   # K_wv
    (0.01,   1),   # delta_v
    (0.001, 0.1),  # alpha_w
    (0.1,   10),   # beta_w
    (0.01,   1),   # K_uw
    (0.01,   1),   # K_vw
    (0.01,   1),   # delta_w
]

n_samples = 1_000_000
DU, DV, DW = 0.0, 1.0, 1.0   # u immobile

sampler = qmc.LatinHypercube(d=14, seed=42)
samples = sampler.random(n=n_samples)
params_log = np.zeros((n_samples, 14))

for i in range(14):
    log_min = np.log10(param_ranges[i][0])
    log_max = np.log10(param_ranges[i][1])
    params_log[:, i] = 10**(log_min + samples[:, i] * (log_max - log_min))


# In[64]:


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
        J = compute_jacobian_type3(steady, params)

        if is_stable(J):
            stable_count += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

# print results
rob_diego   = 100 * turing_diego   / stable_count if stable_count > 0 else 0.0
rob_shaberi = 100 * turing_shaberi / stable_count if stable_count > 0 else 0.0

results_type3_diego_lhs = {
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

print(f"{'Type 3':<10} "
      f"{results_type3_diego_lhs['n_samples']:<10,} "
      f"{results_type3_diego_lhs['steady_found']:<10,} "
      f"{results_type3_diego_lhs['stable']:<10,} "
      f"{results_type3_diego_lhs['diego']:<10,} "
      f"{results_type3_diego_lhs['shaberi']:<10,} "
      f"{results_type3_diego_lhs['rob_diego']:>11.7f}% "
      f"{results_type3_diego_lhs['rob_shaberi']:>11.7f}%")

