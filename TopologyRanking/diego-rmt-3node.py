#!/usr/bin/env python
# coding: utf-8

# ## Type I - III diffusable nodes based on the Diego et al. 2018 using and Random matrix theory (RMT)

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
# We should see Type I collapse to zero as d → 1, while Type II and III stay robustly non-zero...

# In[42]:


import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.linalg import eig


# In[43]:


# Adjacency Matrix

#       u  v  w
#     ┌─────────┐
#  u  │ 0  1  0 │  
#  v  │ 1  0  1 │
#  w  │ 1  1  1 │
#     └─────────┘

# The destabilizing module is the u-v mutual activation cycle: fuv * fvu > 0, this is the positive feedback loop that drives instability.


# In[44]:


adjacency_matrix = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])

# the diagonal of the adjacency should always be 0
# the w self-loop [2,2] is already handled by J = G - I giving diagonal = -1.
# self-decay handled by J = G - I
# adjacency_matrix[i,j] = 1 means species j affects species i


# ### Type I
# 
# _v is immobile, w and u > 1_

# In[45]:


# apply sign constraints for specific topology and type (activating (+) or inhibiting (-))
# for type I, the u-v cycle is destabilising and the v-w cycle is stabilising, so we have:

def sign_constraints(J):
    J[0, 1] =  abs(J[0, 1]) # v activates u, u-v destabilising
    J[1, 0] =  abs(J[1, 0]) # u activates v
    J[1, 2] =  abs(J[1, 2]) # w activates v
    J[2, 1] = -abs(J[2, 1]) # v inhibits w, v-w cycle must be stabilising → edges opposite sign
    # J[2, 0] unconstrained

    return J

# template:
# J[i, j] =  abs(J[i, j])   # j activates i
# J[i, j] = -abs(J[i, j])   # j inhibits i
# leave unconstrained edges untouched

# so we don't do: J = G - np.eye(3), because it has no sign constraints (want to enforce the sign constraints for the specific topology)


# In[46]:


def generate_jacobian_type1(sigma):

    # random matrix
    G = np.random.normal(0, sigma, (3, 3))
    np.fill_diagonal(G, 0)

    # J = G - I  (rmt convention, may 1972), diagonal becomes -1, self-decay handled by J = G - I, off-diagonal from N(0, sigma)
    J = G - np.eye(3)

    # apply sparsity mask after sampling from adjacency matrix, but only to off-diagonal elements, the diagonal is already -1 from J = G - I
    for i in range(3): 
        for j in range(3):
            if i != j and adjacency_matrix[i, j] == 0:
                J[i, j] = 0

    J = sign_constraints(J)

    return J


# In[47]:


def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)


# In[48]:


# check turing instabilitiy: does diffusion destabilise a mode that was stable without diffusion?

# two ways to check turing stability:
# 1. shaberi et al (2025)
#    - compute all eigenvalues and check if real part < 0
#    - detects any instability, including oscillatory ones (complex eigenvalues)
#    - detects Turing I only (defined wavelength, restabilises at large k)
# 2. diego et al (2018) 
#    - characteristic polynomial and Routh-Hurwitz criteria
#    - detects only stationary instabilities, not oscillary ones
#    - condition: a3 < 0 AND a1 > 0 AND a2 > 0 for some k > 0.

# CHECK AGAIN LATER: WHAT IS THE DIFFERNECE EBTWEEN THOSE TWO???

# def is_turing_shaberi(J, DU, DV, DW):
#     D = np.diag([DU, DV, DW])
#     any_unstable = False
#     for k in np.linspace(0.1, 100, 100):                        # changed np.arange(0.1, 10, 100) to np.linspace(0.1, 10, 100) bc long run time and then i changed it to 0.1, 100,100 to check larger k values bc shaberi should give us more instabilities than diego, but it was giving us the same number, so maybe we need to check larger k values to see the restabilisation at large k for turing I instabilities
#         M = J - k**2 * D
#         if np.max(np.real(eigvals(M))) > 0:
#             any_unstable = True
#             break

#     if not any_unstable:
#         return False

#     M_large = J - 10**2 * D
#     return np.all(np.real(eigvals(M_large)) < 0)

def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.linspace(0.1, 10, 100):
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            return True
    return False

def is_turing_diego(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.logspace(-1, 2, 100):                          # changed (-1, 2, 500) to (-1, 2, 100) bc long run time
        M  = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        if a3 < 0 and a1 > 0 and a2 > 0:
            return True
    return False


# In[49]:


# type I specifications

n_samples = 100_000
sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#sigma = [0.6, 0.7]
DU, DV, DW = 1.0, 0.0, 1.0 # try to vary DW to see if we get a different robustness score


# In[50]:


results_type1_diego_rmt = []

for sig in sigma:
    np.random.seed(42)
    stable = 0
    turing_diego = 0
    turing_shaberi = 0

    for _ in range(n_samples):
        J = generate_jacobian_type1(sig)
        if is_stable(J):
            stable += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

    rob_diego   = 100 * turing_diego   / stable if stable > 0 else 0.0
    rob_shaberi = 100 * turing_shaberi / stable if stable > 0 else 0.0

    results_type1_diego_rmt.append({
        "sigma":        sig,
        "stable":       stable,
        "diego":        turing_diego,
        "shaberi":      turing_shaberi,
        "rob_diego":    rob_diego,
        "rob_shaberi":  rob_shaberi,
    })

print(f"{'Sigma':<8} {'Tested':>8} {'Stable':>8} {'Diego_Tu':>10} {'Shaberi_Tu':>12} {'Diego_Ro':>11} {'Shaberi_Ro':>14}")
print("-" * 80)

for r in results_type1_diego_rmt:
    print(f"{r['sigma']:<6.1f} {n_samples:>10,} {r['stable']:>7,} {r['diego']:>8,} {r['shaberi']:>11,} "
          f"{r['rob_diego']:>14.7f}% {r['rob_shaberi']:>14.7f}%")


# In[53]:


results_type1_diego_rmt_d10_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99954, 'diego': 1, 'shaberi': 1, 'rob_diego': 0.0010004602116973809, 'rob_shaberi': 0.0010004602116973809}, {'sigma': 0.5, 'stable': 99408, 'diego': 19, 'shaberi': 18, 'rob_diego': 0.0191131498470948, 'rob_shaberi': 0.018107194591984548}, {'sigma': 0.6, 'stable': 97740, 'diego': 104, 'shaberi': 101, 'rob_diego': 0.10640474728872519, 'rob_shaberi': 0.1033353795784735}, {'sigma': 0.7, 'stable': 94897, 'diego': 320, 'shaberi': 315, 'rob_diego': 0.33720770941125644, 'rob_shaberi': 0.3319388389517055}, {'sigma': 0.8, 'stable': 91371, 'diego': 697, 'shaberi': 690, 'rob_diego': 0.7628240907946723, 'rob_shaberi': 0.7551630167120859}, {'sigma': 0.9, 'stable': 87394, 'diego': 1219, 'shaberi': 1203, 'rob_diego': 1.3948325972034694, 'rob_shaberi': 1.376524704213104}, {'sigma': 1.0, 'stable': 83442, 'diego': 1851, 'shaberi': 1842, 'rob_diego': 2.218307327245272, 'rob_shaberi': 2.2075213921046957}, {'sigma': 1.1, 'stable': 79613, 'diego': 2445, 'shaberi': 2432, 'rob_diego': 3.071106477585319, 'rob_shaberi': 3.0547774860889554}, {'sigma': 1.2, 'stable': 75997, 'diego': 3103, 'shaberi': 3088, 'rob_diego': 4.083055910101715, 'rob_shaberi': 4.063318288879824}, {'sigma': 1.3, 'stable': 72634, 'diego': 3666, 'shaberi': 3650, 'rob_diego': 5.0472230635790405, 'rob_shaberi': 5.025194812346835}, {'sigma': 1.4, 'stable': 69373, 'diego': 4136, 'shaberi': 4121, 'rob_diego': 5.961973678520462, 'rob_shaberi': 5.9403514335548415}, {'sigma': 1.5, 'stable': 66232, 'diego': 4512, 'shaberi': 4496, 'rob_diego': 6.812416958569876, 'rob_shaberi': 6.788259451624592}, {'sigma': 1.6, 'stable': 63419, 'diego': 4862, 'shaberi': 4845, 'rob_diego': 7.666472192875952, 'rob_shaberi': 7.639666346047714}, {'sigma': 1.7, 'stable': 60806, 'diego': 5217, 'shaberi': 5199, 'rob_diego': 8.579745419859883, 'rob_shaberi': 8.550143077985725}, {'sigma': 1.8, 'stable': 58348, 'diego': 5504, 'shaberi': 5483, 'rob_diego': 9.433056831425242, 'rob_shaberi': 9.397065880578598}, {'sigma': 1.9, 'stable': 56014, 'diego': 5681, 'shaberi': 5663, 'rob_diego': 10.14210733031028, 'rob_shaberi': 10.109972506873282}, {'sigma': 2.0, 'stable': 53775, 'diego': 5859, 'shaberi': 5850, 'rob_diego': 10.895397489539748, 'rob_shaberi': 10.878661087866108}]

results_type1_diego_rmt_d5_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99954, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.5, 'stable': 99408, 'diego': 4, 'shaberi': 4, 'rob_diego': 0.004023821020441011, 'rob_shaberi': 0.004023821020441011}, {'sigma': 0.6, 'stable': 97740, 'diego': 32, 'shaberi': 31, 'rob_diego': 0.032739922242684676, 'rob_shaberi': 0.03171679967260078}, {'sigma': 0.7, 'stable': 94897, 'diego': 118, 'shaberi': 116, 'rob_diego': 0.1243453428454008, 'rob_shaberi': 0.12223779466158045}, {'sigma': 0.8, 'stable': 91371, 'diego': 280, 'shaberi': 279, 'rob_diego': 0.30644296330345516, 'rob_shaberi': 0.3053485241487999}, {'sigma': 0.9, 'stable': 87394, 'diego': 561, 'shaberi': 553, 'rob_diego': 0.6419204979746893, 'rob_shaberi': 0.6327665514795066}, {'sigma': 1.0, 'stable': 83442, 'diego': 861, 'shaberi': 856, 'rob_diego': 1.0318544617818366, 'rob_shaberi': 1.0258622755926272}, {'sigma': 1.1, 'stable': 79613, 'diego': 1167, 'shaberi': 1157, 'rob_diego': 1.4658410058658762, 'rob_shaberi': 1.4532802431763656}, {'sigma': 1.2, 'stable': 75997, 'diego': 1583, 'shaberi': 1577, 'rob_diego': 2.082976959616827, 'rob_shaberi': 2.0750819111280707}, {'sigma': 1.3, 'stable': 72634, 'diego': 1896, 'shaberi': 1887, 'rob_diego': 2.6103477710163285, 'rob_shaberi': 2.597956879698213}, {'sigma': 1.4, 'stable': 69373, 'diego': 2110, 'shaberi': 2101, 'rob_diego': 3.0415291251639687, 'rob_shaberi': 3.028555778184596}, {'sigma': 1.5, 'stable': 66232, 'diego': 2320, 'shaberi': 2307, 'rob_diego': 3.5028385070660706, 'rob_shaberi': 3.483210532673028}, {'sigma': 1.6, 'stable': 63419, 'diego': 2524, 'shaberi': 2515, 'rob_diego': 3.979879846733629, 'rob_shaberi': 3.965688516059856}, {'sigma': 1.7, 'stable': 60806, 'diego': 2753, 'shaberi': 2743, 'rob_diego': 4.52751373219748, 'rob_shaberi': 4.511067986711837}, {'sigma': 1.8, 'stable': 58348, 'diego': 2877, 'shaberi': 2870, 'rob_diego': 4.930760265990266, 'rob_shaberi': 4.918763282374718}, {'sigma': 1.9, 'stable': 56014, 'diego': 2973, 'shaberi': 2963, 'rob_diego': 5.3076016710108185, 'rob_shaberi': 5.289748991323598}, {'sigma': 2.0, 'stable': 53775, 'diego': 3078, 'shaberi': 3071, 'rob_diego': 5.7238493723849375, 'rob_shaberi': 5.710832171083217}]

results_type1_diego_rmt_d1_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99954, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.5, 'stable': 99408, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.6, 'stable': 97740, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.7, 'stable': 94897, 'diego': 2, 'shaberi': 2, 'rob_diego': 0.0021075481838203525, 'rob_shaberi': 0.0021075481838203525}, {'sigma': 0.8, 'stable': 91371, 'diego': 1, 'shaberi': 1, 'rob_diego': 0.0010944391546551969, 'rob_shaberi': 0.0010944391546551969}, {'sigma': 0.9, 'stable': 87394, 'diego': 3, 'shaberi': 3, 'rob_diego': 0.003432729935693526, 'rob_shaberi': 0.003432729935693526}, {'sigma': 1.0, 'stable': 83442, 'diego': 10, 'shaberi': 10, 'rob_diego': 0.011984372378418541, 'rob_shaberi': 0.011984372378418541}, {'sigma': 1.1, 'stable': 79613, 'diego': 15, 'shaberi': 13, 'rob_diego': 0.01884114403426576, 'rob_shaberi': 0.01632899149636366}, {'sigma': 1.2, 'stable': 75997, 'diego': 19, 'shaberi': 18, 'rob_diego': 0.025000986881061095, 'rob_shaberi': 0.023685145466268407}, {'sigma': 1.3, 'stable': 72634, 'diego': 29, 'shaberi': 29, 'rob_diego': 0.03992620535837211, 'rob_shaberi': 0.03992620535837211}, {'sigma': 1.4, 'stable': 69373, 'diego': 18, 'shaberi': 18, 'rob_diego': 0.025946693958744757, 'rob_shaberi': 0.025946693958744757}, {'sigma': 1.5, 'stable': 66232, 'diego': 22, 'shaberi': 22, 'rob_diego': 0.03321657204976446, 'rob_shaberi': 0.03321657204976446}, {'sigma': 1.6, 'stable': 63419, 'diego': 29, 'shaberi': 28, 'rob_diego': 0.04572762105993472, 'rob_shaberi': 0.04415080654062663}, {'sigma': 1.7, 'stable': 60806, 'diego': 28, 'shaberi': 28, 'rob_diego': 0.04604808735980002, 'rob_shaberi': 0.04604808735980002}, {'sigma': 1.8, 'stable': 58348, 'diego': 28, 'shaberi': 28, 'rob_diego': 0.047987934462192364, 'rob_shaberi': 0.047987934462192364}, {'sigma': 1.9, 'stable': 56014, 'diego': 23, 'shaberi': 22, 'rob_diego': 0.041061163280608416, 'rob_shaberi': 0.039275895311886314}, {'sigma': 2.0, 'stable': 53775, 'diego': 28, 'shaberi': 28, 'rob_diego': 0.05206880520688052, 'rob_shaberi': 0.05206880520688052}]


# ### Type II
# 
# _w is immobile, u and v > 1_
# 
# for type II the destabilizing module switches from the u-v to the v-w pair!!

# In[14]:


# apply sign constraints for specific topology and type (activating (+) or inhibiting (-))

def sign_constraints_type2(J):
    J[1, 2] =  abs(J[1, 2])   # w activates v
    J[2, 1] =  abs(J[2, 1])   # v activates w  → v-w cycle positive = destabilising
    J[0, 1] =  abs(J[0, 1])   # v activates u
    J[1, 0] = -abs(J[1, 0])   # u inhibits v   → u-v cycle negative = stabilising
    return J


# In[ ]:


def generate_jacobian_type2(sigma):

    # random matrix
    G = np.random.normal(0, sigma, (3, 3))
    np.fill_diagonal(G, 0)

    # J = G - I  (rmt convention, may 1972), diagonal becomes -1, self-decay handled by J = G - I, off-diagonal from N(0, sigma)
    J = G - np.eye(3)

    # apply sparsity mask after sampling from adjacency matrix, but only to off-diagonal elements, the diagonal is already -1 from J = G - I
    for i in range(3): 
        for j in range(3):
            if i != j and adjacency_matrix[i, j] == 0:
                J[i, j] = 0

    J = sign_constraints_type2(J)

    return J


# In[5]:


def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)


# In[6]:


# check turing instabilitiy: does diffusion destabilise a mode that was stable without diffusion?

# two ways to check turing stability:
# 1. shaberi et al (2025)
#    - compute all eigenvalues and check if real part < 0
#    - detects any instability, including oscillatory ones (complex eigenvalues)
#    - detects Turing I only (defined wavelength, restabilises at large k)
# 2. diego et al (2018) 
#    - characteristic polynomial and Routh-Hurwitz criteria
#    - detects only stationary instabilities, not oscillary ones
#    - condition: a3 < 0 AND a1 > 0 AND a2 > 0 for some k > 0.

# CHECK AGAIN LATER: WHAT IS THE DIFFERNECE EBTWEEN THOSE TWO???

# def is_turing_shaberi(J, DU, DV, DW):
#     D = np.diag([DU, DV, DW])
#     any_unstable = False
#     for k in np.linspace(0.1, 100, 100):                        # changed np.arange(0.1, 10, 100) to np.linspace(0.1, 10, 100) bc long run time and then i changed it to 0.1, 100,100 to check larger k values bc shaberi should give us more instabilities than diego, but it was giving us the same number, so maybe we need to check larger k values to see the restabilisation at large k for turing I instabilities
#         M = J - k**2 * D
#         if np.max(np.real(eigvals(M))) > 0:
#             any_unstable = True
#             break

#     if not any_unstable:
#         return False

#     M_large = J - 10**2 * D
#     return np.all(np.real(eigvals(M_large)) < 0)


def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.linspace(0.1, 10, 100):
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            return True
    return False


def is_turing_diego(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.logspace(-1, 2, 100):                          # changed (-1, 2, 500) to (-1, 2, 100) bc long run time
        M  = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        if a3 < 0 and a1 > 0 and a2 > 0:
            return True
    return False


# In[7]:


# type II specifications

n_samples = 100_000
sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#sigma = [0.8, 0.9, 1.1]
DU, DV, DW = 1.0, 1.0, 0.0


# In[8]:


results_type2_diego_rmt = []

for sig in sigma:
    np.random.seed(42)
    stable = 0
    turing_diego = 0
    turing_shaberi = 0

    for _ in range(n_samples):
        J = generate_jacobian_type2(sig)
        if is_stable(J):
            stable += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

    rob_diego   = 100 * turing_diego   / stable if stable > 0 else 0.0
    rob_shaberi = 100 * turing_shaberi / stable if stable > 0 else 0.0

    results_type2_diego_rmt.append({
        "sigma":        sig,
        "stable":       stable,
        "diego":        turing_diego,
        "shaberi":      turing_shaberi,
        "rob_diego":    rob_diego,
        "rob_shaberi":  rob_shaberi,
    })

print(f"{'Sigma':<8} {'Tested':>8} {'Stable':>8} {'Diego_Tu':>10} {'Shaberi_Tu':>12} {'Diego_Ro':>11} {'Shaberi_Ro':>14}")
print("-" * 80)

for r in results_type2_diego_rmt:
    print(f"{r['sigma']:<6.1f} {n_samples:>10,} {r['stable']:>7,} {r['diego']:>8,} {r['shaberi']:>11,} "
          f"{r['rob_diego']:>14.7f}% {r['rob_shaberi']:>14.7f}%")


# In[11]:


results_type2_diego_rmt_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99947, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.5, 'stable': 99409, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.6, 'stable': 97747, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.7, 'stable': 94972, 'diego': 2, 'shaberi': 2, 'rob_diego': 0.002105883839447416, 'rob_shaberi': 0.002105883839447416}, {'sigma': 0.8, 'stable': 91349, 'diego': 14, 'shaberi': 13, 'rob_diego': 0.015325838268618157, 'rob_shaberi': 0.014231135535145432}, {'sigma': 0.9, 'stable': 87499, 'diego': 37, 'shaberi': 37, 'rob_diego': 0.0422861975565435, 'rob_shaberi': 0.0422861975565435}, {'sigma': 1.0, 'stable': 83661, 'diego': 67, 'shaberi': 66, 'rob_diego': 0.08008510536570206, 'rob_shaberi': 0.07888980528561695}, {'sigma': 1.1, 'stable': 79829, 'diego': 120, 'shaberi': 120, 'rob_diego': 0.150321311803981, 'rob_shaberi': 0.150321311803981}, {'sigma': 1.2, 'stable': 76201, 'diego': 196, 'shaberi': 195, 'rob_diego': 0.25721447225102034, 'rob_shaberi': 0.25590215351504575}, {'sigma': 1.3, 'stable': 72837, 'diego': 267, 'shaberi': 266, 'rob_diego': 0.36657193459368176, 'rob_shaberi': 0.36519900599969796}, {'sigma': 1.4, 'stable': 69560, 'diego': 346, 'shaberi': 345, 'rob_diego': 0.4974123059229442, 'rob_shaberi': 0.49597469810235767}, {'sigma': 1.5, 'stable': 66540, 'diego': 440, 'shaberi': 439, 'rob_diego': 0.6612563871355576, 'rob_shaberi': 0.6597535317102494}, {'sigma': 1.6, 'stable': 63630, 'diego': 532, 'shaberi': 530, 'rob_diego': 0.8360836083608361, 'rob_shaberi': 0.832940436900833}, {'sigma': 1.7, 'stable': 60955, 'diego': 589, 'shaberi': 589, 'rob_diego': 0.9662866048724469, 'rob_shaberi': 0.9662866048724469}, {'sigma': 1.8, 'stable': 58421, 'diego': 691, 'shaberi': 690, 'rob_diego': 1.1827938583728455, 'rob_shaberi': 1.1810821451190496}, {'sigma': 1.9, 'stable': 56019, 'diego': 789, 'shaberi': 786, 'rob_diego': 1.408450704225352, 'rob_shaberi': 1.4030953783537727}, {'sigma': 2.0, 'stable': 53833, 'diego': 824, 'shaberi': 823, 'rob_diego': 1.5306596325673842, 'rob_shaberi': 1.5288020359259191}]


# ### Type III
# 
# _u is immobile, w and v are mobile, w > 0 !!_

# In[18]:


# apply sign constraints for specific topology and type (activating (+) or inhibiting (-))

def sign_constraints_type3(J):
    J[0, 1] =  abs(J[0, 1])   # v activates u
    J[1, 0] =  abs(J[1, 0])   # u activates v  → u-v cycle positive = destabilising
    J[1, 2] =  abs(J[1, 2])   # w activates v
    J[2, 1] = -abs(J[2, 1])   # v inhibits w   → v-w cycle negative = stabilising
    return J


# In[19]:


def generate_jacobian_type3(sigma):

    # random matrix
    G = np.random.normal(0, sigma, (3, 3))
    np.fill_diagonal(G, 0)

    # J = G - I  (rmt convention, may 1972), diagonal becomes -1, self-decay handled by J = G - I, off-diagonal from N(0, sigma)
    J = G - np.eye(3)

    # apply sparsity mask after sampling from adjacency matrix, but only to off-diagonal elements, the diagonal is already -1 from J = G - I
    for i in range(3): 
        for j in range(3):
            if i != j and adjacency_matrix[i, j] == 0:
                J[i, j] = 0

    J = sign_constraints_type3(J)

    return J


# In[20]:


def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)


# In[21]:


# check turing instabilitiy: does diffusion destabilise a mode that was stable without diffusion?

# two ways to check turing stability:
# 1. shaberi et al (2025)
#    - compute all eigenvalues and check if real part < 0
#    - detects any instability, including oscillatory ones (complex eigenvalues)
#    - detects Turing I only (defined wavelength, restabilises at large k)
# 2. diego et al (2018) 
#    - characteristic polynomial and Routh-Hurwitz criteria
#    - detects only stationary instabilities, not oscillary ones
#    - condition: a3 < 0 AND a1 > 0 AND a2 > 0 for some k > 0.

def is_turing_shaberi(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    any_unstable = False
    for k in np.linspace(0.1, 10, 100):                        # changed np.arange(0.1, 10, 100) to np.linspace(0.1, 10, 100) bc long run time
        M = J - k**2 * D
        if np.max(np.real(eigvals(M))) > 0:
            any_unstable = True
            break

    if not any_unstable:
        return False

    M_large = J - 10**2 * D
    return np.all(np.real(eigvals(M_large)) < 0)

def is_turing_diego(J, DU, DV, DW):
    D = np.diag([DU, DV, DW])
    for k in np.logspace(-1, 2, 100):                          # changed (-1, 2, 500) to (-1, 2, 100) bc long run time
        M  = J - k**2 * D
        a1 = -np.trace(M)
        a2 = (M[0,0]*M[1,1] - M[0,1]*M[1,0] +
              M[0,0]*M[2,2] - M[0,2]*M[2,0] +
              M[1,1]*M[2,2] - M[1,2]*M[2,1])
        a3 = -np.linalg.det(M)
        if a3 < 0 and a1 > 0 and a2 > 0:
            return True
    return False


# In[22]:


# type III specifications

n_samples = 100_000
sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#sigma = [0.6, 0.7]
DU, DV, DW = 0.0, 1.0, 1.0


# In[23]:


results_type3_diego_rmt = []

for sig in sigma:
    np.random.seed(42)
    stable = 0
    turing_diego = 0
    turing_shaberi = 0

    for _ in range(n_samples):
        J = generate_jacobian_type3(sig)
        if is_stable(J):
            stable += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

    rob_diego   = 100 * turing_diego   / stable if stable > 0 else 0.0
    rob_shaberi = 100 * turing_shaberi / stable if stable > 0 else 0.0

    results_type3_diego_rmt.append({
        "sigma":        sig,
        "stable":       stable,
        "diego":        turing_diego,
        "shaberi":      turing_shaberi,
        "rob_diego":    rob_diego,
        "rob_shaberi":  rob_shaberi,
    })

print(f"{'Sigma':<8} {'Tested':>8} {'Stable':>8} {'Diego_Tu':>10} {'Shaberi_Tu':>12} {'Diego_Ro':>11} {'Shaberi_Ro':>14}")
print("-" * 80)

for r in results_type3_diego_rmt:
    print(f"{r['sigma']:<6.1f} {n_samples:>10,} {r['stable']:>7} {r['diego']:>8} {r['shaberi']:>11} "
          f"{r['rob_diego']:>14.7f}% {r['rob_shaberi']:>14.7f}%")


# In[25]:


results_type3_diego_rmt_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99954, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.5, 'stable': 99408, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.6, 'stable': 97740, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.7, 'stable': 94897, 'diego': 4, 'shaberi': 4, 'rob_diego': 0.004215096367640705, 'rob_shaberi': 0.004215096367640705}, {'sigma': 0.8, 'stable': 91371, 'diego': 9, 'shaberi': 9, 'rob_diego': 0.009849952391896772, 'rob_shaberi': 0.009849952391896772}, {'sigma': 0.9, 'stable': 87394, 'diego': 23, 'shaberi': 23, 'rob_diego': 0.026317596173650364, 'rob_shaberi': 0.026317596173650364}, {'sigma': 1.0, 'stable': 83442, 'diego': 72, 'shaberi': 71, 'rob_diego': 0.0862874811246135, 'rob_shaberi': 0.08508904388677165}, {'sigma': 1.1, 'stable': 79613, 'diego': 110, 'shaberi': 110, 'rob_diego': 0.13816838958461558, 'rob_shaberi': 0.13816838958461558}, {'sigma': 1.2, 'stable': 75997, 'diego': 175, 'shaberi': 175, 'rob_diego': 0.2302722475887206, 'rob_shaberi': 0.2302722475887206}, {'sigma': 1.3, 'stable': 72634, 'diego': 260, 'shaberi': 259, 'rob_diego': 0.3579590825233362, 'rob_shaberi': 0.35658231682132335}, {'sigma': 1.4, 'stable': 69373, 'diego': 319, 'shaberi': 318, 'rob_diego': 0.4598330762688654, 'rob_shaberi': 0.45839159327115736}, {'sigma': 1.5, 'stable': 66232, 'diego': 386, 'shaberi': 386, 'rob_diego': 0.5827998550549583, 'rob_shaberi': 0.5827998550549583}, {'sigma': 1.6, 'stable': 63419, 'diego': 488, 'shaberi': 487, 'rob_diego': 0.7694854854223497, 'rob_shaberi': 0.7679086709030417}, {'sigma': 1.7, 'stable': 60806, 'diego': 598, 'shaberi': 598, 'rob_diego': 0.9834555800414433, 'rob_shaberi': 0.9834555800414433}, {'sigma': 1.8, 'stable': 58348, 'diego': 660, 'shaberi': 660, 'rob_diego': 1.1311441694659627, 'rob_shaberi': 1.1311441694659627}, {'sigma': 1.9, 'stable': 56014, 'diego': 733, 'shaberi': 731, 'rob_diego': 1.308601421073303, 'rob_shaberi': 1.305030885135859}, {'sigma': 2.0, 'stable': 53775, 'diego': 800, 'shaberi': 800, 'rob_diego': 1.487680148768015, 'rob_shaberi': 1.487680148768015}]


# In[ ]:




