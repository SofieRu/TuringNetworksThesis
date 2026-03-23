#!/usr/bin/env python
# coding: utf-8

# ## Type I - III diffusable nodes for 3954 topology using Random Matrix Theory (RMT)

# **#3954 topology:**  
# U: self-activating + inhibited by V  
# V: activated by U + inhibited by W  
# W: inhibited by U + inhibited by V + self-activating  
# 
# **Classigying Type I to III:**  
# The destabilising cycle involves V and W. The complementary node is U. 
# 
# From this we can anticipate:  
# DU, DV, DW = 10.0, 0.0, 1.0     # Type I: V immobile, w is OUTSIDE → w must be fast → DW=10, DU=1  
# DU, DV, DW = 1.0, 1.0, 0.0      # Type II: V immobile  
# DU, DV, DW = 1.0, 1.0, 0.0      # Type III: W immobile  

# In[20]:


import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.linalg import eig


# In[21]:


adjacency_matrix_3954 = np.array([
    [0, 1, 0],   # u affected by: v (self-loop via -I)
    [1, 0, 1],   # v affected by: u, w
    [1, 1, 0],   # w affected by: u, v (self-loop via -I)
])

# adjacency[i,j] = 1 means j affects i
# diagonal = 0, self-loops handled by J = G - I
# u: self-activating(-I) + inhibited by v → row 0: v affects u
# v: activated by u + inhibited by w → row 1: u and w affect v
# w: inhibited by u + inhibited by v + self-activating(-I) → row 2: u and v affect w


# ### Type I
# 
# _ is immobile,  and  > 1_

# In[22]:


def sign_constraints_3954_type1(J):
    J[0, 1] = -abs(J[0, 1])   # v inhibits u
    J[1, 0] =  abs(J[1, 0])   # u activates v
    J[1, 2] = -abs(J[1, 2])   # w inhibits v
    J[2, 0] = -abs(J[2, 0])   # u inhibits w
    J[2, 1] = -abs(J[2, 1])   # v inhibits w
    return J

# v → u : inhibition  → J[0,1] < 0
# u → v : activation  → J[1,0] > 0
# w → v : inhibition  → J[1,2] < 0
# u → w : inhibition  → J[2,0] < 0
# v → w : inhibition  → J[2,1] < 0
# self-loops (u→u, w→w): handled by diagonal = -1


# In[23]:


def generate_jacobian_3954_type1(sigma):

    # random matrix
    G = np.random.normal(0, sigma, (3, 3))
    np.fill_diagonal(G, 0)

    # J = G - I  (rmt convention, may 1972), diagonal becomes -1, self-decay handled by J = G - I, off-diagonal from N(0, sigma)
    J = G - np.eye(3)

    # apply sparsity mask after sampling from adjacency matrix, but only to off-diagonal elements, the diagonal is already -1 from J = G - I
    for i in range(3): 
        for j in range(3):
            if i != j and adjacency_matrix_3954[i, j] == 0:
                J[i, j] = 0

    J = sign_constraints_3954_type1(J)

    return J


# In[28]:


def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)

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


# In[ ]:


# type I specifications

n_samples = 100_000
#sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
sigma = [0.9, 1.0]
DU, DV, DW = 0.0, 1.0, 1.0 # try to vary DW to see if we get a different robustness score


# In[29]:


results_type1_3954_rmt = []

for sig in sigma:
    np.random.seed(42)
    stable = 0
    turing_diego = 0
    turing_shaberi = 0

    for _ in range(n_samples):
        J = generate_jacobian_3954_type1(sig)
        if is_stable(J):
            stable += 1
            if is_turing_diego(J, DU, DV, DW):
                turing_diego += 1
            if is_turing_shaberi(J, DU, DV, DW):
                turing_shaberi += 1

    rob_diego   = 100 * turing_diego   / stable if stable > 0 else 0.0
    rob_shaberi = 100 * turing_shaberi / stable if stable > 0 else 0.0

    results_type1_3954_rmt.append({
        "sigma":        sig,
        "stable":       stable,
        "diego":        turing_diego,
        "shaberi":      turing_shaberi,
        "rob_diego":    rob_diego,
        "rob_shaberi":  rob_shaberi,
    })

print(f"{'Sigma':<8} {'Tested':>8} {'Stable':>8} {'Diego_Tu':>10} {'Shaberi_Tu':>12} {'Diego_Ro':>11} {'Shaberi_Ro':>14}")
print("-" * 80)

for r in results_type1_3954_rmt:
    print(f"{r['sigma']:<6.1f} {n_samples:>10,} {r['stable']:>7,} {r['diego']:>8,} {r['shaberi']:>11,} "
          f"{r['rob_diego']:>14.7f}% {r['rob_shaberi']:>14.7f}%")


# In[ ]:


print(results_type1_3954_rmt)

# DU, DV, DW = 10.0, 0.0, 1.0
results_type1_3954_rmt_d10_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99981, 'diego': 2, 'shaberi': 2, 'rob_diego': 0.0020003800722137206, 'rob_shaberi': 0.0020003800722137206}, {'sigma': 0.5, 'stable': 99732, 'diego': 35, 'shaberi': 35, 'rob_diego': 0.035094052059519515, 'rob_shaberi': 0.035094052059519515}, {'sigma': 0.6, 'stable': 98966, 'diego': 181, 'shaberi': 177, 'rob_diego': 0.18289109391104016, 'rob_shaberi': 0.17884930178040942}, {'sigma': 0.7, 'stable': 97782, 'diego': 522, 'shaberi': 519, 'rob_diego': 0.5338405841565932, 'rob_shaberi': 0.53077253482236}, {'sigma': 0.8, 'stable': 96149, 'diego': 1212, 'shaberi': 1204, 'rob_diego': 1.2605435313939823, 'rob_shaberi': 1.252223112044847}, {'sigma': 0.9, 'stable': 94311, 'diego': 2191, 'shaberi': 2180, 'rob_diego': 2.3231648482149483, 'rob_shaberi': 2.3115013094973014}, {'sigma': 1.0, 'stable': 92345, 'diego': 3263, 'shaberi': 3243, 'rob_diego': 3.533488548378364, 'rob_shaberi': 3.511830635118306}, {'sigma': 1.1, 'stable': 90198, 'diego': 4271, 'shaberi': 4251, 'rob_diego': 4.7351382514024705, 'rob_shaberi': 4.712964810749684}, {'sigma': 1.2, 'stable': 88080, 'diego': 5416, 'shaberi': 5391, 'rob_diego': 6.148955495004541, 'rob_shaberi': 6.120572207084469}, {'sigma': 1.3, 'stable': 85689, 'diego': 6255, 'shaberi': 6233, 'rob_diego': 7.299653397752337, 'rob_shaberi': 7.273979157184703}, {'sigma': 1.4, 'stable': 83226, 'diego': 7085, 'shaberi': 7054, 'rob_diego': 8.51296469853171, 'rob_shaberi': 8.475716723139403}, {'sigma': 1.5, 'stable': 80727, 'diego': 7678, 'shaberi': 7665, 'rob_diego': 9.51106816802309, 'rob_shaberi': 9.494964510015237}, {'sigma': 1.6, 'stable': 78077, 'diego': 8137, 'shaberi': 8116, 'rob_diego': 10.42176313126785, 'rob_shaberi': 10.394866606042752}, {'sigma': 1.7, 'stable': 75469, 'diego': 8601, 'shaberi': 8581, 'rob_diego': 11.396732433184486, 'rob_shaberi': 11.370231485775616}, {'sigma': 1.8, 'stable': 72870, 'diego': 8850, 'shaberi': 8827, 'rob_diego': 12.14491560312886, 'rob_shaberi': 12.113352545629203}, {'sigma': 1.9, 'stable': 70323, 'diego': 9027, 'shaberi': 9006, 'rob_diego': 12.836483085192611, 'rob_shaberi': 12.806620877948893}, {'sigma': 2.0, 'stable': 67959, 'diego': 9222, 'shaberi': 9202, 'rob_diego': 13.569946585441222, 'rob_shaberi': 13.540517076472579}]

#results_type1_3954_rmt_d5_backup = 

#results_type1_3954_rmt_d1_backup = 

# DU, DV, DW = 1.0, 1.0, 0.0
results_type2_3954_rmt_backup = [{'sigma': 0.1, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.2, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.3, 'stable': 100000, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.4, 'stable': 99981, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.5, 'stable': 99732, 'diego': 0, 'shaberi': 0, 'rob_diego': 0.0, 'rob_shaberi': 0.0}, {'sigma': 0.6, 'stable': 98966, 'diego': 2, 'shaberi': 2, 'rob_diego': 0.0020208960653153606, 'rob_shaberi': 0.0020208960653153606}, {'sigma': 0.7, 'stable': 97782, 'diego': 8, 'shaberi': 8, 'rob_diego': 0.008181464891288785, 'rob_shaberi': 0.008181464891288785}, {'sigma': 0.8, 'stable': 96149, 'diego': 24, 'shaberi': 23, 'rob_diego': 0.02496125804740559, 'rob_shaberi': 0.02392120562876369}, {'sigma': 0.9, 'stable': 94311, 'diego': 65, 'shaberi': 65, 'rob_diego': 0.06892091060427734, 'rob_shaberi': 0.06892091060427734}, {'sigma': 1.0, 'stable': 92345, 'diego': 123, 'shaberi': 123, 'rob_diego': 0.13319616654935296, 'rob_shaberi': 0.13319616654935296}, {'sigma': 1.1, 'stable': 90198, 'diego': 226, 'shaberi': 226, 'rob_diego': 0.25055987937648283, 'rob_shaberi': 0.25055987937648283}, {'sigma': 1.2, 'stable': 88080, 'diego': 323, 'shaberi': 322, 'rob_diego': 0.3667120799273388, 'rob_shaberi': 0.3655767484105359}, {'sigma': 1.3, 'stable': 85689, 'diego': 438, 'shaberi': 435, 'rob_diego': 0.5111507894828975, 'rob_shaberi': 0.5076497566782201}, {'sigma': 1.4, 'stable': 83226, 'diego': 536, 'shaberi': 535, 'rob_diego': 0.6440295100088914, 'rob_shaberi': 0.6428279624155913}, {'sigma': 1.5, 'stable': 80727, 'diego': 680, 'shaberi': 678, 'rob_diego': 0.842345188103113, 'rob_shaberi': 0.8398677022557509}, {'sigma': 1.6, 'stable': 78077, 'diego': 786, 'shaberi': 784, 'rob_diego': 1.0066985155679649, 'rob_shaberi': 1.0041369417370032}, {'sigma': 1.7, 'stable': 75469, 'diego': 877, 'shaberi': 877, 'rob_diego': 1.1620665438789437, 'rob_shaberi': 1.1620665438789437}, {'sigma': 1.8, 'stable': 72870, 'diego': 968, 'shaberi': 967, 'rob_diego': 1.3283930286812131, 'rob_shaberi': 1.327020721833402}, {'sigma': 1.9, 'stable': 70323, 'diego': 1074, 'shaberi': 1073, 'rob_diego': 1.5272385990358772, 'rob_shaberi': 1.525816589167129}, {'sigma': 2.0, 'stable': 67959, 'diego': 1128, 'shaberi': 1129, 'rob_diego': 1.6598243058314572, 'rob_shaberi': 1.6612957812798894}]

# DU, DV, DW = 0.0, 1.0, 1.0
#results_type3_3954_rmt_backup =


# In[ ]:




