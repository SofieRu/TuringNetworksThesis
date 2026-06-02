# Quick standalone script, not part of main pipeline
import numpy as np
from homogenous_ring import (build_ring_jacobian_homogeneous, 
                              compute_jacobian, ode_system)
import pandas as pd

# Load both configs
df = pd.read_csv('../TopologyRanking/Topology3954/3954_NEW_lhs_results_parameters.csv')

for config_id in [13, 2]:
    row = df[(df['config_id'] == config_id) & (df['param_rank'] == 1)].iloc[0]
    params = np.array([row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'],
                       row['delta_u'], row['alpha_v'], row['beta_v'], row['K_uv'],
                       row['K_wv'], row['delta_v'], row['alpha_w'], row['beta_w'],
                       row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']])
    ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}
    
    for N in [5, 10, 20, 30]:
        J = build_ring_jacobian_homogeneous(N, ss, params, hopping)
        eig = np.max(np.real(np.linalg.eigvals(J)))
        print(f"Config {config_id}, N={N}: max Re(λ) = {eig:.6f}")