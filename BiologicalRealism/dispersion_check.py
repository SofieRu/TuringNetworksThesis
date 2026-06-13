import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heterogenous_ring_3954 import compute_jacobian, find_steady_state, ode_system

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

# Load some parameter sets from config 54 (VWFreeze, only U diffuses)
df = pd.read_csv('../TopologyRanking/Topology3954/3954_FILTER_lhs_results_parameters.csv')
config_54 = df[df['config_id'] == 54].head(8)

dU, dV, dW = 1.0, 0.0, 0.0  # config 54's diffusion

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
k_values = np.arange(0.01, 30.01, 0.05)

for ax, (_, row) in zip(axes.flatten(), config_54.iterrows()):
    params = np.array([row[c] for c in ['alpha_u','beta_u','K_uu','K_vu','delta_u',
                                          'alpha_v','beta_v','K_uv','K_wv','delta_v',
                                          'alpha_w','beta_w','K_ww','K_uw','K_vw','delta_w']])
    ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    J = compute_jacobian(ss, params)
    D = np.diag([dU, dV, dW])
    
    max_reals = []
    for k in k_values:
        M = J - k**2 * D
        max_reals.append(np.max(np.real(np.linalg.eigvals(M))))
    
    ax.plot(k_values, max_reals, 'b-')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('max Re(λ)')
    ax.set_title(f"Sample (k_max from CSV: {row.get('max_growth_rate', 'N/A')})")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_config54_dispersion.png', dpi=200)
print("Saved diagnostic")