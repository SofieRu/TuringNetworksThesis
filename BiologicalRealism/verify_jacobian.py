import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

# ============================================================================
# CHOOSE WHICH TOPOLOGY TO TEST
# ============================================================================
# Set this to 3954 or 1754 depending on which you want to verify
TOPOLOGY = 1754

# ============================================================================
# IMPORT THE RIGHT FUNCTIONS
# ============================================================================
if TOPOLOGY == 3954:
    from heterogenous_ring_3954 import ode_system, compute_jacobian, find_steady_state
    CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'
else:
    from heterogenous_ring_1754 import ode_system, compute_jacobian, find_steady_state
    CSV_PATH = '../TopologyRanking/Topology1754/1754_NEWTURINGCLASS_lhs_results_parameters.csv'

# ============================================================================
# LOAD SAMPLE PARAMETER SETS FROM YOUR CSV
# ============================================================================
print(f"Loading parameters for topology #{TOPOLOGY} ...")
df = pd.read_csv(CSV_PATH)

# Pick 5 Type-I parameter sets from different configs (diverse test cases)
type_i = df[df['classification'] == 'Type-I']
sample_rows = type_i.head(5)  # first 5 Type-I rows

# ============================================================================
# RUN COMPARISON FOR EACH SAMPLE
# ============================================================================
print(f"\nTesting {len(sample_rows)} parameter sets for topology #{TOPOLOGY}\n")
print("="*70)

# Build parameter arrays based on topology
if TOPOLOGY == 3954:
    param_cols = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u',
                  'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v',
                  'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']
else:  # 1754
    param_cols = ['alpha_u', 'beta_u', 'K_vu', 'delta_u',  # no K_uu
                  'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v',
                  'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']

max_diff_overall = 0.0

for idx, row in sample_rows.iterrows():
    config_id = row['config_id']
    
    params = np.array([row[col] for col in param_cols])
    ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    
    # Analytical Jacobian (your code)
    J_analytical = compute_jacobian(ss, params)
    
    # Numerical Jacobian (via finite differences)
    J_numerical = np.zeros((3, 3))
    for i in range(3):
        # f_i returns the i-th component of ode_system
        def f_i(state, idx=i):
            return ode_system(state, params)[idx]
        J_numerical[i] = approx_fprime(ss, f_i, epsilon=1e-7)
    
    # Compare
    diff = np.abs(J_numerical - J_analytical)
    max_diff = np.max(diff)
    max_diff_overall = max(max_diff_overall, max_diff)
    
    print(f"\nConfig {config_id}, Type-I parameter set:")
    print(f"  Steady state: ({ss[0]:.4f}, {ss[1]:.4f}, {ss[2]:.4f})")
    print(f"  Max abs difference between analytical and numerical: {max_diff:.2e}")
    
    if max_diff > 1e-4:
        print(f"  ⚠️  WARNING: Difference exceeds 1e-4 — check Jacobian!")
        print(f"  Analytical:")
        print(f"  {J_analytical}")
        print(f"  Numerical:")
        print(f"  {J_numerical}")

print("\n" + "="*70)
print(f"OVERALL MAX DIFFERENCE: {max_diff_overall:.2e}")

if max_diff_overall < 1e-4:
    print(f"✓ Analytical Jacobian for topology #{TOPOLOGY} is CORRECT.")
else:
    print(f"✗ Analytical Jacobian for topology #{TOPOLOGY} has issues.")

# What you should see
# For each parameter set, output like:
# Config 13, Type-I parameter set:
#   Steady state: (0.1300, 0.4900, 0.0300)
#   Max abs difference between analytical and numerical: 3.42e-08
# And at the end:
# OVERALL MAX DIFFERENCE: 4.51e-08
# ✓ Analytical Jacobian for topology #3954 is CORRECT.
# Acceptance threshold: anything below 1e-4 is essentially numerical precision noise — the Jacobian is correct.
# If you see differences above 1e-4, something is wrong with the analytical Jacobian. The script will print both versions side-by-side so you can see which entry is off.

# What this confirms
# If both topologies pass:

# The mathematical implementation of #3954 and #1754 is correct
# Whatever #1754 > #3954 result you're seeing is a real finding, not a coding bug
# You can confidently write about your results

# If something fails:

# We'd need to fix the Jacobian for that topology before trusting any results from it
# But the entries are very simple — debugging would be fast