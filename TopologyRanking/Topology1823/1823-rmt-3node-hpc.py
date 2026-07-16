#!/usr/bin/env python3
import numpy as np
import sys
import pickle
import os
import pandas as pd
from numpy.linalg import eigvals
from scipy.optimize import fsolve

# Create directories (for local testing)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


################ RMT ANALYSIS FOR TOPOLOGY #1823 ################

# Topology: u activates v, u inhibits w, v inhibits u, v activates w, w activates v, w inhibits w

adjacency_matrix_1823 = np.array([
    [0, 1, 0],  # u affected by: v
    [1, 0, 1],  # v affected by: u, w
    [1, 1, 0],  # w affected by: u, v
])

def sign_constraints_1823(J):
    J[0, 1] = -abs(J[0, 1])   # v inhibits u → NEGATIVE
    J[1, 0] =  abs(J[1, 0])   # u activates v → POSITIVE
    J[1, 2] =  abs(J[1, 2])   # w activates v → POSITIVE
    J[2, 0] = -abs(J[2, 0])   # u inhibits w → NEGATIVE
    J[2, 1] =  abs(J[2, 1])   # v activates w → POSITIVE
    return J

# v → u : inhibition  → J[0,1] < 0
# u → v : activation  → J[1,0] > 0
# w → v : activation  → J[1,2] > 0
# u → w : inhibition  → J[2,0] < 0
# v → w : activation  → J[2,1] > 0
# self-loops (w→w): handled by diagonal = -1

def generate_jacobian_1823(sigma):
    
    # Random matrix
    G = np.random.normal(0, sigma, (3, 3))
    np.fill_diagonal(G, 0)
    
    # J = G - I (diagonal becomes -1, self-decay)
    J = G - np.eye(3)
    
    # Apply sparsity mask (off-diagonal only)
    for i in range(3): 
        for j in range(3):
            if i != j and adjacency_matrix_1823[i, j] == 0:
                J[i, j] = 0

    J = sign_constraints_1823(J)

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
    # STEP 1: Homogeneous steady state must be stable
    if np.max(np.real(eigs_0)) >= 0:
        return None
    
    # STEP 2: Sweep k ∈ [0, 10] with step 0.01 (Shaberi 2025 methodology)
    D = np.diag([DU, DV, DW])
    k_values = np.arange(0.01, 10.01, 0.1) # change later back to 0.01
    
    max_reals = np.zeros(len(k_values))
    has_complex_unstable = False
    
    for i, k in enumerate(k_values):
        M = J - (k**2) * D
        eigs_k = np.linalg.eigvals(M)
        max_reals[i] = np.max(np.real(eigs_k))
        
        if max_reals[i] > 0:
            unstable_eigs = eigs_k[np.real(eigs_k) > 0]
            if np.any(np.abs(np.imag(unstable_eigs)) > 1e-8):
                has_complex_unstable = True
    
    if np.max(max_reals) <= 0:
        return None
    
    if has_complex_unstable:
        return 'Hopf'
    
    # STEP 3: Type-I = restabilises (goes negative) by k=10
    if max_reals[-1] < 0:
        return 'Type-I'
    
    # STEP 4: Distinguish Filter from Type-II by peak location
    # Filter (Diego 2018): monotonic — max sits at the END of the range
    # Type-II: has an interior peak — max is somewhere in the middle
    max_idx = np.argmax(max_reals)
    
    # Allow a tiny buffer for floating-point noise (last 0.2% of range)
    if max_idx >= len(k_values) - 2:
        return 'Filter'
    return 'Type-II'


# DIFFUSION CONFIGURATIONS
# LATER ADD VARIATIONS SO EQUAL AND UNEQUAL AND LIMIT DIFFUSION RATES!!

DIFFUSION_CONFIGS = {
    
    # TYPE 1
    # controls
    0:  {"name": "NEW_RMT_1823_Type1_Control_Slow",         "dU": 0.1,  "dV": 0.1,  "dW": 0.1},
    1:  {"name": "NEW_RMT_1823_Type1_Control_Equal",        "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    2:  {"name": "NEW_RMT_1823_Type1_Control_Fast",         "dU": 10.0, "dV": 10.0, "dW": 10.0},

    # node u diffuses faster than v and w
    3:  {"name": "NEW_RMT_1823_Type1_UFast_Unequal1",       "dU": 10.0, "dV": 1.0,  "dW": 1.0},
    4:  {"name": "NEW_RMT_1823_Type1_UFast_Unequal2",       "dU": 1.0,  "dV": 0.1,  "dW": 0.1},
    5:  {"name": "NEW_RMT_1823_Type1_UFast_Unequal3",       "dU": 10.0, "dV": 0.1,  "dW": 1.0},
    6:  {"name": "NEW_RMT_1823_Type1_UFast_Unequal4",       "dU": 10.0, "dV": 1.0,  "dW": 0.1},
    7:  {"name": "NEW_RMT_1823_Type1_UFast_Unequal5",       "dU": 10.0, "dV": 0.1,  "dW": 0.1},

    # TYPE 2
    # node u immobile
    8:  {"name": "NEW_RMT_1823_Type2_UFreeze_Equal1",       "dU": 0.0, "dV": 1.0,  "dW": 1.0},
    9:  {"name": "NEW_RMT_1823_Type2_UFreeze_Equal2",       "dU": 0.0, "dV": 0.1,  "dW": 0.1},
    10: {"name": "NEW_RMT_1823_Type2_UFreeze_Equal3",       "dU": 0.0, "dV": 10.0, "dW": 10.0},
    11: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal1",     "dU": 0.0, "dV": 1.0,  "dW": 0.1},
    12: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal2",     "dU": 0.0, "dV": 0.1,  "dW": 1.0},
    13: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal3",     "dU": 0.0, "dV": 10.0, "dW": 1.0},
    14: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal4",     "dU": 0.0, "dV": 1.0,  "dW": 10.0},
    15: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal5",     "dU": 0.0, "dV": 0.1,  "dW": 10.0},
    16: {"name": "NEW_RMT_1823_Type2_UFreeze_Unequal6",     "dU": 0.0, "dV": 10.0, "dW": 0.1},

    # TYPE 3
    # node w is immobile
    17: {"name": "NEW_RMT_1823_Type3_WFreeze_Equal1",       "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    18: {"name": "NEW_RMT_1823_Type3_WFreeze_Equal2",       "dU": 0.1,  "dV": 0.1,  "dW": 0.0},
    19: {"name": "NEW_RMT_1823_Type3_WFreeze_Equal3",       "dU": 10.0, "dV": 10.0, "dW": 0.0},
    20: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal1",     "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
    21: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal2",     "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
    22: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal3",     "dU": 10.0, "dV": 1.0,  "dW": 0.0},
    23: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal4",     "dU": 1.0,  "dV": 10.0, "dW": 0.0},
    24: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal5",     "dU": 0.1,  "dV": 10.0, "dW": 0.0},
    25: {"name": "NEW_RMT_1823_Type3_WFreeze_Unequal6",     "dU": 10.0, "dV": 0.1,  "dW": 0.0},

    # node v immobile
    26: {"name": "NEW_RMT_1823_Type3_VFreeze_Equal1",       "dU": 1.0,  "dV": 0.0,  "dW": 1.0},
    27: {"name": "NEW_RMT_1823_Type3_VFreeze_Equal2",       "dU": 0.1,  "dV": 0.0,  "dW": 0.1},
    28: {"name": "NEW_RMT_1823_Type3_VFreeze_Equal3",       "dU": 10.0, "dV": 0.0,  "dW": 10.0},
    29: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal1",     "dU": 0.1,  "dV": 0.0,  "dW": 1.0},
    30: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal2",     "dU": 1.0,  "dV": 0.0,  "dW": 0.1},
    31: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal3",     "dU": 0.1,  "dV": 0.0,  "dW": 10.0},
    32: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal4",     "dU": 10.0, "dV": 0.0,  "dW": 0.1},
    33: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal5",     "dU": 1.0,  "dV": 0.0,  "dW": 10.0},
    34: {"name": "NEW_RMT_1823_Type3_VFreeze_Unequal6",     "dU": 10.0, "dV": 0.0,  "dW": 1.0},
}

#full range sigma values but to test we do less values
SIGMA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0 , 8.0, 9.0, 10.0]

# MAIN ANALYSIS FUNCTION
def run_analysis(config_id, n_samples):
    
    config = DIFFUSION_CONFIGS[config_id]
    DU, DV, DW = config["dU"], config["dV"], config["dW"]
    config_name = config["name"]
    
    print(f"Starting {config_name}: dU={DU}, dV={DV}, dW={DW}")
    print(f"Testing {len(SIGMA_VALUES)} sigma values with {n_samples:,} samples each")
    print(f"=" * 70)
    
    # Store results for all sigma values
    results_by_sigma = []
    
    for sigma_idx, sigma in enumerate(SIGMA_VALUES):
        np.random.seed(42)  # Same seed for all configs for fair comparison
        
        # Initialize counters
        stable = 0
        diego_turing = 0
        shaberi_total = 0
        shaberi_type_I = 0
        shaberi_type_II = 0
        shaberi_hopf = 0
        filter_count = 0
        
        # Main loop
        for i in range(n_samples):
            J = generate_jacobian_1823(sigma)
            eigs_0 = eigvals(J)

            if np.max(np.real(eigs_0)) < 0:  #Use existing eigenvalues!
                stable += 1
                
                # Diego method
                if is_turing_diego(J, DU, DV, DW):
                    diego_turing += 1
                
                # Shaberi method
                turing_type = is_turing_shaberi(J, eigs_0, DU, DV, DW)
                
                if turing_type is not None: # NEW, do not count Hopf as Turing for Shaberi
                    if turing_type == 'Hopf':
                        shaberi_hopf += 1
                    else:
                        shaberi_total += 1
                        if turing_type == 'Type-I':
                            shaberi_type_I += 1
                        elif turing_type == 'Type-II':
                            shaberi_type_II += 1
                        elif turing_type == 'Filter':
                            filter_count += 1
                    
            # Progress indicator
            if (i + 1) % 100000 == 0:
                print(f"  [sig={sigma:.1f}] {i+1:,}/{n_samples:,} | Stable: {stable} | "
                      f"Diego: {diego_turing} | Shaberi: {shaberi_total}")
        
        # Calculate robustness
        rob_diego = 100 * diego_turing / n_samples
        rob_shaberi_total = 100 * shaberi_total / n_samples
        rob_shaberi_type_I = 100 * shaberi_type_I / n_samples
        
        # Store results for this sigma
        sigma_result = {
            "sigma": sigma,
            "n_samples": n_samples,
            "stable": stable,
            "diego_turing": diego_turing,
            "shaberi_total": shaberi_total,
            "shaberi_type_I": shaberi_type_I,
            "shaberi_type_II": shaberi_type_II,
            "shaberi_hopf": shaberi_hopf,
            "filter_count": filter_count,
            "rob_diego": rob_diego,
            "rob_shaberi_total": rob_shaberi_total,
            "rob_shaberi_type_I": rob_shaberi_type_I,
        }

        results_by_sigma.append(sigma_result)
    
    # Create summary results
    results = {
        "config_name": config_name,
        "config_id": config_id,
        "diffusion": {"dU": DU, "dV": DV, "dW": DW},
        "n_samples_per_sigma": n_samples,
        "n_sigma_values": len(SIGMA_VALUES),
        "sigma_values": SIGMA_VALUES,
        "results_by_sigma": results_by_sigma,
    }
    
    return results

# HPC EXECUTION

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 1823-rmt-3node-hpc.py <config_id>")
        sys.exit(1)
    
    config_id = int(sys.argv[1])
    n_samples = 500_000  # 100K samples per sigma value but ater 1mio
    
    results = run_analysis(config_id, n_samples)
    
    # Save as pickle
    output_pkl = f"results/{results['config_name']}_1mio.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    # Save as CSV (flatten sigma results)
    csv_rows = []
    for sigma_result in results['results_by_sigma']:
        row = {
            'config_name': results['config_name'],
            'config_id': results['config_id'],
            'dU': results['diffusion']['dU'],
            'dV': results['diffusion']['dV'],
            'dW': results['diffusion']['dW'],
            'sigma': sigma_result['sigma'],
            'n_samples': sigma_result['n_samples'],
            'stable': sigma_result['stable'],
            'diego_turing': sigma_result['diego_turing'],
            'shaberi_total': sigma_result['shaberi_total'],
            'shaberi_type_I': sigma_result['shaberi_type_I'],
            'shaberi_type_II': sigma_result['shaberi_type_II'],
            'shaberi_hopf': sigma_result['shaberi_hopf'],
            'filter_count': sigma_result['filter_count'],
            'rob_diego': sigma_result['rob_diego'],
            'rob_shaberi_total': sigma_result['rob_shaberi_total'],
            'rob_shaberi_type_I': sigma_result['rob_shaberi_type_I'],
        }
        csv_rows.append(row)
    
    output_csv = f"results/{results['config_name']}_100k.csv"
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPLETED: {results['config_name']}")
    print(f"{'='*70}")
    print(f"Tested {len(SIGMA_VALUES)} sigma values")
    print(f"{n_samples:,} samples per sigma")
    print(f"Total: {len(SIGMA_VALUES) * n_samples:,} Jacobians generated")
    print(f"\nSaved to:")