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

################ RMT ANALYSIS FOR TOPOLOGY #3954 ################

# 3954: u activates u, u activates v, u inhibits w, v inhibits u, v inhibits w, w activates w, w inhibits v

#adjency matrix would be:     
# [1, 1, 0],
# [1, 0, 1],
# [1, 1, 1],
#but bc the diagonal always has to be 0, we set the diagonal to 0 and get:

adjacency_matrix_3954 = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])

def sign_constraints_3954(J):
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

def generate_jacobian_3954(sigma):
    
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

    J = sign_constraints_3954(J)

    return J

# TURING DETECTION METHODS

def is_stable(J):
    return np.all(np.real(eigvals(J)) < 0)

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
    # STEP 1: Check stability at k=0
    if np.max(np.real(eigs_0)) >= 0:
        return None
    
    # STEP 2: Check for instability with diffusion
    # SUPPOSED TO BE 0.01 STEP, BUT INCREASED TO 0.1 FOR SPEED, CHANGE BACK LATER
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
    
    # STEP 3: Type I vs Type II
    k_high_values = np.linspace(10, 50, 20)
    for k in k_high_values:
        M = J - k**2 * D
        eigs_k = np.linalg.eigvals(M)
        if np.max(np.real(eigs_k)) < 0:
            return 'Type-I'
    
    return 'Type-II'

# OLD SHABERI FUNCTION WITHOUT HOPF DETECTION, KEPT FOR REFERENCE, CHANGE BACK LATER, CHECK WHICH ONE IS THE RIGHT ONE!!!!
# def is_turing_shaberi(J, DU, DV, DW):
#     D = np.diag([DU, DV, DW])
#     for k in np.linspace(0.1, 10, 100):
#         M = J - k**2 * D
#         if np.max(np.real(eigvals(M))) > 0:
#             return True
#     return False





# EXTENDED VERSION

DIFFUSION_CONFIGS = {
    # DCC: A=Destable, B=Complementary, C=Complementary
    0:  {"name": "RMT_3954_DCC_Type1",          "dU": 1.0,  "dV": 10.0, "dW": 10.0},
    1:  {"name": "RMT_3954_DCC_Type1_OneFast",  "dU": 1.0,  "dV": 10.0, "dW": 1.0},
    2:  {"name": "RMT_3954_DCC_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    
    3:  {"name": "RMT_3954_DCC_Type2_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    4:  {"name": "RMT_3954_DCC_Type2_Unequal1", "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
    5:  {"name": "RMT_3954_DCC_Type2_Unequal2", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
    6:  {"name": "RMT_3954_DCC_Type2_Limit",    "dU": 0.1,  "dV": 0.1,  "dW": 0.0},
    
    7:  {"name": "RMT_3954_DCC_Type3_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
    8:  {"name": "RMT_3954_DCC_Type3_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
    9:  {"name": "RMT_3954_DCC_Type3_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
    10: {"name": "RMT_3954_DCC_Type3_Limit",    "dU": 0.0,  "dV": 0.1,  "dW": 0.1},
    
    # CDD: A=Complementary, B=Destable, C=Destable
    11: {"name": "RMT_3954_CDD_Type1",          "dU": 10.0, "dV": 1.0,  "dW": 1.0},
    12: {"name": "RMT_3954_CDD_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},   
    
    13: {"name": "RMT_3954_CDD_Type2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
    14: {"name": "RMT_3954_CDD_Type2_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
    15: {"name": "RMT_3954_CDD_Type2_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
    16: {"name": "RMT_3954_CDD_Type2_Limit",    "dU": 0.0,  "dV": 0.1,  "dW": 0.1},
    
    17: {"name": "RMT_3954_CDD_Type3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    18: {"name": "RMT_3954_CDD_Type3_Unequal1", "dU": 1.0,  "dV": 0.1,  "dW": 0.0},
    19: {"name": "RMT_3954_CDD_Type3_Unequal2", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
    20: {"name": "RMT_3954_CDD_Type3_Limit",    "dU": 0.1,  "dV": 0.1,  "dW": 0.0},
    
    # CCD: A=Compl., B=Compl., C=Destable
    21: {"name": "RMT_3954_CCD_Type1",          "dU": 10.0, "dV": 10.0, "dW": 1.0},
    22: {"name": "RMT_3954_CCD_Type1_OneFast",  "dU": 10.0, "dV": 1.0,  "dW": 1.0},
    23: {"name": "RMT_3954_CCD_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 1.0},
    
    24: {"name": "RMT_3954_CCD_Type2_Equal",    "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
    25: {"name": "RMT_3954_CCD_Type2_Unequal1", "dU": 0.0,  "dV": 0.1,  "dW": 1.0},
    26: {"name": "RMT_3954_CCD_Type2_Unequal2", "dU": 0.0,  "dV": 1.0,  "dW": 10.0},
    27: {"name": "RMT_3954_CCD_Type2_Limit",    "dU": 0.0,  "dV": 0.1,  "dW": 0.1},
    
    28: {"name": "RMT_3954_CCD_Type3_Equal",    "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    29: {"name": "RMT_3954_CCD_Type3_Unequal1", "dU": 0.1,  "dV": 1.0,  "dW": 0.0},
    30: {"name": "RMT_3954_CCD_Type3_Unequal2", "dU": 10.0, "dV": 1.0,  "dW": 0.0},
    31: {"name": "RMT_3954_CCD_Type3_Limit",    "dU": 0.1,  "dV": 0.1,  "dW": 0.0},

    # DCI: A=Destable, B=Compl., C=Immobile
    32: {"name": "RMT_3954_DCI_Type1",          "dU": 1.0,  "dV": 10.0, "dW": 0.0},
    33: {"name": "RMT_3954_DCI_Type1_Control",  "dU": 1.0,  "dV": 1.0,  "dW": 0.0},
    34: {"name": "RMT_3954_DCI_Type2",          "dU": 1.0,  "dV": 0.0,  "dW": 0.0},
    35: {"name": "RMT_3954_DCI_Type2_Limit",    "dU": 0.1,  "dV": 0.0,  "dW": 0.0},
    36: {"name": "RMT_3954_DCI_Type3",          "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
    37: {"name": "RMT_3954_DCI_Type3_Limit",    "dU": 0.0,  "dV": 0.1,  "dW": 0.0},

    # # ICD: A=Immobile, B=Compl., C=Destabile
    # 38: {"name": "RMT_3954_ICD_Type1",          "dU": 0.0,  "dV": 10.0, "dW": 1.0},
    # 39: {"name": "RMT_3954_ICD_Type1_Control",  "dU": 0.0,  "dV": 1.0,  "dW": 1.0},
    # 40: {"name": "RMT_3954_ICD_Type2",          "dU": 0.0,  "dV": 0.0,  "dW": 1.0},
    # 41: {"name": "RMT_3954_ICD_Type2_Limit",    "dU": 0.0,  "dV": 0.0,  "dW": 0.1},
    # 42: {"name": "RMT_3954_ICD_Type3",          "dU": 0.0,  "dV": 1.0,  "dW": 0.0},
    # 43: {"name": "RMT_3954_ICD_Type3_Limit",    "dU": 0.0,  "dV": 0.1,  "dW": 0.0},
}

#full range sigma values but to test we do less values
#SIGMA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SIGMA_VALUES = [0.2, 0.6, 1.0, 1.4, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 3.8, 4.0, 6.0, 8.0, 10.0]


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
        
        # Main loop
        for i in range(n_samples):
            J = generate_jacobian_3954(sigma)
            eigs_0 = eigvals(J)
            
            if is_stable(J):
                stable += 1
                
                # Diego method
                if is_turing_diego(J, DU, DV, DW):
                    diego_turing += 1
                
                # Shaberi method
                turing_type = is_turing_shaberi(J, eigs_0, DU, DV, DW)
                
                if turing_type is not None:
                    shaberi_total += 1
                    if turing_type == 'Type-I':
                        shaberi_type_I += 1
                    elif turing_type == 'Type-II':
                        shaberi_type_II += 1
                    elif turing_type == 'Hopf':
                        shaberi_hopf += 1
            
            # Progress indicator
            if (i + 1) % 100000 == 0:
                print(f"  [sig={sigma:.1f}] {i+1:,}/{n_samples:,} | Stable: {stable} | "
                      f"Diego: {diego_turing} | Shaberi: {shaberi_total}")
        
        # Calculate robustness
        rob_diego = 100 * diego_turing / n_samples
        rob_shaberi_total = 100 * shaberi_total / n_samples
        rob_shaberi_type_I = 100 * shaberi_type_I / n_samples
        rob_shaberi_excl_II = 100 * (shaberi_type_I + shaberi_hopf) / n_samples
        
        # rob_diego = 100 * diego_turing / stable if stable > 0 else 0.0
        # rob_shaberi_total = 100 * shaberi_total / stable if stable > 0 else 0.0
        # rob_shaberi_type_I = 100 * shaberi_type_I / stable if stable > 0 else 0.0
        # rob_shaberi_excl_II = 100 * (shaberi_type_I + shaberi_hopf) / stable if stable > 0 else 0.0
        
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
            "rob_diego": rob_diego,
            "rob_shaberi_total": rob_shaberi_total,
            "rob_shaberi_type_I": rob_shaberi_type_I,
            "rob_shaberi_excl_II": rob_shaberi_excl_II,
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
        print("Usage: python 3954-rmt-3node-hpc.py <config_id>")
        sys.exit(1)
    
    config_id = int(sys.argv[1])
    n_samples = 100_000  # 1M samples per sigma value
    
    results = run_analysis(config_id, n_samples)
    
    # Save as pickle
    output_pkl = f"results/{results['config_name']}_{n_samples//1000}k.pkl"
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
            'rob_diego': sigma_result['rob_diego'],
            'rob_shaberi_total': sigma_result['rob_shaberi_total'],
            'rob_shaberi_type_I': sigma_result['rob_shaberi_type_I'],
            'rob_shaberi_excl_II': sigma_result['rob_shaberi_excl_II'],
        }
        csv_rows.append(row)
    
    output_csv = f"results/{results['config_name']}_{n_samples//1000}k.csv"
    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPLETED: {results['config_name']}")
    print(f"{'='*70}")
    print(f"Tested {len(SIGMA_VALUES)} sigma values")
    print(f"{n_samples:,} samples per sigma")
    print(f"Total: {len(SIGMA_VALUES) * n_samples:,} Jacobians generated")
    print(f"\nSaved to:")
    print(f"  {output_pkl}")
    print(f"  {output_csv}")
    print(f"{'='*70}")