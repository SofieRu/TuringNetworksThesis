#!/usr/bin/env python3
import pickle
import glob
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a, module load SciPy-bundle/2024.05-gfbf-2024a

result_files = sorted(glob.glob('results/NEW_LHS_1754_*_1mio_with_params.pkl'))

# PART 1: SUMMARY CSV (one row per configuration)

all_results = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        
        # Flatten ONLY the summary stats (not the parameter arrays!)
        row = {
            'config_name': result['config_name'],
            'config_id': result['config_id'],
            'dU': result['diffusion']['dU'],
            'dV': result['diffusion']['dV'],
            'dW': result['diffusion']['dW'],
            'n_samples': result['n_samples'],
            'steady_states': result['steady_states'],
            'stable_without_diffusion': result['stable_without_diffusion'],
            'diego_turing': result['diego_turing'],
            'shaberi_total': result['shaberi_total'],
            'shaberi_type_I': result['shaberi_type_I'],
            'shaberi_type_II': result['shaberi_type_II'],
            'shaberi_hopf': result['shaberi_hopf'],
            'rob_diego': result['rob_diego'],
            'rob_shaberi_total': result['rob_shaberi_total'],
            'rob_shaberi_type_I': result['rob_shaberi_type_I'],
        }
        all_results.append(row)

# Create DataFrame
df = pd.DataFrame(all_results)
df = df.sort_values('config_id', ascending=True)

# Save as CSV for Excel
df.to_csv('1754_PREFINAL_lhs_results_summary.csv', index=False)
print("\nSaved to: 1754_PREFINAL_lhs_results_summary.csv")

# PART 2: DETAILED CSV (one row per saved parameter set)

all_params = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        
        # Check if this config saved parameters
        if 'successful_params' in result and result['successful_params']:
            
            for idx, param_set in enumerate(result['successful_params']):
                params_array = param_set['params_array']
                steady_state = param_set['steady_state']
                
                # Create one row with all info
                row = {
                    'config_name': result['config_name'],
                    'config_id': result['config_id'],
                    'dU': result['diffusion']['dU'],
                    'dV': result['diffusion']['dV'],
                    'dW': result['diffusion']['dW'],
                    'param_rank': idx + 1,  # 1st best, 2nd best, etc.
                    'max_growth_rate': param_set['max_growth_rate'],
                    
                    # Parameters (16 values)
                    'alpha_u': params_array[0],
                    'beta_u': params_array[1],
                    #'K_uu': params_array[2],
                    'K_vu': params_array[2],
                    'delta_u': params_array[3],
                    'alpha_v': params_array[4],
                    'beta_v': params_array[5],
                    'K_uv': params_array[6],
                    'K_wv': params_array[7],
                    'delta_v': params_array[8],
                    'alpha_w': params_array[9],
                    'beta_w': params_array[10],
                    'K_ww': params_array[11],
                    'K_uw': params_array[12],
                    'K_vw': params_array[13],
                    'delta_w': params_array[14],
                    
                    # Steady state
                    'u_star': steady_state[0],
                    'v_star': steady_state[1],
                    'w_star': steady_state[2],
                }
                all_params.append(row)

if all_params:
    df_params = pd.DataFrame(all_params)
    df_params = df_params.sort_values(['config_id', 'param_rank'], ascending=True)
    df_params.to_csv('1754_PREFINAL_lhs_results_parameters.csv', index=False)
    print("Saved to: 1754_PREFINAL_lhs_results_parameters.csv")
else:
    print("No successful parameter sets found in the results.")