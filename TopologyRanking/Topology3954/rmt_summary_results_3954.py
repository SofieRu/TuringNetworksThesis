#!/usr/bin/env python3
import pickle
import glob
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a, module load SciPy-bundle/2024.05-gfbf-2024a

# Find all RMT pickle files (only RMT_3954 files, 100k samples)
result_files = sorted(glob.glob('results/FINAL_RMT_3954_*_1mio.pkl'))

# Load all results and flatten sigma results
all_rows = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        config_name = result['config_name']
        config_id = result['config_id']
        dU = result['diffusion']['dU']
        dV = result['diffusion']['dV']
        dW = result['diffusion']['dW']
        
        for sigma_result in result['results_by_sigma']:
            row = {
                'config_name': config_name,
                'config_id': config_id,
                'dU': dU,
                'dV': dV,
                'dW': dW,
                'sigma': sigma_result['sigma'],
                'n_samples': sigma_result['n_samples'],
                'stable_without_diffusion': sigma_result['stable'],
                'diego_turing': sigma_result['diego_turing'],
                'shaberi_total': sigma_result['shaberi_total'],
                'shaberi_type_I': sigma_result['shaberi_type_I'],
                'shaberi_type_II': sigma_result['shaberi_type_II'],
                'shaberi_hopf': sigma_result['shaberi_hopf'],
                'rob_diego': sigma_result['rob_diego'],
                'rob_shaberi_total': sigma_result['rob_shaberi_total'],
                'rob_shaberi_type_I': sigma_result['rob_shaberi_type_I'],
            }
            all_rows.append(row)

print(f"Extracted {len(all_rows)} rows (config × sigma combinations)")

df = pd.DataFrame(all_rows)

# Sort by config_id then sigma
df = df.sort_values(['config_id', 'sigma'], ascending=True)

cols_display = [
    'config_name',
    'config_id',
    'sigma',
    'stable_without_diffusion',
    'diego_turing',
    'shaberi_total',
    'rob_diego',
    'rob_shaberi_total'
]

print("TOPOLOGY #3954 - RMT TURING PATTERN ROBUSTNESS SUMMARY")
print("="*140)
print(df[cols_display].to_string(index=False))

output_file = '3954_FINAL_rmt_results_summary.csv'
df.to_csv(output_file, index=False)