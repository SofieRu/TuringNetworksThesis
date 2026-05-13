#!/usr/bin/env python3

import pickle
import glob
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a, module load SciPy-bundle/2024.05-gfbf-2024a

# Find all pickle files
#result_files = sorted(glob.glob('results/*_1000k.pkl'))
result_files = sorted(glob.glob('results/NEW_LHS_3954_*_1mio.pkl'))

# Load all results
all_results = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        all_results.append(result)

# Create DataFrame
df = pd.DataFrame(all_results)

# OLD (sorted by robustness):
#df = df.sort_values('rob_diego', ascending=False)

# NEW (sorted by config_id):
df = df.sort_values('config_id', ascending=True)

# Select columns to display
cols = [
    'config_name',
    'stable_without_diffusion',
    'diego_turing',
    'shaberi_total',
    'shaberi_type_I',
    'shaberi_type_II',
    'shaberi_hopf',
    'rob_diego',
    'rob_shaberi_total',
    'rob_shaberi_type_I'
]

# Print table
print("\n" + "="*120)
print("TOPOLOGY #3954 - TURING PATTERN ROBUSTNESS SUMMARY")
print("="*120)
print(df[cols].to_string(index=False))
print("="*120)

# Save as CSV for Excel
df.to_csv('3954_NEW_lhs_results_summary.csv', index=False)
print("\nSaved to: 3954_NEW_lhs_results_summary.csv")