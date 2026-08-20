#!/usr/bin/env python3
import pickle
import glob
import pandas as pd

result_files = sorted(glob.glob('results/NEW_LHS_1823_*_1mio.pkl'))

all_results = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        all_results.append(result)
df = pd.DataFrame(all_results)

# NEW (sorted by config_id):
df = df.sort_values('config_id', ascending=True)

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

df.to_csv('1823_PREFINAL_lhs_results_summary.csv', index=False)