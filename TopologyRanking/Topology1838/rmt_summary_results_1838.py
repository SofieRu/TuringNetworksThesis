#!/usr/bin/env python3
import pickle
import glob
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a, module load SciPy-bundle/2024.05-gfbf-2024a
result_files = sorted(glob.glob('results/NEW_RMT_1838_*_1mio.pkl'))

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
                'filter_count': sigma_result['filter_count'],
                'rob_diego': sigma_result['rob_diego'],
                'rob_shaberi_total': sigma_result['rob_shaberi_total'],
                'rob_shaberi_type_I': sigma_result['rob_shaberi_type_I'],
            }
            all_rows.append(row)

df = pd.DataFrame(all_rows)
df = df.sort_values(['config_id', 'sigma'], ascending=True)

cols_display = [
    'config_name',
    'config_id',
    'sigma',
    'stable_without_diffusion',
    'diego_turing',
    'shaberi_total',
    'rob_diego',
    'rob_shaberi_total',
    'filter_count'
]

output_file = '1838_FINAL_rmt_results_summary.csv'
df.to_csv(output_file, index=False)