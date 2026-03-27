#!/usr/bin/env python3
"""
Load all pickle results and create summary table
"""
import pickle
import glob
import pandas as pd

# Find all pickle files
result_files = sorted(glob.glob('results/*.pkl'))

# Load all results
all_results = []
for filepath in result_files:
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
        all_results.append(result)

# Create DataFrame
df = pd.DataFrame(all_results)

# Sort by robustness
df = df.sort_values('rob_diego', ascending=False)

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
df.to_csv('results_summary.csv', index=False)
print("\nSaved to: results_summary.csv")

# Print top 3
print("\n" + "="*60)
print("TOP 3 MOST ROBUST CONFIGURATIONS:")
print("="*60)
for idx, row in df.head(3).iterrows():
    print(f"\n{row['config_name']}: {row['rob_diego']:.4f}%")
    print(f"  Diffusion: dA={row['diffusion']['dA']}, dB={row['diffusion']['dB']}, dC={row['diffusion']['dC']}")
    print(f"  Diego: {row['diego_turing']} patterns")
    print(f"  Shaberi: {row['shaberi_type_I']} Type-I, {row['shaberi_type_II']} Type-II, {row['shaberi_hopf']} Hopf")