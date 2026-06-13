import pandas as pd

CSV_PATH = '../TopologyRanking/Topology3954/3954_ALLPARAMSNEW_lhs_results_parameters.csv'
print(f"Reading: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print(f"Loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First row preview:")
print(df.head(1))