import pandas as pd

SUMMARY_CSVS = {
    3954: 'Topology3954/3954_NEWTURINGCLASS_lhs_results_summary.csv',
    1754: 'Topology1754/1754_NEWTURINGCLASS_lhs_results_summary.csv',
}

for topo_id, csv_path in SUMMARY_CSVS.items():   # ← added .items()
    df = pd.read_csv(csv_path)

    total_configs = len(df)
    configs_with_type_I = (df['shaberi_type_I'] > 0).sum()
    configs_with_genuine_turing = (df['shaberi_type_I'] + 
                                    df['shaberi_type_II'] + 
                                    df['shaberi_hopf'] > 0).sum()
    
    fraction_active_type_I = configs_with_type_I / total_configs
    fraction_active_genuine = configs_with_genuine_turing / total_configs
    
    print(f"Topology #{topo_id}:")
    print(f"  Total configs:                 {total_configs}")
    print(f"  Configs with Type-I:           {configs_with_type_I} ({fraction_active_type_I:.2%})")
    print(f"  Configs with genuine Turing:   {configs_with_genuine_turing} ({fraction_active_genuine:.2%})")

for topo_id, csv_path in SUMMARY_CSVS.items():
    df = pd.read_csv(csv_path)
    
    type_I_per_config = df['rob_shaberi_type_I']  # already a percentage per config
    
    print(f"\nTopology #{topo_id} — per-config Type-I robustness statistics:")
    print(f"  Total configs:         {len(df)}")
    print(f"  Mean:                  {type_I_per_config.mean():.4f}")
    print(f"  Median:                {type_I_per_config.median():.4f}")
    print(f"  Max:                   {type_I_per_config.max():.4f}")
    print(f"  Configs with rob > 0:  {(type_I_per_config > 0).sum()}")
    print(f"  Configs with rob > 1%: {(type_I_per_config > 0.01).sum()}")
    print(f"  Configs with rob > 5%: {(type_I_per_config > 0.05).sum()}")