import pandas as pd

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

SUMMARY_CSVS = {
    3954: 'Topology3954/3954_NEWTURINGCLASS_lhs_results_summary.csv',
    1754: 'Topology1754/1754_NEWTURINGCLASS_lhs_results_summary.csv',
}

# ============================================================================
# AGGREGATE PER TOPOLOGY
# ============================================================================

print("="*70)
print("AGGREGATED ROBUSTNESS COMPARISON")
print("="*70)

results = {}

for topo_id, csv_path in SUMMARY_CSVS.items():
    df = pd.read_csv(csv_path)
    
    print(f"\nTopology #{topo_id}: {len(df)} diffusion configurations")
    
    # Sum across all configs
    total_samples = df['n_samples'].sum()
    total_stable = df['stable_without_diffusion'].sum()
    
    total_type_I = df['shaberi_type_I'].sum()
    total_type_II = df['shaberi_type_II'].sum()
    total_hopf = df['shaberi_hopf'].sum()
    total_filter = df['filter_count'].sum()
    total_diego_turing = df['diego_turing'].sum()
    
    total_genuine_turing = total_type_I + total_type_II + total_hopf
    
    # Robustness calculations
    # Scholes-style: instabilities / total samples
    rob_total_scholes = total_genuine_turing / total_samples
    rob_type_I_scholes = total_type_I / total_samples
    rob_with_filters_scholes = (total_genuine_turing + total_filter) / total_samples
    
    # Alternative: instabilities / stable-at-k=0 samples (conditional robustness)
    rob_total_conditional = total_genuine_turing / total_stable
    rob_type_I_conditional = total_type_I / total_stable
    
    results[topo_id] = {
        'total_samples': total_samples,
        'total_stable': total_stable,
        'type_I': total_type_I,
        'type_II': total_type_II,
        'hopf': total_hopf,
        'filter': total_filter,
        'genuine_turing': total_genuine_turing,
        'rob_total_scholes': rob_total_scholes,
        'rob_type_I_scholes': rob_type_I_scholes,
        'rob_with_filters_scholes': rob_with_filters_scholes,
        'rob_total_conditional': rob_total_conditional,
        'rob_type_I_conditional': rob_type_I_conditional,
    }
    
    print(f"  Total samples:           {total_samples:>12,}")
    print(f"  Stable without diffusion: {total_stable:>12,}")
    print(f"  Type-I count:            {total_type_I:>12,}")
    print(f"  Type-II count:           {total_type_II:>12,}")
    print(f"  Hopf count:              {total_hopf:>12,}")
    print(f"  Filter count:            {total_filter:>12,}")
    print(f"  Genuine Turing (I+II+H): {total_genuine_turing:>12,}")


# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("HEADLINE COMPARISON")
print("="*70)

print(f"\n{'Metric':<45} {'#3954':>12} {'#1754':>12} {'Ratio':>8}")
print("-" * 80)

metrics = [
    ('Type-I robustness (Scholes-style)', 'rob_type_I_scholes'),
    ('Genuine Turing robustness (Scholes-style)', 'rob_total_scholes'),
    ('All-including-filters robustness', 'rob_with_filters_scholes'),
    ('Type-I robustness (conditional)', 'rob_type_I_conditional'),
    ('Genuine Turing robustness (conditional)', 'rob_total_conditional'),
]

for label, key in metrics:
    r_3954 = results[3954][key]
    r_1754 = results[1754][key]
    ratio = r_3954 / r_1754 if r_1754 > 0 else float('inf')
    
    print(f"{label:<45} {r_3954*100:>11.3f}% {r_1754*100:>11.3f}% {ratio:>7.2f}x")


# ============================================================================
# COMPARE TO SCHOLES PUBLISHED VALUES
# ============================================================================

print("\n" + "="*70)
print("COMPARISON TO SCHOLES (2019) PUBLISHED VALUES")
print("="*70)

scholes_total = {3954: 0.003833, 1754: 0.001873}
scholes_intracellular = {3954: 0.014750, 1754: 0.007824}

print(f"\n{'Metric':<45} {'#3954':>12} {'#1754':>12} {'Ratio':>8}")
print("-" * 80)

print(f"{'Scholes Total Robustness (published)':<45} "
      f"{scholes_total[3954]*100:>11.3f}% {scholes_total[1754]*100:>11.3f}% "
      f"{scholes_total[3954]/scholes_total[1754]:>7.2f}x")

print(f"{'Scholes Intracellular Robustness (published)':<45} "
      f"{scholes_intracellular[3954]*100:>11.3f}% {scholes_intracellular[1754]*100:>11.3f}% "
      f"{scholes_intracellular[3954]/scholes_intracellular[1754]:>7.2f}x")

print(f"{'Our Type-I robustness (Scholes-style)':<45} "
      f"{results[3954]['rob_type_I_scholes']*100:>11.3f}% "
      f"{results[1754]['rob_type_I_scholes']*100:>11.3f}% "
      f"{results[3954]['rob_type_I_scholes']/results[1754]['rob_type_I_scholes']:>7.2f}x")


# ============================================================================
# VERDICT
# ============================================================================

print("\n" + "="*70)
print("VERDICT")
print("="*70)

ratio_genuine = results[3954]['rob_total_scholes'] / results[1754]['rob_total_scholes']
ratio_type_I = results[3954]['rob_type_I_scholes'] / results[1754]['rob_type_I_scholes']

print(f"\nOn aggregated Type-I robustness:")
if ratio_type_I > 1.0:
    print(f"  ✓ #3954 > #1754 (ratio {ratio_type_I:.2f}x)")
    print(f"  This MATCHES Scholes's ordering.")
else:
    print(f"  ✗ #1754 > #3954 (ratio {1/ratio_type_I:.2f}x in the other direction)")
    print(f"  This CONTRADICTS Scholes's ordering.")

print(f"\nOn aggregated genuine Turing robustness (Type-I + II + Hopf):")
if ratio_genuine > 1.0:
    print(f"  ✓ #3954 > #1754 (ratio {ratio_genuine:.2f}x)")
    print(f"  This MATCHES Scholes's ordering.")
else:
    print(f"  ✗ #1754 > #3954 (ratio {1/ratio_genuine:.2f}x in the other direction)")
    print(f"  This CONTRADICTS Scholes's ordering.")