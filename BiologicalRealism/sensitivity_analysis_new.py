#!/usr/bin/env python3
import numpy as np
import pickle
import pandas as pd
from scipy.optimize import fsolve
from homogenous_ring import (
    CONFIG_TO_TEST,
    build_ring_jacobian_homogeneous,
    compute_jacobian,
    steady_state_expected,
    baseline_params,
    hopping,
    ode_system,
)

# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# run using: python sensitivity_analysis.py

# ============================================================================
# ONE-AT-A-TIME LOCAL SENSITIVITY ANALYSIS
# Uses continuation-based steady-state finding to avoid bistability artifacts
# ============================================================================

PERTURBATION = 0.10  # ±10% perturbation per parameter
LOCAL_TOLERANCE = 0.5  # accept steady states within 50% of baseline (per component)

PARAM_LABELS = {
    'alpha_u': 'u basal production',
    'beta_u': 'u regulated production',
    'K_uu': 'u self-activation (K)',
    'K_vu': 'v-u inhibition (K)',
    'delta_u': 'u degradation',
    
    'alpha_v': 'v basal production',
    'beta_v': 'v regulated production',
    'K_uv': 'u-v activation (K)',
    'K_wv': 'w-v inhibition (K)',
    'delta_v': 'v degradation',
    
    'alpha_w': 'w basal production',
    'beta_w': 'w regulated production',
    'K_ww': 'w self-activation (K)',
    'K_uw': 'u-w inhibition (K)',
    'K_vw': 'v-w inhibition (K)',
    'delta_w': 'w degradation'
}

PARAM_NAMES = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u',
               'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v',
               'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']

N_cells = 10


def find_steady_state_local(params, baseline_ss, tol=LOCAL_TOLERANCE):
    """Find steady state in the same basin as baseline_ss.
    
    Starts fsolve from baseline_ss directly (continuation approach).
    Returns the steady state only if it stays within `tol` relative
    distance of baseline_ss. Otherwise returns None — meaning the
    perturbation has pushed the system out of the local basin (likely
    crossing a bifurcation).
    """
    sol = fsolve(ode_system, baseline_ss, args=(params,), full_output=True)
    ss, info, ier, _ = sol
    residuals = ode_system(ss, params)
    
    if not (ier == 1 and np.max(np.abs(residuals)) < 1e-8 and np.all(ss > 0)):
        return None  # solver failed
    
    # Check that the solution is in the same basin as baseline
    relative_dist = np.max(np.abs(ss - baseline_ss) / baseline_ss)
    if relative_dist > tol:
        return None  # solver found a different fixed point (bifurcation crossed)
    
    return ss


# ============================================================================
# Load config info dynamically
# ============================================================================

df_params = pd.read_csv('../TopologyRanking/Topology3954/3954_NEW_lhs_results_parameters.csv')
config_data = df_params[(df_params['config_id'] == CONFIG_TO_TEST) &
                        (df_params['param_rank'] == 1)].iloc[0]
config_name = config_data['config_name']

# Baseline output (max eigenvalue of homogeneous ring)
J_baseline = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
eigs_baseline = np.linalg.eigvals(J_baseline)
baseline_output = np.max(np.real(eigs_baseline))

if baseline_output <= 0:
    print(f"WARNING: Baseline ring max Re(λ) = {baseline_output:.6f} ≤ 0")
    print("This config does not exhibit Turing instability in the discrete ring.")
    print("Sensitivity analysis will measure response around a stable point, not a Turing peak.")

print(f"Config: {config_name} (id {CONFIG_TO_TEST})")
print(f"Baseline steady state: {steady_state_expected}")
print(f"Baseline max Re(λ): {baseline_output:.6f}")
print(f"Perturbation: ±{PERTURBATION*100:.0f}%")
print(f"Local-basin tolerance: ±{LOCAL_TOLERANCE*100:.0f}% per component\n")

sensitivities_plus = []
sensitivities_minus = []
ss_plus_record = []   # record perturbed steady states for diagnostics
ss_minus_record = []

# Test each parameter
for i, param_name in enumerate(PARAM_NAMES):
    print(f"Testing parameter {i+1}/16: {param_name}")
    
    # =====================================================================
    # TEST +PERTURBATION
    # =====================================================================
    test_params_plus = baseline_params.copy()
    test_params_plus[i] = test_params_plus[i] * (1 + PERTURBATION)
    
    ss_plus = find_steady_state_local(test_params_plus, steady_state_expected)
    
    if ss_plus is None:
        print(f"  +{PERTURBATION*100:.0f}%: no local steady state (bifurcation crossed or solver failed)")
        sensitivity_plus = np.nan
        ss_plus_record.append(None)
    else:
        J_plus = build_ring_jacobian_homogeneous(N_cells, ss_plus, test_params_plus, hopping)
        eigs_plus = np.linalg.eigvals(J_plus)
        output_plus = np.max(np.real(eigs_plus))
        change_plus = abs(output_plus - baseline_output)
        sensitivity_plus = change_plus / PERTURBATION
        print(f"  +{PERTURBATION*100:.0f}%: ss = {ss_plus}")
        print(f"           Δ = {change_plus:.6f}, S = {sensitivity_plus:.6f}")
        ss_plus_record.append(ss_plus)
    
    sensitivities_plus.append(sensitivity_plus)
    
    # =====================================================================
    # TEST -PERTURBATION
    # =====================================================================
    test_params_minus = baseline_params.copy()
    test_params_minus[i] = test_params_minus[i] * (1 - PERTURBATION)
    
    ss_minus = find_steady_state_local(test_params_minus, steady_state_expected)
    
    if ss_minus is None:
        print(f"  -{PERTURBATION*100:.0f}%: no local steady state (bifurcation crossed or solver failed)")
        sensitivity_minus = np.nan
        ss_minus_record.append(None)
    else:
        J_minus = build_ring_jacobian_homogeneous(N_cells, ss_minus, test_params_minus, hopping)
        eigs_minus = np.linalg.eigvals(J_minus)
        output_minus = np.max(np.real(eigs_minus))
        change_minus = abs(output_minus - baseline_output)
        sensitivity_minus = change_minus / PERTURBATION
        print(f"  -{PERTURBATION*100:.0f}%: ss = {ss_minus}")
        print(f"           Δ = {change_minus:.6f}, S = {sensitivity_minus:.6f}")
        ss_minus_record.append(ss_minus)
    
    sensitivities_minus.append(sensitivity_minus)
    
    # Average using nanmean — gives one-sided result if only one direction worked
    avg_sensitivity = np.nanmean([sensitivity_plus, sensitivity_minus])
    if np.isnan(avg_sensitivity):
        print(f"  Average: BOTH directions failed (bifurcation parameter)\n")
    else:
        print(f"  Average: {avg_sensitivity:.6f}\n")

# Convert to arrays for clean NaN handling
sensitivities_plus = np.array(sensitivities_plus)
sensitivities_minus = np.array(sensitivities_minus)
sensitivities = np.nanmean(np.vstack([sensitivities_plus, sensitivities_minus]), axis=0)

# ============================================================================
# Summary
# ============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)

# Categorise parameters
n_clean = np.sum(~np.isnan(sensitivities_plus) & ~np.isnan(sensitivities_minus))
n_one_sided = np.sum(np.isnan(sensitivities_plus) ^ np.isnan(sensitivities_minus))
n_both_failed = np.sum(np.isnan(sensitivities_plus) & np.isnan(sensitivities_minus))

print(f"Parameters with clean local sensitivity (both directions): {n_clean}/16")
print(f"Parameters with one-sided sensitivity:                     {n_one_sided}/16")
print(f"Parameters where both directions cross bifurcation:        {n_both_failed}/16")
print()

# List bifurcation-crossing parameters (potentially the most critical)
print("Bifurcation-crossing parameters (cannot be perturbed locally):")
for i, name in enumerate(PARAM_NAMES):
    plus_failed = np.isnan(sensitivities_plus[i])
    minus_failed = np.isnan(sensitivities_minus[i])
    if plus_failed and minus_failed:
        print(f"  {name}: both directions")
    elif plus_failed:
        print(f"  {name}: +{PERTURBATION*100:.0f}% direction only")
    elif minus_failed:
        print(f"  {name}: -{PERTURBATION*100:.0f}% direction only")
print()

# Ranking of clean sensitivities
print("Sensitivity ranking (clean values only):")
sorted_indices = np.argsort(sensitivities)[::-1]
for rank, idx in enumerate(sorted_indices):
    label = PARAM_LABELS[PARAM_NAMES[idx]]
    val = sensitivities[idx]
    if np.isnan(val):
        print(f"  {rank+1:2d}. {label:35s} -- nonlocal (bifurcation)")
    else:
        print(f"  {rank+1:2d}. {label:35s} {val:.6f}")
print()

# ============================================================================
# Save results
# ============================================================================

sensitivity_data = {
    'param_names': PARAM_NAMES,
    'sensitivities': sensitivities,
    'sensitivities_plus': sensitivities_plus,
    'sensitivities_minus': sensitivities_minus,
    'ss_plus_record': ss_plus_record,
    'ss_minus_record': ss_minus_record,
    'baseline_output': baseline_output,
    'baseline_ss': steady_state_expected,
    'config_name': config_name,
    'config_id': CONFIG_TO_TEST,
    'perturbation': PERTURBATION,
    'local_tolerance': LOCAL_TOLERANCE,
    'n_clean': int(n_clean),
    'n_one_sided': int(n_one_sided),
    'n_both_failed': int(n_both_failed),
}

output_file = f'sensitivity_results_NEW_config{CONFIG_TO_TEST}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(sensitivity_data, f)

print(f"Sensitivity analysis saved to {output_file}")