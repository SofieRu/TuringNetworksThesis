#!/usr/bin/env python3
import numpy as np
import pickle
import pandas as pd

from homogenous_ring import (
    CONFIG_TO_TEST,
    build_ring_jacobian_homogeneous,
    compute_jacobian,
    steady_state_expected,
    baseline_params,
    hopping,
    find_steady_state,
)


# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# run using: python sensitivity_analysis.py

# ============================================================================
# ONE-AT-A-TIME SENSITIVITY ANALYSIS
# ============================================================================

PERTURBATION = 0.05  # ±5% perturbation per parameter (was 0.10)

PARAM_LABELS = {
    'alpha_u': 'u basal production',
    'beta_u': 'u regulated production',
    'K_uu': 'u self-activation',
    'K_vu': 'v to u inhibition',
    'delta_u': 'u degradation',
    'alpha_v': 'v basal production',
    'beta_v': 'v regulated production',
    'K_uv': 'u to v activation',
    'K_wv': 'w to v inhibition',
    'delta_v': 'v degradation',
    'alpha_w': 'w basal production',
    'beta_w': 'w regulated production',
    'K_ww': 'w self-activation',
    'K_uw': 'u to w inhibition',
    'K_vw': 'v to w inhibition',
    'delta_w': 'w degradation'
}

PARAM_NAMES = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u', 'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v', 'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w']

N_cells = 10

# Baseline output (max eigenvalue)
J_baseline = build_ring_jacobian_homogeneous(N_cells, steady_state_expected, baseline_params, hopping)
eigs_baseline = np.linalg.eigvals(J_baseline)
baseline_output = np.max(np.real(eigs_baseline))

print(f"Baseline max Re(λ): {baseline_output:.6f}\n")

sensitivities_plus = []
sensitivities_minus = []

# Test each parameter
for i, param_name in enumerate(PARAM_NAMES):
    print(f"Testing parameter {i+1}/16: {param_name}")
    
    # TEST +PERTURBATION
    test_params_plus = baseline_params.copy()
    test_params_plus[i] = test_params_plus[i] * (1 + PERTURBATION)
    
    # Find steady state with perturbed params
    ss_plus = find_steady_state(test_params_plus)
    
    if ss_plus is None:
        # Can't find steady state - use baseline
        # params_i = baseline_params.copy()
        # ss_plus = steady_state_expected.copy()
        print(f"  +{PERTURBATION*100:.0f}%: nonviable (no steady state found)")
        sensitivity_plus = np.nan
    else:
        # Compute Jacobian and eigenvalue
        J_plus = build_ring_jacobian_homogeneous(N_cells, ss_plus, test_params_plus, hopping)
        eigs_plus = np.linalg.eigvals(J_plus)
        output_plus = np.max(np.real(eigs_plus))
        # Calculate sensitivity
        change_plus = abs(output_plus - baseline_output)
        sensitivity_plus = change_plus / PERTURBATION # sensitivity_plus = change_plus / 0.1  # Divided by 10% change
        print(f"  +{PERTURBATION*100:.0f}%: Δ={change_plus:.6f}, S={sensitivity_plus:.6f}")
    
    sensitivities_plus.append(sensitivity_plus)
    
    # TEST -PERTURBATION
    test_params_minus = baseline_params.copy()
    test_params_minus[i] = test_params_minus[i] * (1 - PERTURBATION)
    
    ss_minus = find_steady_state(test_params_minus)

    if ss_minus is None:
        # ss_minus = steady_state_expected.copy()
        sensitivity_minus = np.nan
        print(f"  -{PERTURBATION*100:.0f}%: nonviable (no steady state found)")
    else:
        J_minus = build_ring_jacobian_homogeneous(N_cells, ss_minus, test_params_minus, hopping)
        eigs_minus = np.linalg.eigvals(J_minus)
        output_minus = np.max(np.real(eigs_minus))
        change_minus = abs(output_minus - baseline_output)
        sensitivity_minus = change_minus / PERTURBATION # sensitivity_minus = change_minus / 0.1
        print(f"  -{PERTURBATION*100:.0f}%: Δ={change_minus:.6f}, S={sensitivity_minus:.6f}")
    
    sensitivities_minus.append(sensitivity_minus)
    
    # Average sensitivity (symmetric measure)
    # avg_sensitivity = (sensitivity_plus + sensitivity_minus) / 2
    avg_sensitivity = np.nanmean([sensitivity_plus, sensitivity_minus])
    
    # print(f"+10%: Δ={change_plus:.6f}, S={sensitivity_plus:.6f}")
    # print(f"-10%: Δ={change_minus:.6f}, S={sensitivity_minus:.6f}")
    # print(f"Average: {avg_sensitivity:.6f}\n")

# Average sensitivities
# sensitivities = [(s_plus + s_minus)/2 for s_plus, s_minus in zip(sensitivities_plus, sensitivities_minus)]
sensitivities_plus = np.array(sensitivities_plus)
sensitivities_minus = np.array(sensitivities_minus)
sensitivities = np.nanmean(np.vstack([sensitivities_plus, sensitivities_minus]), axis=0)

# Save results
# sensitivity_data = {
#     'param_names': PARAM_NAMES,
#     'sensitivities': sensitivities,
#     'sensitivities_plus': sensitivities_plus,
#     'sensitivities_minus': sensitivities_minus,
#     'baseline_output': baseline_output,
#     'config_name': 'NEW_LHS_3954_Type2_V2_Unequal3',
#     'config_id': CONFIG_TO_TEST
# }
sensitivity_data = {
    'param_names': PARAM_NAMES,
    'sensitivities': sensitivities,
    'sensitivities_plus': sensitivities_plus,
    'sensitivities_minus': sensitivities_minus,
    'baseline_output': baseline_output,
    'config_name': 'NEW_LHS_3954_Type2_V2_Unequal3',
    'config_id': CONFIG_TO_TEST,
    'perturbation': PERTURBATION,
    'n_nan_plus': int(np.sum(np.isnan(sensitivities_plus))),
    'n_nan_minus': int(np.sum(np.isnan(sensitivities_minus))),
}

output_file = f'sensitivity_results_config{CONFIG_TO_TEST}.pkl'

with open(output_file, 'wb') as f:
    pickle.dump(sensitivity_data, f)

print(f"Sensitivity analysis completed and saved to {output_file}")