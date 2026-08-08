#!/usr/bin/env python3
"""
Local one-at-a-time (OAT) parameter sensitivity of the Turing growth rate,
Sethna-inspired (log-parameter coordinate).

  - Sensitivity  S_i = |d(maxRe) / d ln(p_i)|  (central difference, ABSOLUTE
    change in the growth rate; NOT normalised by the output, because max Re
    crosses zero at the Turing threshold and dividing by it is unstable).
  - Steady state tracked along the baseline branch by continuation, so large
    but on-branch responses are kept (those are the stiff parameters).
  - A one-sided fold keeps the finite derivative from the other side (with a
    flag) -- it is NOT auto-labelled "most stiff", because a fold is often just
    a concentration hitting zero, which can happen to a sloppy parameter too.
    Only a both-sided fold is reported as regime-changing (NaN).
  - Convergence check at 5/10/20% verifies the ranking is step-size robust
    (Spearman correlation + stability of the top-5 stiff set). 10% is reported.
"""

import numpy as np
import pickle
from scipy.optimize import fsolve
from scipy.stats import spearmanr

# CHANGE THIS import line for 3954 vs 1754
from heterogenous_ring_1754_earlyversion import (
    CONFIG_TO_TEST,
    N_cells, 
    build_ring_jacobian_homogeneous,
    steady_state_expected,
    baseline_params,
    hopping,
    ode_system,
    df,
)

TOPOLOGY_TAG = '1754'  # CHANGE THIS for 3954 vs 1754
CONT_STEPS   = 10 
STEPS        = [0.05, 0.10, 0.15]
REPORT_STEP  = 0.10

PARAM_LABELS = {
    'alpha_u': 'u basal production', 'beta_u': 'u regulated production',
    #'K_uu': 'u self-activation',   # uncomment for 3954
    'K_vu': 'v to u inhibition', 'delta_u': 'u degradation',
    'alpha_v': 'v basal production', 'beta_v': 'v regulated production',
    'K_uv': 'u to v activation', 'K_wv': 'w to v inhibition', 'delta_v': 'v degradation',
    'alpha_w': 'w basal production', 'beta_w': 'w regulated production',
    'K_ww': 'w self-activation', 'K_uw': 'u to w inhibition',
    'K_vw': 'v to w inhibition', 'delta_w': 'w degradation',
}
# CHANGE THIS for 3954 (add 'K_uu' after 'beta_u')
#PARAM_NAMES = ['alpha_u', 'beta_u', 'K_uu', 'K_vu', 'delta_u', 'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v', 'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w'] # 3954
PARAM_NAMES = ['alpha_u', 'beta_u', 'K_vu', 'delta_u', 'alpha_v', 'beta_v', 'K_uv', 'K_wv', 'delta_v', 'alpha_w', 'beta_w', 'K_ww', 'K_uw', 'K_vw', 'delta_w'] # 1754

NPAR = len(PARAM_NAMES)

def ring_maxre(ss, params):
    """Output = max Re(lambda) of the homogeneous ring."""
    J = build_ring_jacobian_homogeneous(N_cells, ss, params, hopping)
    return np.max(np.real(np.linalg.eigvals(J)))

def steady_state_continuation(p0, p1, ss0, n_steps=CONT_STEPS):
    """Track the steady state on the baseline branch from p0 to p1. Returns None
    only if the branch folds (no positive convergent root at a sub-step)."""
    ss = np.array(ss0, dtype=float)
    for a in np.linspace(1.0 / n_steps, 1.0, n_steps):
        p = (1.0 - a) * p0 + a * p1
        sol, info, ier, _ = fsolve(ode_system, ss, args=(p,), full_output=True)
        if not (ier == 1 and np.max(np.abs(ode_system(sol, p))) < 1e-8 and np.all(sol > 0)):
            return None
        ss = sol
    return ss

def output_at(target_params, ss0):
    ss = steady_state_continuation(baseline_params, target_params, ss0)
    return ring_maxre(ss, target_params) if ss is not None else np.nan

def calculate_sensitivities(h, baseline_output):
    """OAT log-parameter sensitivities at log-step h. Returns (values, flags)."""
    vals, flags = [], []
    for i in range(NPAR):
        p_plus  = baseline_params.copy(); p_plus[i]  *= np.exp(+h)
        p_minus = baseline_params.copy(); p_minus[i] *= np.exp(-h)
        out_plus  = output_at(p_plus,  steady_state_expected)
        out_minus = output_at(p_minus, steady_state_expected)

        if np.isnan(out_plus) and np.isnan(out_minus):
            S, flag = np.nan, 'fold-both'                       # regime-changing
        elif np.isnan(out_plus):
            S, flag = abs(out_minus - baseline_output) / h, 'fold+ (one-sided)'
        elif np.isnan(out_minus):
            S, flag = abs(out_plus - baseline_output) / h, 'fold- (one-sided)'
        else:
            S, flag = abs(out_plus - out_minus) / (2.0 * h), 'clean'
        vals.append(S); flags.append(flag)
    return np.array(vals), flags

config_data = df[(df['config_id'] == CONFIG_TO_TEST) & (df['param_rank'] == 1)].iloc[0]
config_name = config_data['config_name']
baseline_output = ring_maxre(steady_state_expected, baseline_params)

# convergence check
results_by_step, flags_by_step = {}, {}
for h in STEPS:
    v, fl = calculate_sensitivities(h, baseline_output)
    results_by_step[h], flags_by_step[h] = v, fl
    top3 = [PARAM_NAMES[i] for i in np.argsort(np.where(np.isnan(v), -np.inf, v))[::-1][:3]]

def top5(v):
    return set(np.argsort(np.where(np.isnan(v), -np.inf, v))[::-1][:5].tolist())

s5, s15 = results_by_step[0.05], results_by_step[0.15]
mask = np.isfinite(s5) & np.isfinite(s15)
rho = spearmanr(s5[mask], s15[mask]).correlation if mask.sum() >= 3 else np.nan
same_top5 = top5(s5) == top5(s15)
print(f"  Spearman rho = {rho:.3f}   top-5 stiff set identical: {same_top5}")

#  final ranking (10%)
sens = results_by_step[REPORT_STEP]
flags = flags_by_step[REPORT_STEP]
order = np.argsort(np.where(np.isnan(sens), -np.inf, sens))[::-1]

print(f"\n final ranking")
for rank, idx in enumerate(order):
    label = PARAM_LABELS.get(PARAM_NAMES[idx], PARAM_NAMES[idx])
    if np.isnan(sens[idx]):
        print(f"  {rank+1:2d}. {label:26s} regime-changing (folds both sides)")
    else:
        note = '' if flags[idx] == 'clean' else f'  [{flags[idx]}]'
        print(f"  {rank+1:2d}. {label:26s} {sens[idx]:.5f}{note}")

out_file = f'{TOPOLOGY_TAG}_sensitivity_results_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
with open(out_file, 'wb') as f:
    pickle.dump({'param_names': PARAM_NAMES,
                 'sensitivities': sens, 
                 'flags': flags,
                 'all_step_data': results_by_step, 
                 'spearman_5_20': rho, 'top5_stable': bool(same_top5),
                 'baseline_output': baseline_output,
                 'config_name': config_name, 'config_id': CONFIG_TO_TEST,
                 'report_step': REPORT_STEP, 'n_cells': N_cells}, f)
print(f"\nSaved -> {out_file}")