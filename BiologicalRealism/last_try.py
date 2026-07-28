#!/usr/bin/env python3
"""
Robustness of Turing patterns to parameter heterogeneity -- by DIRECT SIMULATION.

Why simulation instead of linear analysis:
  The linear (dispersion / Jacobian) approach needs a coupled homogeneous base
  state to linearise around and a strict uniform-mode-stable condition. For these
  near-critical configs that base state folds (saddle-node) and the uniform
  margin is razor-thin, so the linear yes/no test collapses to ~0 robustness for
  reasons that have little to do with whether a pattern actually forms.

  Here we instead ask the biological question directly: seed the ring near its
  homogeneous state with a small perturbation, integrate the coupled ODEs
  forward in time, and measure whether a SPATIAL PATTERN emerges (spatial
  coefficient of variation of the final state). No base-state solve, no folds,
  no wavenumber decomposition -- robust to heterogeneity and bistability.

Metric per trial:  amp = max over species of  std_over_cells / mean_over_cells
                   of the final state (a dimensionless pattern amplitude).
Robustness       :  fraction of trials whose amp stays >= RETAIN * (CV=0 amp),
                   i.e. that keep at least RETAIN of the unperturbed pattern.
All amplitudes are saved, so you can re-threshold or just boxplot the
distribution vs CV without committing to a cut-off.

# module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pandas as pd
import pickle

# ---------------------------------------------------------------- settings
TOPOLOGY       = '1754'      # '3954' (u self-activates, 16 params) or '1754' (15 params)
CONFIG_TO_TEST = 49
CONFIG_LABEL   = 'high'
N_cells        = 10
n_trials       = 200
CV_LIST        = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

T_END          = 600.0       # integration time (increase if patterns not saturated)
SEED_NOISE     = 0.02        # initial perturbation around the homogeneous state
RETAIN         = 0.5         # pattern "kept" if amp >= RETAIN * baseline(CV=0) amp
RNG_SEED       = 42

CSV_PATH = f'../TopologyRanking/Topology{TOPOLOGY}/{TOPOLOGY}_FINAL_lhs_results_parameters.csv'

# ---------------------------------------------------------------- kinetics
n = 2
Ha  = lambda X, K: X**n / (K**n + X**n)
Hi  = lambda X, K: K**n / (K**n + X**n)

if TOPOLOGY == '3954':
    NP = 16
    def ode_cell(s, p):
        u, v, w = s
        au,bu,Kuu,Kvu,du = p[0:5]; av,bv,Kuv,Kwv,dv = p[5:10]; aw,bw,Kww,Kuw,Kvw,dw = p[10:16]
        return (au + bu*Ha(u,Kuu)*Hi(v,Kvu) - du*u,
                av + bv*Ha(u,Kuv)*Hi(w,Kwv) - dv*v,
                aw + bw*Ha(w,Kww)*Hi(u,Kuw)*Hi(v,Kvw) - dw*w)
    def params_from_row(row):
        return np.array([row['alpha_u'],row['beta_u'],row['K_uu'],row['K_vu'],row['delta_u'],
                         row['alpha_v'],row['beta_v'],row['K_uv'],row['K_wv'],row['delta_v'],
                         row['alpha_w'],row['beta_w'],row['K_ww'],row['K_uw'],row['K_vw'],row['delta_w']])
elif TOPOLOGY == '1754':
    NP = 15
    def ode_cell(s, p):
        u, v, w = s
        au,bu,Kvu,du = p[0:4]; av,bv,Kuv,Kwv,dv = p[4:9]; aw,bw,Kww,Kuw,Kvw,dw = p[9:15]
        return (au + bu*Hi(v,Kvu) - du*u,
                av + bv*Ha(u,Kuv)*Hi(w,Kwv) - dv*v,
                aw + bw*Ha(w,Kww)*Hi(u,Kuw)*Hi(v,Kvw) - dw*w)
    def params_from_row(row):
        return np.array([row['alpha_u'],row['beta_u'],row['K_vu'],row['delta_u'],
                         row['alpha_v'],row['beta_v'],row['K_uv'],row['K_wv'],row['delta_v'],
                         row['alpha_w'],row['beta_w'],row['K_ww'],row['K_uw'],row['K_vw'],row['delta_w']])
else:
    raise ValueError("TOPOLOGY must be '3954' or '1754'")

# ---------------------------------------------------------------- ring dynamics
def build_L(N, h):
    L = np.zeros((3*N, 3*N))
    for i in range(N):
        idx = 3*i; l = (i-1) % N; r = (i+1) % N
        for s in range(3):
            L[idx+s, idx+s] -= 2*h[s]; L[idx+s, 3*l+s] += h[s]; L[idx+s, 3*r+s] += h[s]
    return L

def ring_rhs(t, X, params_list, L, N):
    F = L @ X
    for i in range(N):
        F[3*i:3*i+3] += ode_cell(X[3*i:3*i+3], params_list[i])
    return F

def simulate(params_list, L, N, x0):
    """Integrate to T_END; return final state (clipped positive)."""
    sol = solve_ivp(ring_rhs, (0.0, T_END), x0, args=(params_list, L, N),
                    method='LSODA', rtol=1e-6, atol=1e-8, t_eval=[T_END])
    Xf = sol.y[:, -1]
    return np.clip(Xf, 0.0, None)

def pattern_amplitude(X, N):
    """max over species of spatial CV (std/mean across cells) of the final state."""
    Xc = X.reshape(N, 3)
    return max(np.std(Xc[:, s]) / (abs(np.mean(Xc[:, s])) + 1e-12) for s in range(3))

# ---------------------------------------------------------------- load config
df = pd.read_csv(CSV_PATH)
df = df[df['classification'] == 'Type-I']
row = df[(df['config_id'] == CONFIG_TO_TEST) & (df['param_rank'] == 1)].iloc[0]
baseline_params = params_from_row(row)
assert len(baseline_params) == NP
baseline_ss = np.array([row['u_star'], row['v_star'], row['w_star']])
h = np.array([row['dU'], row['dV'], row['dW']])
L = build_L(N_cells, h)

print("=" * 70)
print(f"Topology {TOPOLOGY}  config {CONFIG_TO_TEST}  N={N_cells}")
print(f"baseline ss (u*,v*,w*) = {np.round(baseline_ss, 4)}   diffusion = {h}")
print("=" * 70)

rng = np.random.default_rng(RNG_SEED)

def one_trial(params_list):
    x0 = np.tile(baseline_ss, N_cells) * (1.0 + SEED_NOISE * rng.standard_normal(3 * N_cells))
    x0 = np.clip(x0, 1e-9, None)
    return pattern_amplitude(simulate(params_list, L, N_cells, x0), N_cells)

# ---- baseline (CV = 0) pattern amplitude (calibrates the "retain" threshold) ----
base_amps = [one_trial([baseline_params] * N_cells) for _ in range(30)]
A0 = float(np.median(base_amps))
print(f"CV=0 pattern amplitude (median of 30) = {A0:.4f}"
      f"   {'[OK: config patterns]' if A0 > 0.02 else '[WARNING: no pattern at CV=0!]'}")
thresh = RETAIN * A0
print(f"pattern-retained threshold = {thresh:.4f}  ({int(RETAIN*100)}% of baseline)")
print("=" * 70)

# ---- CV sweep ----
results = []
for CV in CV_LIST:
    if CV == 0:
        amps = np.array(base_amps)
    else:
        sigma = np.sqrt(np.log(1 + CV**2)); mu = -sigma**2 / 2
        amps = []
        for _ in range(n_trials):
            pl = [baseline_params * rng.lognormal(mu, sigma, NP) for _ in range(N_cells)]
            amps.append(one_trial(pl))
        amps = np.array(amps)

    robust = 100.0 * np.mean(amps >= thresh)
    results.append({'CV': CV, 'all_amplitude': amps,
                    'mean_amp': float(np.mean(amps)), 'median_amp': float(np.median(amps)),
                    'robustness': robust, 'n_trials': len(amps)})
    print(f"CV={CV:<5} mean amp={np.mean(amps):.4f}  median={np.median(amps):.4f}  "
          f"pattern retained = {robust:.1f}%")

print("\n" + "=" * 60)
print(f"{'CV':<8}{'mean amp':<12}{'median amp':<12}{'robustness %'}")
print("-" * 60)
for r in results:
    print(f"{r['CV']:<8.2f}{r['mean_amp']:<12.4f}{r['median_amp']:<12.4f}{r['robustness']:.1f}")
print("=" * 60)

out = f'{TOPOLOGY}_sim_sweep_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl'
with open(out, 'wb') as f:
    pickle.dump({'results': results, 'baseline_params': baseline_params, 'hopping':
                 {'h_u': h[0], 'h_v': h[1], 'h_w': h[2]}, 'baseline_amp': A0,
                 'config_id': CONFIG_TO_TEST, 'topology': TOPOLOGY,
                 'config_name': row.get('config_name', '')}, f)
print(f"Saved -> {out}")