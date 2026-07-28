#!/usr/bin/env python3
"""
Diagnose WHY the 3954 coupled base state is discarded under noise.

For each discarded trial the homotopy solver records the failure mechanism:
  concentration->0  : a species crossed zero as noise ramped up
                      (genuine positivity fold -- the base state really leaves
                       the biological, all-positive regime)
  saddle-node fold  : Newton stopped converging AND the ring Jacobian became
                      near-singular (min|eig| ~ 0) -> a real bifurcation where
                      the near-uniform branch collides with another and vanishes
  solver-suspect    : Newton stopped converging BUT the Jacobian was healthy
                      (min|eig| not small) -> the base state probably still
                      exists; the solver failed. If this bucket is large, the
                      discard rate is an artifact, not physics.

Also reports how far the noise ramp got (s_reached in [0,1]) before failure and,
for a few trials, which species/cell went bad.

# module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
import pandas as pd
from collections import Counter

# ---- 3954 kinetics ----
n = 2
Ha  = lambda X, K: X**n / (K**n + X**n)
Hi  = lambda X, K: K**n / (K**n + X**n)
dHa = lambda x, K: n*K**n*x**(n-1)/(K**n + x**n)**2
dHi = lambda x, K: -n*K**n*x**(n-1)/(K**n + x**n)**2

def ode(s, p):
    u, v, w = s
    au,bu,Kuu,Kvu,du = p[0:5]; av,bv,Kuv,Kwv,dv = p[5:10]; aw,bw,Kww,Kuw,Kvw,dw = p[10:16]
    return np.array([au+bu*Ha(u,Kuu)*Hi(v,Kvu)-du*u,
                     av+bv*Ha(u,Kuv)*Hi(w,Kwv)-dv*v,
                     aw+bw*Ha(w,Kww)*Hi(u,Kuw)*Hi(v,Kvw)-dw*w])

def jac(s, p):
    u, v, w = s
    au,bu,Kuu,Kvu,du = p[0:5]; av,bv,Kuv,Kwv,dv = p[5:10]; aw,bw,Kww,Kuw,Kvw,dw = p[10:16]
    J = np.zeros((3,3))
    J[0,0]=bu*dHa(u,Kuu)*Hi(v,Kvu)-du; J[0,1]=bu*Ha(u,Kuu)*dHi(v,Kvu)
    J[1,0]=bv*dHa(u,Kuv)*Hi(w,Kwv); J[1,1]=-dv; J[1,2]=bv*Ha(u,Kuv)*dHi(w,Kwv)
    J[2,0]=bw*Ha(w,Kww)*dHi(u,Kuw)*Hi(v,Kvw); J[2,1]=bw*Ha(w,Kww)*Hi(u,Kuw)*dHi(v,Kvw)
    J[2,2]=bw*dHa(w,Kww)*Hi(u,Kuw)*Hi(v,Kvw)-dw
    return J

def Lop(N, h):
    L = np.zeros((3*N, 3*N))
    for i in range(N):
        idx=3*i; l=(i-1)%N; r=(i+1)%N
        for s in range(3):
            L[idx+s,idx+s]-=2*h[s]; L[idx+s,3*l+s]+=h[s]; L[idx+s,3*r+s]+=h[s]
    return L

def rhs(X, pl, L, N):
    F = L @ X
    for i in range(N): F[3*i:3*i+3] += ode(X[3*i:3*i+3], pl[i])
    return F

def fjac(X, pl, L, N):
    J = L.copy()
    for i in range(N): J[3*i:3*i+3,3*i:3*i+3] += jac(X[3*i:3*i+3], pl[i])
    return J

def newton(X, pl, L, N, it=40):
    X = np.array(X, float)
    for _ in range(it):
        F = rhs(X, pl, L, N)
        if np.linalg.norm(F) < 1e-11: break
        try: dX = np.linalg.solve(fjac(X, pl, L, N), -F)
        except np.linalg.LinAlgError: return None, 'singular'
        X = X + dX
    if np.max(np.abs(rhs(X, pl, L, N))) < 1e-7:
        return (X, 'ok') if np.all(X > 0) else (None, 'nonpositive', X)
    return None, 'no_converge'

def solve_diag(base, noise, L, N, ss0):
    """Homotopy with failure diagnosis. Returns (X, info)."""
    X = np.tile(ss0, N).astype(float); s, ds = 0.0, 1/8
    Xlast, slast = X.copy(), 0.0
    while s < 1 - 1e-9:
        st = min(s+ds, 1.0)
        pl = [base*(nz**st) for nz in noise]
        out = newton(X, pl, L, N)
        if out[0] is None:
            reason = out[1]
            bad_state = out[2] if len(out) > 2 else None
            ds *= 0.5
            if ds < 1e-4:
                pll = [base*(nz**slast) for nz in noise]
                mineig = np.min(np.abs(np.linalg.eigvals(fjac(Xlast, pll, L, N))))
                return None, {'reason': reason, 's': slast, 'min_abs_eig': mineig,
                              'bad_state': bad_state}
            continue
        X, s = out[0], st; Xlast, slast = X.copy(), s
    return X, {'reason': 'ok'}

# ======================================================================
if __name__ == "__main__":
    CONFIG_TO_TEST = 49
    N_cells = 30
    FOLD_TOL = 1e-2          # min|eig| below this => treat as genuine near-singular fold

    df = pd.read_csv('../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv')
    df = df[df['classification'] == 'Type-I']
    row = df[(df['config_id']==CONFIG_TO_TEST) & (df['param_rank']==1)].iloc[0]
    base = np.array([row['alpha_u'],row['beta_u'],row['K_uu'],row['K_vu'],row['delta_u'],
                     row['alpha_v'],row['beta_v'],row['K_uv'],row['K_wv'],row['delta_v'],
                     row['alpha_w'],row['beta_w'],row['K_ww'],row['K_uw'],row['K_vw'],row['delta_w']])
    ss0 = np.array([row['u_star'], row['v_star'], row['w_star']])
    h = np.array([row['dU'], row['dV'], row['dW']]); L = Lop(N_cells, h)

    print("baseline ss (u*,v*,w*):", np.round(ss0, 4), " diffusion:", h)
    print("baseline ode residual:", np.max(np.abs(ode(ss0, base))))
    print("="*78)

    np.random.seed(42)
    for CV in [0.05, 0.10, 0.15]:
        sg = np.sqrt(np.log(1+CV**2)); mu = -sg**2/2
        buckets = Counter(); s_reached = []; eig_at_fold = []; examples = []
        for _ in range(1000):
            noise = [np.random.lognormal(mu, sg, size=16) for _ in range(N_cells)]
            X, info = solve_diag(base, noise, L, N_cells, ss0)
            if info['reason'] == 'ok':
                buckets['converged'] += 1
                continue
            s_reached.append(info['s'])
            if info['reason'] == 'nonpositive':
                buckets['concentration->0'] += 1
                if len(examples) < 3 and info.get('bad_state') is not None:
                    bs = info['bad_state'].reshape(N_cells, 3)
                    cell = np.argmin(bs.min(axis=1)); sp = ['u','v','w'][np.argmin(bs[cell])]
                    examples.append(f"cell {cell} species {sp} -> {bs[cell].min():+.4f}")
            else:  # no_converge or singular
                if info['min_abs_eig'] < FOLD_TOL:
                    buckets['saddle-node fold'] += 1
                else:
                    buckets['solver-suspect'] += 1
                eig_at_fold.append(info['min_abs_eig'])

        disc = 1000 - buckets['converged']
        print(f"CV={CV}:  converged={buckets['converged']}/1000   discarded={disc} ({disc/10:.0f}%)")
        for k in ['concentration->0','saddle-node fold','solver-suspect']:
            if buckets[k]:
                print(f"     {k:<18}: {buckets[k]}")
        if s_reached:
            print(f"     mean noise-ramp reached before failure: s={np.mean(s_reached):.2f} (1.0 = full CV)")
        if eig_at_fold:
            print(f"     mean min|eig| of ring Jacobian at failure: {np.mean(eig_at_fold):.4f} "
                  f"(near 0 => real fold; large => solver)")
        if examples:
            print(f"     example positivity failures: {examples}")
        print("-"*78)

    print("READ: if 'solver-suspect' dominates -> discard rate is an artifact and 3954 is more")
    print("robust than the raw numbers suggest. If 'concentration->0' / 'saddle-node fold'")
    print("dominate -> the base state genuinely dies, and config 49's fragility is real.")