#!/usr/bin/env python3
"""
Heterogeneous ring analysis for 3-node Turing GRNs (topology 3954).

Three things are computed per noise realisation:

  1. COUPLED STEADY STATE of the full 3N ring, found by damped Newton in
     log-coordinates with homotopy continuation in the noise amplitude.
     If the branch folds (saddle-node), that is recorded as a result, not
     as a failure.

  2. MODE-RESOLVED DISPERSION at that steady state: every eigenvalue of the
     3N x 3N ring Jacobian is assigned the dominant spatial wavenumber of its
     eigenvector. On a homogeneous ring this reproduces the textbook
     dispersion relation exactly; on a heterogeneous ring it is the correct
     generalisation, because translation invariance is broken and the Fourier
     modes no longer block-diagonalise the Jacobian.

  3. NONLINEAR OUTCOME: the ring ODEs are integrated to steady state and the
     surviving pattern is characterised by amplitude and dominant wavenumber.
     This is the observable that stays well defined at every noise level.

module load SciPy-bundle/2024.05-gfbf-2024a
"""

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

N_HILL = 2

# ----------------------------------------------------------------------
# reaction kinetics (unchanged from objective 1)
# ----------------------------------------------------------------------

def hill_activation(X, K):
    return X**N_HILL / (K**N_HILL + X**N_HILL)

def hill_inhibition(X, K):
    return K**N_HILL / (K**N_HILL + X**N_HILL)

def dH_act(x, K):
    return N_HILL * K**N_HILL * x**(N_HILL - 1) / (K**N_HILL + x**N_HILL)**2

def dH_inh(x, K):
    return -N_HILL * K**N_HILL * x**(N_HILL - 1) / (K**N_HILL + x**N_HILL)**2

def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    du = alpha_u + beta_u * hill_activation(u, K_uu) * hill_inhibition(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = (alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw)
          * hill_inhibition(v, K_vw) - delta_w * w)
    return np.array([du, dv, dw])

def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    J = np.zeros((3, 3))
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[0, 2] = 0.0
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = (beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw)
               * hill_inhibition(v, K_vw) - delta_w)
    return J


def find_steady_state(params, n_attempts=100, rng=None, reference=None):
    """Positive root of the single-cell reaction system.

    If `reference` is given, all roots found are collected and the one closest
    to `reference` in log-concentration is returned. Without that, a multistable
    parameter set hands back whichever branch fsolve happened to land on, which
    injects branch-switching noise that has nothing to do with the intended
    parameter perturbation.
    """
    rng = np.random.default_rng() if rng is None else rng
    roots = []
    for _ in range(n_attempts):
        guess = 10**rng.uniform(-2, 1, 3)
        s, _, ier, _ = fsolve(ode_system, guess, args=(params,), full_output=True)
        if ier == 1 and np.max(np.abs(ode_system(s, params))) < 1e-10 and np.all(s > 0):
            if not any(np.allclose(s, r, rtol=1e-6, atol=1e-10) for r in roots):
                roots.append(s)
            if reference is None:
                return s
    if not roots:
        return None
    if reference is None:
        return roots[0]
    d = [np.linalg.norm(np.log(r) - np.log(reference)) for r in roots]
    return roots[int(np.argmin(d))]


# ----------------------------------------------------------------------
# continuum classification (objective 1, kept for cross-checking)
# ----------------------------------------------------------------------

def classify_continuum(J, D, k_max=10.0, dk=0.01):
    if np.max(np.real(np.linalg.eigvals(J))) >= 0:
        return None, None, None
    ks = np.arange(dk, k_max + dk, dk)
    max_re = np.empty(ks.size)
    complex_unstable = False
    for i, k in enumerate(ks):
        e = np.linalg.eigvals(J - k**2 * D)
        max_re[i] = np.max(np.real(e))
        if max_re[i] > 0 and np.any(np.abs(np.imag(e[np.real(e) > 0])) > 1e-8):
            complex_unstable = True
    if np.max(max_re) <= 0:
        return None, ks, max_re
    if complex_unstable:
        return "Hopf", ks, max_re
    if max_re[-1] < 0:
        return "Type-I", ks, max_re
    if np.argmax(max_re) >= ks.size - 2:
        return "Filter", ks, max_re
    return "Type-II", ks, max_re


# ----------------------------------------------------------------------
# ring operators
# ----------------------------------------------------------------------

def build_diffusion_operator(N_cells, hopping):
    """Discrete Laplacian (L (x) D) on a periodic ring, lattice spacing 1."""
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    L = np.zeros((3 * N_cells, 3 * N_cells))
    for i in range(N_cells):
        left, right = (i - 1) % N_cells, (i + 1) % N_cells
        for s in range(3):
            L[3 * i + s, 3 * i + s] -= 2 * h[s]
            L[3 * i + s, 3 * left + s] += h[s]
            L[3 * i + s, 3 * right + s] += h[s]
    return L


def k_effective(N_cells):
    """Wavenumbers a ring of N cells can actually carry, k_m = 2 sin(pi m / N).

    The largest is k = 2 regardless of N. Any continuum Turing band whose peak
    sits above k = 2 is invisible to this lattice unless the hopping rates are
    rescaled or the lattice spacing is chosen differently.
    """
    return 2 * np.sin(np.pi * np.arange(N_cells // 2 + 1) / N_cells)


def ring_rhs(x, params_list, Ldiff, N_cells):
    X = x.reshape(N_cells, 3)
    reaction = np.concatenate([ode_system(X[i], params_list[i]) for i in range(N_cells)])
    return reaction + Ldiff @ x


def ring_jacobian(x, params_list, Ldiff, N_cells):
    J = Ldiff.copy()
    X = x.reshape(N_cells, 3)
    for i in range(N_cells):
        J[3 * i:3 * i + 3, 3 * i:3 * i + 3] += compute_jacobian(X[i], params_list[i])
    return J


# ----------------------------------------------------------------------
# coupled steady state: damped Newton in log-coordinates
# ----------------------------------------------------------------------

def newton_log(x0, params_list, Ldiff, N_cells, tol=1e-10, itmax=60, step_cap=0.5):
    """Solve R(x) + (L(x)D) x = 0 for x > 0.

    Newton converges to saddles as happily as to sinks, so a Turing-unstable
    ring is not an obstacle. Working in y = log x keeps concentrations positive;
    the backtracking line search keeps it from jumping off the branch.
    """
    y = np.log(x0)
    f = ring_rhs(np.exp(y), params_list, Ldiff, N_cells)
    nf = np.linalg.norm(f)
    stall = 0
    for _ in range(itmax):
        if nf < tol:
            return np.exp(y), nf, True
        x = np.exp(y)
        Jy = ring_jacobian(x, params_list, Ldiff, N_cells) * x[None, :]
        try:
            dy = np.linalg.solve(Jy, -f)
        except np.linalg.LinAlgError:
            return np.exp(y), nf, False
        t = min(1.0, step_cap / max(np.max(np.abs(dy)), 1e-300))
        for _ in range(25):
            yt = y + t * dy
            ft = ring_rhs(np.exp(yt), params_list, Ldiff, N_cells)
            if np.all(np.isfinite(ft)) and np.linalg.norm(ft) < nf * (1 - 1e-4 * t):
                nf_new = np.linalg.norm(ft)
                stall = stall + 1 if nf_new > 0.9 * nf else 0
                y, f, nf = yt, ft, nf_new
                break
            t *= 0.5
        else:
            return np.exp(y), nf, False
        if stall >= 6:                      # crawling: we are sitting on the fold
            return np.exp(y), nf, False
    return np.exp(y), nf, nf < tol


def coupled_steady_state(noise, baseline_params, x_uniform, Ldiff, N_cells,
                         steps=20, refine=6):
    """Track the near-uniform branch from CV = 0 up to the full noise amplitude.

    params_i(s) = baseline * noise_i**s, s: 0 -> 1. At s = 0 the exact solution
    is the uniform state, so the continuation starts from a known point.
    Returns (x, s_reached). s_reached < 1 means the branch folded before the
    requested noise level was reached.
    """
    x = x_uniform.copy()
    s_prev = 0.0
    for s in np.linspace(0, 1, steps + 1)[1:]:
        plist = [baseline_params * noise[i]**s for i in range(N_cells)]
        xn, res, ok = newton_log(x, plist, Ldiff, N_cells)
        if not ok:
            lo, hi = s_prev, s
            for _ in range(refine):
                mid = 0.5 * (lo + hi)
                plist2 = [baseline_params * noise[i]**mid for i in range(N_cells)]
                xm, res2, ok2 = newton_log(x, plist2, Ldiff, N_cells)
                if ok2:
                    x, lo = xm, mid
                else:
                    hi = mid
            return None, hi
        x, s_prev = xn, s
    return x, 1.0


# ----------------------------------------------------------------------
# mode-resolved dispersion relation
# ----------------------------------------------------------------------

def mode_resolved_spectrum(J_ring, N_cells):
    """Every eigenvalue tagged with the dominant wavenumber index of its eigenvector.

    Homogeneous ring: each eigenvector is a pure Fourier mode, so this returns the
    classical dispersion relation. Heterogeneous ring: eigenvectors are mixtures,
    the dominant index is the honest label, and `purity` says how much to trust it.
    """
    lam, V = np.linalg.eig(J_ring)
    Vc = V.reshape(N_cells, 3, -1)
    power = np.abs(np.fft.fft(Vc, axis=0))**2          # (N, 3, 3N)
    power = power.sum(axis=1)                           # (N, 3N)
    n_half = N_cells // 2 + 1
    folded = power[:n_half].copy()
    for m in range(1, (N_cells + 1) // 2):
        folded[m] += power[N_cells - m]
    folded /= folded.sum(axis=0, keepdims=True)
    m_dom = np.argmax(folded, axis=0)
    purity = folded[m_dom, np.arange(folded.shape[1])]
    return lam, m_dom, purity


def dispersion_envelope(lam, m_dom, N_cells):
    """Largest Re(lambda) per wavenumber bin; -inf where no eigenvalue lands."""
    env = np.full(N_cells // 2 + 1, -np.inf)
    for l, m in zip(lam, m_dom):
        env[m] = max(env[m], np.real(l))
    return env


def is_turing_ring(env):
    """Uniform mode stable, at least one finite wavenumber unstable."""
    finite = env[1:][np.isfinite(env[1:])]
    return bool(np.isfinite(env[0]) and env[0] < 0 and finite.size and np.max(finite) > 0)


# ----------------------------------------------------------------------
# legacy route, kept only so the methodological artefact can be quantified
# ----------------------------------------------------------------------

def fourier_projectors(N_cells):
    projs = []
    for m in range(N_cells // 2 + 1):
        phi = np.exp(2j * np.pi * m * np.arange(N_cells) / N_cells) / np.sqrt(N_cells)
        P = np.zeros((3 * N_cells, 3), dtype=complex)
        for j in range(N_cells):
            for s in range(3):
                P[3 * j + s, s] = phi[j]
        projs.append(P)
    return projs


def galerkin_dispersion(J_ring, projectors):
    """P_m^H J P_m, one 3x3 block per wavenumber.

    Exact on a homogeneous ring. On a heterogeneous ring the off-diagonal blocks
    P_m^H J P_m' are non-zero and are silently discarded here, so the result is a
    restriction of a non-normal operator with no bounding property. Use only to
    reproduce the older numbers, never as a stability criterion.
    """
    return np.array([np.max(np.real(np.linalg.eigvals(P.conj().T @ J_ring @ P)))
                     for P in projectors])


def frozen_coefficient_state(params_list, N_cells, reference, rng, n_attempts=100):
    """Concatenated isolated single-cell fixed points.

    This is NOT a fixed point of the coupled ring: the diffusive flux
    (L (x) D) x* does not vanish. Returned together with that residual so the
    inconsistency can be reported rather than hidden.
    """
    ss = []
    for i in range(N_cells):
        s = find_steady_state(params_list[i], n_attempts=n_attempts, rng=rng,
                              reference=reference)
        if s is None:
            return None
        ss.append(s)
    return np.concatenate(ss)


# ----------------------------------------------------------------------
# nonlinear outcome
# ----------------------------------------------------------------------

def simulate_ring(params_list, Ldiff, N_cells, x_init, t_end=5000.0,
                  rtol=1e-6, atol=1e-9, stages=(50.0, 200.0, 1000.0, 5000.0),
                  settle_tol=1e-6):
    """Integrate the ring to its attractor.

    Two things make this fast enough to run 200 trials per CV: the analytic
    Jacobian is handed to the stiff solver instead of being finite-differenced
    (30 extra RHS evaluations per Jacobian, repeatedly), and the integration
    stops as soon as the profile stops moving instead of always running to
    t_end.
    """
    f = lambda t, x: ring_rhs(x, params_list, Ldiff, N_cells)
    jf = lambda t, x: ring_jacobian(x, params_list, Ldiff, N_cells)
    x = np.asarray(x_init, dtype=float)
    t0 = 0.0
    for t1 in [s for s in stages if s <= t_end] + ([t_end] if t_end not in stages else []):
        if t1 <= t0:
            continue
        sol = solve_ivp(f, (t0, t1), x, method="LSODA", jac=jf,
                        rtol=rtol, atol=atol, t_eval=[t1])
        if not sol.success:
            return None
        x_new = sol.y[:, -1]
        if not np.all(np.isfinite(x_new)):
            return None
        drift = np.max(np.abs(x_new - x) / (np.abs(x) + 1e-12))
        x, t0 = x_new, t1
        if drift < settle_tol:
            break
    return x


def pattern_metrics(x, N_cells, reference, species=0):
    """Turn a final ring profile into the two Turing conditions plus fidelity.

    condition 1 (uniform mode did not run away)  -> bulk_drift
    condition 2 (finite wavenumber grew)         -> amplitude, m_dom > 0
    condition 3 (the intended wavelength won)    -> m_dom == m_star
    """
    X = x.reshape(N_cells, 3)
    prof = X[:, species]
    mean = prof.mean()
    amp = (prof.max() - prof.min()) / abs(mean) if mean != 0 else np.inf
    bulk_drift = np.max(np.abs(X.mean(axis=0) - reference) / np.abs(reference))
    # note: a saturated Turing pattern always shifts the spatial mean, so
    # bulk_drift is descriptive only. It is NOT a test of the m=0 condition.
    P = np.abs(np.fft.fft(prof - mean))**2
    n_half = N_cells // 2 + 1
    folded = P[:n_half].copy()
    for m in range(1, (N_cells + 1) // 2):
        folded[m] += P[N_cells - m]
    m_dom = int(np.argmax(folded[1:]) + 1) if folded[1:].sum() > 0 else 0
    spec = folded / folded.sum() if folded.sum() > 0 else folded
    return {"amp": amp, "m_dom": m_dom, "bulk_drift": bulk_drift, "spectrum": spec}


def single_cell_turing_fraction(params_list, hopping, rng, reference):
    """Scholes-style intracellular robustness: would each cell still be Turing
    capable on its own? Separates 'the kinetics broke' from 'the ring broke'."""
    D = np.diag([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    ok = 0
    for p in params_list:
        ss = find_steady_state(p, n_attempts=40, rng=rng, reference=reference)
        if ss is None:
            continue
        cls, _, _ = classify_continuum(compute_jacobian(ss, p), D, k_max=4.0, dk=0.02)
        if cls == "Type-I":
            ok += 1
    return ok / len(params_list)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run(baseline_params, steady_state_expected, hopping, N_cells=10, n_trials=200,
        n_init=4, amp_tol=0.05, drift_tol=0.25, per_cell_check=False,
        cv_grid=(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4), seed=42,
        do_nonlinear=True, verbose=True):

    Ldiff = build_diffusion_operator(N_cells, hopping)
    x_uniform = np.tile(steady_state_expected, N_cells)
    keff = k_effective(N_cells)
    D = np.diag([hopping["h_u"], hopping["h_v"], hopping["h_w"]])

    # --- sanity checks on the baseline -------------------------------------
    res_ss = np.max(np.abs(ode_system(steady_state_expected, baseline_params)))
    cls, _, _ = classify_continuum(compute_jacobian(steady_state_expected, baseline_params), D)
    J0 = ring_jacobian(x_uniform, [baseline_params] * N_cells, Ldiff, N_cells)
    lam0, m0, pur0 = mode_resolved_spectrum(J0, N_cells)
    env0 = dispersion_envelope(lam0, m0, N_cells)
    m_star = int(np.argmax(np.where(np.isfinite(env0), env0, -np.inf)))

    if verbose:
        print("=" * 78)
        print("BASELINE")
        print("=" * 78)
        print(f"residual of tabulated steady state : {res_ss:.3e}")
        print(f"continuum classification           : {cls}")
        print(f"ring residual at uniform state     : "
              f"{np.max(np.abs(ring_rhs(x_uniform, [baseline_params]*N_cells, Ldiff, N_cells))):.3e}")
        print(f"accessible wavenumbers k_m         : {np.round(keff, 4)}")
        print(f"ring dispersion envelope           : {np.round(env0, 5)}")
        print(f"ring is Turing                     : {is_turing_ring(env0)}  "
              f"(dominant mode m = {m_star}, k = {keff[m_star]:.4f})")
        print(f"least unstable positive eigenvalue : "
              f"{np.min(np.real(lam0)[np.real(lam0) > 0]) if np.any(np.real(lam0) > 0) else float('nan'):.5f}")
        print()

    rng = np.random.default_rng(seed)
    results = []

    for CV in cv_grid:
        sigma = np.sqrt(np.log(1 + CV**2))
        mu = -sigma**2 / 2

        n_exist = n_fold = n_turing_lin = 0
        env_stack, m_lead, growth_lead = [], [], []
        s_fold = []
        nl_amp, nl_mode, nl_drift, nl_ok = [], [], [], 0
        c1 = c2 = c3 = 0
        classes = {}
        nl_split = []
        percell = []

        trials = 1 if CV == 0 else n_trials
        for it in range(trials):
            if verbose and trials > 1 and it and it % 25 == 0:
                print(f"    ... trial {it}/{trials}", flush=True)
            noise = (np.ones((N_cells, 16)) if CV == 0
                     else rng.lognormal(mu, sigma, (N_cells, 16)))
            plist = [baseline_params * noise[i] for i in range(N_cells)]

            x_ss, s_reached = coupled_steady_state(noise, baseline_params, x_uniform,
                                                   Ldiff, N_cells)
            if x_ss is None:
                n_fold += 1
                s_fold.append(s_reached)
            else:
                n_exist += 1
                J = ring_jacobian(x_ss, plist, Ldiff, N_cells)
                lam, md, pur = mode_resolved_spectrum(J, N_cells)
                env = dispersion_envelope(lam, md, N_cells)
                env_stack.append(env)
                if is_turing_ring(env):
                    n_turing_lin += 1
                i_lead = int(np.argmax(np.real(lam)))
                m_lead.append(md[i_lead])
                growth_lead.append(np.real(lam[i_lead]))

            if do_nonlinear:
                outcomes = []
                for _ in range(n_init):
                    x_init = x_uniform * (1 + 1e-3 * rng.standard_normal(3 * N_cells))
                    xf = simulate_ring(plist, Ldiff, N_cells, x_init)
                    if xf is None or not np.all(np.isfinite(xf)) or not np.all(xf > 0):
                        continue
                    outcomes.append(pattern_metrics(xf, N_cells, steady_state_expected))
                if outcomes:
                    nl_ok += 1
                    # outcome classes, majority vote over initial conditions
                    def cls_one(o):
                        if o["amp"] > amp_tol and o["m_dom"] > 0:
                            return "patterned_correct" if o["m_dom"] == m_star else "patterned_wrong"
                        return ("flat_baseline" if o["bulk_drift"] < drift_tol
                                else "flat_switched")
                    labs = [cls_one(o) for o in outcomes]
                    lab = max(set(labs), key=labs.count)
                    classes[lab] = classes.get(lab, 0) + 1
                    if lab.startswith("patterned"):
                        c2 += 1
                        if lab == "patterned_correct":
                            c3 += 1
                    else:
                        c1 += 1          # circuit produced no pattern at all
                    nl_amp.append(np.mean([o["amp"] for o in outcomes]))
                    nl_drift.append(np.mean([o["bulk_drift"] for o in outcomes]))
                    nl_mode.append(int(np.bincount([o["m_dom"] for o in outcomes]).argmax()))
                    nl_split.append(len(set(labs)) > 1)
                if per_cell_check:
                    percell.append(single_cell_turing_fraction(
                        plist, hopping, rng, steady_state_expected))

        env_stack = np.array(env_stack) if env_stack else np.zeros((0, N_cells // 2 + 1))
        with np.errstate(invalid="ignore"):
            env_mean = (np.where(np.isfinite(env_stack), env_stack, np.nan)
                        if env_stack.size else env_stack)
            env_mean = np.nanmean(env_mean, axis=0) if env_stack.size else None

        nl_amp, nl_mode = np.array(nl_amp), np.array(nl_mode)
        nl_drift = np.array(nl_drift)

        r = {
            "CV": CV,
            "n_trials": trials,
            "nl_ok": nl_ok,
            # the two Turing conditions, counted per trial
            "n_no_pattern": int(c1),
            "n_turing_nonlinear": int(c2),
            "n_correct_mode": int(c3),
            "outcome_classes": classes,
            "frac_ic_dependent": float(np.mean(nl_split)) if nl_split else np.nan,
            "robustness": 100 * c2 / trials if trials else np.nan, #100 * c2 / nl_ok if nl_ok else np.nan,
            "fidelity": 100 * c3 / trials if trials else np.nan,
            # linear picture, where the base state still exists
            "ss_exists": n_exist,
            "ss_folded": n_fold,
            "fold_rate": 100 * n_fold / trials,
            "mean_s_fold": float(np.mean(s_fold)) if s_fold else np.nan,
            "n_turing_linear": n_turing_lin,
            "turing_rate_linear": 100 * n_turing_lin / n_exist if n_exist else np.nan,
            "env_mean": env_mean,
            "envelopes": env_stack,
            "m_lead": np.array(m_lead),
            "growth_lead": np.array(growth_lead),
            # raw distributions for the thesis figures
            "nl_amp": nl_amp,
            "nl_mode": nl_mode,
            "nl_drift": nl_drift,
            "per_cell_turing": np.array(percell),
        }
        results.append(r)

        if verbose:
            print(f"CV = {CV:.2f}")
            if nl_ok:
                print(f"  ROBUSTNESS (pattern forms)   : {c2}/{trials} "
                      f"({r['robustness']:.1f}%)")
                print(f"  FIDELITY   (correct m={m_star})      : {c3}/{trials} "
                      f"({r['fidelity']:.1f}%)")
                print(f"  outcome classes              : {classes}")
                print(f"  initial-condition dependent  : "
                      f"{100*r['frac_ic_dependent']:.1f}% of trials")
            if percell:
                print(f"  per-cell Type-I fraction     : {np.mean(percell):.3f}")
            print(f"  near-uniform branch survives : {n_exist}/{trials}"
                  + (f" (folded {n_fold}, mean fold at s = {r['mean_s_fold']:.3f})"
                     if n_fold else ""))
            if n_exist:
                print(f"  linear criterion on those    : {n_turing_lin}/{n_exist} "
                      f"({r['turing_rate_linear']:.1f}%)")
                print(f"  leading growth rate          : "
                      f"{np.mean(growth_lead):+.5f} +/- {np.std(growth_lead):.5f}")
                print(f"  mean dispersion envelope     : {np.round(env_mean, 5)}")
            print(flush=True)

    return results, {"env0": env0, "m_star": m_star, "keff": keff,
                     "classification": cls, "lam0": lam0}


if __name__ == "__main__":
    CONFIG_TO_TEST = 21
    CONFIG_LABEL = "high"
    N_cells = 10
    n_trials = 200

    df = pd.read_csv("../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv")
    df = df[df["classification"] == "Type-I"]
    row = df[(df["config_id"] == CONFIG_TO_TEST) & (df["param_rank"] == 1)].iloc[0]

    baseline_params = np.array([
        row["alpha_u"], row["beta_u"], row["K_uu"], row["K_vu"], row["delta_u"],
        row["alpha_v"], row["beta_v"], row["K_uv"], row["K_wv"], row["delta_v"],
        row["alpha_w"], row["beta_w"], row["K_ww"], row["K_uw"], row["K_vw"], row["delta_w"]])
    steady_state_expected = np.array([row["u_star"], row["v_star"], row["w_star"]])
    hopping = {"h_u": row["dU"], "h_v": row["dV"], "h_w": row["dW"]}

    results, baseline_info = run(baseline_params, steady_state_expected, hopping,
                                 N_cells=N_cells, n_trials=n_trials)

    out = {"results": results, "baseline_info": baseline_info,
           "baseline_params": baseline_params, "hopping": hopping,
           "n_trials": n_trials, "N_cells": N_cells,
           "config_id": CONFIG_TO_TEST, "config_name": row["config_name"]}
    fname = f"3954_ring_{CONFIG_LABEL}_config{CONFIG_TO_TEST}_N{N_cells}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    print("saved", fname)