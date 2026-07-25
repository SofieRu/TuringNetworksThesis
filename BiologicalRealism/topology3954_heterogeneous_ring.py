#!/usr/bin/env python3
"""Robustness of topology 3954 on a heterogeneous periodic ring.

The ring contains N cells and three species per cell.  Every cell has the same
reaction topology but receives its own 16-parameter multiplier.  Diffusion is
nearest-neighbour and periodic.

Important methodological point
------------------------------
Fourier modes are independent only for a homogeneous ring.  In a heterogeneous
ring there is no exact dispersion relation lambda(k), because parameter disorder
couples the Fourier modes.  This script therefore:

1. uses the exact homogeneous dispersion relation only for the CV=0 reference;
2. classifies heterogeneous trials by continuing the equilibrium from zero
   coupling to full coupling and testing whether the spectral abscissa changes
   from negative to positive;
3. uses Fourier power only to describe (not classify) heterogeneous eigenvectors;
4. measures nonlinear growth relative to the trial-specific heterogeneous
   equilibrium, so a forced spatial profile is not called a Turing pattern.

By default, ``noise_level`` is a lognormal coefficient of variation (CV).
Thus 0.4 means CV=40%, not a hard +/-40% bound.  Select
``--noise-model uniform_fraction`` for bounded +/- noise_level multipliers.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


N_HILL = 2
N_PARAMETERS = 16
DEFAULT_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


# ---------------------------------------------------------------------------
# Reaction kinetics
# ---------------------------------------------------------------------------

def hill_activation(x: np.ndarray | float, K: float):
    return x**N_HILL / (K**N_HILL + x**N_HILL)


def hill_inhibition(x: np.ndarray | float, K: float):
    return K**N_HILL / (K**N_HILL + x**N_HILL)


def dH_act(x: float, K: float):
    return (
        N_HILL
        * K**N_HILL
        * x ** (N_HILL - 1)
        / (K**N_HILL + x**N_HILL) ** 2
    )


def dH_inh(x: float, K: float):
    return (
        -N_HILL
        * K**N_HILL
        * x ** (N_HILL - 1)
        / (K**N_HILL + x**N_HILL) ** 2
    )


def ode_system(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    du = (
        alpha_u
        + beta_u * hill_activation(u, K_uu) * hill_inhibition(v, K_vu)
        - delta_u * u
    )
    dv = (
        alpha_v
        + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv)
        - delta_v * v
    )
    dw = (
        alpha_w
        + beta_w
        * hill_activation(w, K_ww)
        * hill_inhibition(u, K_uw)
        * hill_inhibition(v, K_vw)
        - delta_w * w
    )
    return np.array([du, dv, dw], dtype=float)


def compute_jacobian(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    u, v, w = state
    _, beta_u, K_uu, K_vu, delta_u = params[0:5]
    _, beta_v, K_uv, K_wv, delta_v = params[5:10]
    _, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    J = np.zeros((3, 3), dtype=float)
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = (
        beta_w
        * hill_activation(w, K_ww)
        * dH_inh(u, K_uw)
        * hill_inhibition(v, K_vw)
    )
    J[2, 1] = (
        beta_w
        * hill_activation(w, K_ww)
        * hill_inhibition(u, K_uw)
        * dH_inh(v, K_vw)
    )
    J[2, 2] = (
        beta_w
        * dH_act(w, K_ww)
        * hill_inhibition(u, K_uw)
        * hill_inhibition(v, K_vw)
        - delta_w
    )
    return J


# ---------------------------------------------------------------------------
# Ring construction
# ---------------------------------------------------------------------------

def build_diffusion_operator(n_cells: int, hopping: dict[str, float]) -> np.ndarray:
    """Return L (x) D for a periodic nearest-neighbour ring with spacing a=1."""
    if n_cells < 3:
        raise ValueError("n_cells must be at least 3 for the intended ring.")
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]], dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("Hopping coefficients must be finite and non-negative.")

    Ldiff = np.zeros((3 * n_cells, 3 * n_cells), dtype=float)
    for i in range(n_cells):
        left = (i - 1) % n_cells
        right = (i + 1) % n_cells
        for species in range(3):
            Ldiff[3 * i + species, 3 * i + species] -= 2.0 * h[species]
            Ldiff[3 * i + species, 3 * left + species] += h[species]
            Ldiff[3 * i + species, 3 * right + species] += h[species]
    return Ldiff


def effective_wavenumbers(n_cells: int, lattice_spacing: float = 1.0) -> np.ndarray:
    modes = np.arange(n_cells // 2 + 1)
    return 2.0 * np.sin(np.pi * modes / n_cells) / lattice_spacing


def ring_rhs(
    x: np.ndarray,
    params_list: list[np.ndarray],
    Ldiff: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    X = x.reshape(n_cells, 3)
    reaction = np.concatenate(
        [ode_system(X[i], params_list[i]) for i in range(n_cells)]
    )
    return reaction + Ldiff @ x


def ring_jacobian(
    x: np.ndarray,
    params_list: list[np.ndarray],
    Ldiff: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    J = Ldiff.copy()
    X = x.reshape(n_cells, 3)
    for i in range(n_cells):
        block = slice(3 * i, 3 * i + 3)
        J[block, block] += compute_jacobian(X[i], params_list[i])
    return J


def scaled_residual(f: np.ndarray, x: np.ndarray) -> float:
    """Dimensionless-ish residual used consistently by Newton and diagnostics."""
    return float(np.linalg.norm(f, ord=np.inf) / (1.0 + np.linalg.norm(x, ord=np.inf)))


@dataclass
class NewtonResult:
    x: np.ndarray
    residual: float
    converged: bool
    reason: str
    iterations: int


def newton_log(
    x0: np.ndarray,
    params_list: list[np.ndarray],
    Ldiff: np.ndarray,
    n_cells: int,
    tol: float = 1e-9,
    max_iterations: int = 80,
    step_cap: float = 0.5,
) -> NewtonResult:
    """Damped Newton solve in log-concentration coordinates."""
    if np.any(x0 <= 0) or not np.all(np.isfinite(x0)):
        return NewtonResult(np.asarray(x0), np.inf, False, "invalid_initial_state", 0)

    y = np.log(np.asarray(x0, dtype=float))
    x = np.exp(y)
    f = ring_rhs(x, params_list, Ldiff, n_cells)
    residual = scaled_residual(f, x)

    for iteration in range(1, max_iterations + 1):
        if residual < tol:
            return NewtonResult(x, residual, True, "converged", iteration - 1)

        Jx = ring_jacobian(x, params_list, Ldiff, n_cells)
        Jy = Jx * x[None, :]  # dF/dy = dF/dx @ diag(x)
        try:
            dy = np.linalg.solve(Jy, -f)
        except np.linalg.LinAlgError:
            return NewtonResult(x, residual, False, "singular_jacobian", iteration)

        if not np.all(np.isfinite(dy)):
            return NewtonResult(x, residual, False, "nonfinite_newton_step", iteration)

        step = min(1.0, step_cap / max(np.max(np.abs(dy)), 1e-300))
        accepted = False
        for _ in range(30):
            y_trial = y + step * dy
            x_trial = np.exp(y_trial)
            f_trial = ring_rhs(x_trial, params_list, Ldiff, n_cells)
            residual_trial = scaled_residual(f_trial, x_trial)
            if np.isfinite(residual_trial) and residual_trial < residual:
                y, x, f, residual = y_trial, x_trial, f_trial, residual_trial
                accepted = True
                break
            step *= 0.5

        if not accepted:
            return NewtonResult(x, residual, False, "line_search_failed", iteration)

    return NewtonResult(x, residual, residual < tol, "iteration_limit", max_iterations)


@dataclass
class ContinuationResult:
    x: np.ndarray | None
    reached: float
    converged: bool
    reason: str
    path_parameter: np.ndarray
    residual_path: np.ndarray


def continue_branch(
    x_start: np.ndarray,
    system_at: Callable[[float], tuple[list[np.ndarray], np.ndarray]],
    n_cells: int,
    initial_step: float = 0.1,
    minimum_step: float = 1.0 / 2048.0,
) -> ContinuationResult:
    """Adaptive natural-parameter continuation from s=0 to s=1.

    Failure is reported as a continuation failure; it is not labelled a fold.
    """
    params0, L0 = system_at(0.0)
    start = newton_log(x_start, params0, L0, n_cells)
    if not start.converged:
        return ContinuationResult(
            None,
            0.0,
            False,
            f"start_{start.reason}",
            np.array([0.0]),
            np.array([start.residual]),
        )

    x = start.x
    s = 0.0
    step = initial_step
    s_path = [0.0]
    residual_path = [start.residual]

    while s < 1.0 - 1e-14:
        target = min(1.0, s + step)
        params, Ldiff = system_at(target)
        attempt = newton_log(x, params, Ldiff, n_cells)
        if attempt.converged:
            x = attempt.x
            s = target
            s_path.append(s)
            residual_path.append(attempt.residual)
            step = min(initial_step, 1.5 * step)
            continue

        step *= 0.5
        if step < minimum_step:
            return ContinuationResult(
                None,
                s,
                False,
                attempt.reason,
                np.asarray(s_path),
                np.asarray(residual_path),
            )

    return ContinuationResult(
        x,
        1.0,
        True,
        "converged",
        np.asarray(s_path),
        np.asarray(residual_path),
    )


def isolated_states_on_noise_branch(
    noise: np.ndarray,
    baseline_params: np.ndarray,
    baseline_state: np.ndarray,
) -> tuple[np.ndarray | None, list[dict]]:
    """Follow each isolated cell from baseline parameters to its noisy parameters."""
    states = []
    diagnostics = []
    zero_L = np.zeros((3, 3), dtype=float)

    for cell in range(noise.shape[0]):
        factor = noise[cell]

        def system_at(s: float):
            params = baseline_params * factor**s
            return [params], zero_L

        result = continue_branch(
            baseline_state.copy(),
            system_at,
            n_cells=1,
            initial_step=0.1,
        )
        diagnostics.append(
            {
                "cell": cell,
                "converged": result.converged,
                "reached": result.reached,
                "reason": result.reason,
            }
        )
        if not result.converged:
            return None, diagnostics
        states.append(result.x)

    return np.concatenate(states), diagnostics


# ---------------------------------------------------------------------------
# Linear stability and Fourier descriptions
# ---------------------------------------------------------------------------

def spectral_abscissa(J: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(J))))


def homogeneous_dispersion(
    state: np.ndarray,
    params: np.ndarray,
    hopping: dict[str, float],
    n_cells: int,
) -> dict:
    """Exact discrete dispersion relation for the homogeneous reference ring."""
    D = np.diag([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    keff = effective_wavenumbers(n_cells)
    J_reaction = compute_jacobian(state, params)
    eigvals = np.array([np.linalg.eigvals(J_reaction - k**2 * D) for k in keff])
    alpha = np.max(np.real(eigvals), axis=1)
    return {"keff": keff, "eigvals": eigvals, "alpha": alpha}


def folded_fourier_power(V: np.ndarray, n_cells: int) -> np.ndarray:
    """Fourier power of every eigenvector, folded over +/- ring modes."""
    vector_field = V.reshape(n_cells, 3, -1)
    power = np.abs(np.fft.fft(vector_field, axis=0)) ** 2
    power = power.sum(axis=1)
    n_unique = n_cells // 2 + 1
    folded = power[:n_unique].copy()
    for mode in range(1, (n_cells + 1) // 2):
        folded[mode] += power[n_cells - mode]
    denominators = folded.sum(axis=0, keepdims=True)
    return np.divide(folded, denominators, out=np.zeros_like(folded), where=denominators > 0)


def heterogeneous_spectrum(J: np.ndarray, n_cells: int) -> dict:
    """Exact eigenvalues plus descriptive spatial content of their eigenvectors."""
    eigvals, V = np.linalg.eig(J)
    power = folded_fourier_power(V, n_cells)  # (n_modes, n_eigenvalues)
    dominant_mode = np.argmax(power, axis=0)
    purity = power[dominant_mode, np.arange(power.shape[1])]

    cell_power = np.abs(V.reshape(n_cells, 3, -1)) ** 2
    cell_power = cell_power.sum(axis=1)
    cell_power /= cell_power.sum(axis=0, keepdims=True)
    inverse_participation_ratio = np.sum(cell_power**2, axis=0)

    lead = int(np.argmax(np.real(eigvals)))
    return {
        "eigvals": eigvals,
        "fourier_power": power.T,  # (n_eigenvalues, n_modes)
        "dominant_mode": dominant_mode,
        "purity": purity,
        "inverse_participation_ratio": inverse_participation_ratio,
        "leading_index": lead,
        "leading_eigenvalue": eigvals[lead],
        "leading_fourier_power": power[:, lead],
        "leading_dominant_mode": int(dominant_mode[lead]),
        "leading_purity": float(purity[lead]),
        "leading_ipr": float(inverse_participation_ratio[lead]),
        "alpha": float(np.real(eigvals[lead])),
    }


def coupling_continuation(
    params_list: list[np.ndarray],
    x_uncoupled: np.ndarray,
    Ldiff: np.ndarray,
    n_cells: int,
) -> tuple[ContinuationResult, dict]:
    """Continue the equilibrium from zero to full intercellular coupling."""

    def system_at(gamma: float):
        return params_list, gamma * Ldiff

    result = continue_branch(
        x_uncoupled,
        system_at,
        n_cells=n_cells,
        initial_step=0.05,
    )
    if not result.converged:
        return result, {
            "gamma": result.path_parameter,
            "alpha": np.full(result.path_parameter.size, np.nan),
            "crossing_gamma": np.nan,
        }

    # Re-solve along the accepted gamma path to retain stability information.
    x = x_uncoupled.copy()
    gammas = result.path_parameter
    alphas = []
    for gamma in gammas:
        solved = newton_log(x, params_list, gamma * Ldiff, n_cells)
        if not solved.converged:
            alphas.append(np.nan)
            continue
        x = solved.x
        alphas.append(spectral_abscissa(ring_jacobian(x, params_list, gamma * Ldiff, n_cells)))
    alphas = np.asarray(alphas)

    crossing = np.nan
    for i in range(1, len(gammas)):
        a0, a1 = alphas[i - 1], alphas[i]
        if np.isfinite(a0) and np.isfinite(a1) and a0 <= 0 < a1:
            fraction = -a0 / (a1 - a0) if a1 != a0 else 0.0
            crossing = float(gammas[i - 1] + fraction * (gammas[i] - gammas[i - 1]))
            break

    return result, {"gamma": gammas, "alpha": alphas, "crossing_gamma": crossing}


# ---------------------------------------------------------------------------
# Nonlinear outcome relative to the heterogeneous equilibrium
# ---------------------------------------------------------------------------

def simulate_ring(
    params_list: list[np.ndarray],
    Ldiff: np.ndarray,
    n_cells: int,
    x_init: np.ndarray,
    t_end: float = 5000.0,
    stages: tuple[float, ...] = (50.0, 200.0, 1000.0, 5000.0),
    rtol: float = 1e-7,
    atol: float = 1e-10,
    steady_rhs_tol: float = 1e-8,
    minimum_settle_time: float = 1000.0,
) -> dict:
    f = lambda _t, x: ring_rhs(x, params_list, Ldiff, n_cells)
    jf = lambda _t, x: ring_jacobian(x, params_list, Ldiff, n_cells)

    x = np.asarray(x_init, dtype=float)
    t0 = 0.0
    endpoints = sorted(set([t for t in stages if 0 < t <= t_end] + [t_end]))
    rhs_residual = scaled_residual(f(t0, x), x)

    for t1 in endpoints:
        if t1 <= t0:
            continue
        solution = solve_ivp(
            f,
            (t0, t1),
            x,
            method="LSODA",
            jac=jf,
            rtol=rtol,
            atol=atol,
            t_eval=[t1],
        )
        if not solution.success:
            return {
                "success": False,
                "settled": False,
                "reason": solution.message,
                "x": None,
                "t": t0,
                "rhs_residual": np.nan,
            }
        x = solution.y[:, -1]
        t0 = t1
        if not np.all(np.isfinite(x)) or np.any(x <= 0):
            return {
                "success": False,
                "settled": False,
                "reason": "nonpositive_or_nonfinite_state",
                "x": None,
                "t": t0,
                "rhs_residual": np.nan,
            }
        rhs_residual = scaled_residual(f(t0, x), x)
        # Do not stop very near an unstable equilibrium merely because the
        # initial perturbation and hence its instantaneous derivative are small.
        if t0 >= minimum_settle_time and rhs_residual < steady_rhs_tol:
            return {
                "success": True,
                "settled": True,
                "reason": "steady",
                "x": x,
                "t": t0,
                "rhs_residual": rhs_residual,
            }

    return {
        "success": True,
        "settled": rhs_residual < steady_rhs_tol,
        "reason": "t_end",
        "x": x,
        "t": t0,
        "rhs_residual": rhs_residual,
    }


def nonlinear_pattern_metrics(
    x_final: np.ndarray,
    x_equilibrium: np.ndarray,
    n_cells: int,
    species: int = 0,
) -> dict:
    """Measure extra structure grown beyond the forced heterogeneous equilibrium."""
    final_profile = x_final.reshape(n_cells, 3)[:, species]
    base_profile = x_equilibrium.reshape(n_cells, 3)[:, species]
    delta = final_profile - base_profile
    scale = max(abs(np.mean(base_profile)), 1e-12)

    delta_amplitude = float((np.max(delta) - np.min(delta)) / scale)
    total_amplitude = float(
        (np.max(final_profile) - np.min(final_profile))
        / max(abs(np.mean(final_profile)), 1e-12)
    )
    forced_amplitude = float(
        (np.max(base_profile) - np.min(base_profile))
        / max(abs(np.mean(base_profile)), 1e-12)
    )

    centred_delta = delta - np.mean(delta)
    raw_power = np.abs(np.fft.fft(centred_delta)) ** 2
    n_unique = n_cells // 2 + 1
    folded = raw_power[:n_unique].copy()
    for mode in range(1, (n_cells + 1) // 2):
        folded[mode] += raw_power[n_cells - mode]
    spectrum = folded / folded.sum() if folded.sum() > 0 else np.zeros_like(folded)
    dominant_mode = int(np.argmax(spectrum[1:]) + 1) if spectrum[1:].sum() > 0 else 0

    return {
        "delta_amplitude": delta_amplitude,
        "total_amplitude": total_amplitude,
        "forced_equilibrium_amplitude": forced_amplitude,
        "dominant_mode": dominant_mode,
        "delta_fourier_power": spectrum,
    }


# ---------------------------------------------------------------------------
# Sampling and experiment driver
# ---------------------------------------------------------------------------

def noise_multipliers(
    level: float,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    model: str,
) -> np.ndarray:
    if level < 0:
        raise ValueError("Noise levels must be non-negative.")
    if model == "lognormal_cv":
        sigma = np.sqrt(np.log1p(level**2))
        mu = -0.5 * sigma**2
        return np.exp(mu + sigma * normal_draws)
    if model == "uniform_fraction":
        factors = 1.0 + level * uniform_draws
        if np.any(factors <= 0):
            raise ValueError("uniform_fraction produced non-positive parameters.")
        return factors
    raise ValueError(f"Unknown noise model: {model}")


def validate_baseline(
    baseline_params: np.ndarray,
    baseline_state: np.ndarray,
    hopping: dict[str, float],
    n_cells: int,
    stability_tol: float,
) -> dict:
    if baseline_params.shape != (N_PARAMETERS,) or np.any(baseline_params <= 0):
        raise ValueError("baseline_params must contain 16 positive values.")
    if baseline_state.shape != (3,) or np.any(baseline_state <= 0):
        raise ValueError("baseline_state must contain three positive values.")

    residual = scaled_residual(ode_system(baseline_state, baseline_params), baseline_state)
    if residual > 1e-7:
        raise ValueError(
            f"Tabulated baseline state is not a sufficiently accurate equilibrium "
            f"(scaled residual={residual:.3e})."
        )

    dispersion = homogeneous_dispersion(
        baseline_state, baseline_params, hopping, n_cells
    )
    alpha = dispersion["alpha"]
    baseline_turing = bool(alpha[0] < -stability_tol and np.max(alpha[1:]) > stability_tol)
    if not baseline_turing:
        raise ValueError(
            "The selected homogeneous configuration is not Turing unstable on this "
            f"{n_cells}-cell discrete ring. alpha(k_m)={alpha}"
        )
    dominant_mode = int(np.argmax(alpha[1:]) + 1)
    return {
        "scaled_residual": residual,
        "dispersion": dispersion,
        "baseline_turing": baseline_turing,
        "dominant_mode": dominant_mode,
    }


def run_experiment(
    baseline_params: np.ndarray,
    baseline_state: np.ndarray,
    hopping: dict[str, float],
    n_cells: int = 10,
    n_trials: int = 200,
    noise_levels: tuple[float, ...] = DEFAULT_LEVELS,
    noise_model: str = "lognormal_cv",
    seed: int = 42,
    stability_tol: float = 1e-8,
    do_nonlinear: bool = True,
    n_initial_conditions: int = 3,
    perturbation_size: float = 1e-3,
    nonlinear_amplitude_tol: float = 0.05,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Run paired Monte Carlo trials over all requested noise levels."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if n_initial_conditions <= 0:
        raise ValueError("n_initial_conditions must be positive.")

    Ldiff = build_diffusion_operator(n_cells, hopping)
    baseline_info = validate_baseline(
        baseline_params, baseline_state, hopping, n_cells, stability_tol
    )
    baseline_mode = baseline_info["dominant_mode"]

    # Independent reproducible streams.  The same parameter draws and initial
    # perturbation directions are reused at every noise level (paired design).
    seed_sequence = np.random.SeedSequence(seed)
    parameter_seed, initial_seed = seed_sequence.spawn(2)
    parameter_rng = np.random.default_rng(parameter_seed)
    initial_rng = np.random.default_rng(initial_seed)
    common_normal = parameter_rng.standard_normal(
        (n_trials, n_cells, N_PARAMETERS)
    )
    common_uniform = parameter_rng.uniform(
        -1.0, 1.0, (n_trials, n_cells, N_PARAMETERS)
    )
    common_initial_directions = initial_rng.standard_normal(
        (n_trials, n_initial_conditions, 3 * n_cells)
    )

    results = []
    for level in noise_levels:
        trial_records = []
        factors_all = noise_multipliers(
            level, common_normal, common_uniform, noise_model
        )

        for trial in range(n_trials):
            if verbose and trial and trial % 25 == 0:
                print(
                    f"noise={level:.2f}: trial {trial}/{n_trials}",
                    flush=True,
                )

            factors = factors_all[trial]
            params_list = [baseline_params * factors[cell] for cell in range(n_cells)]
            record = {
                "trial": trial,
                "noise_level": level,
                "noise_factors": factors,
                "params": np.asarray(params_list),
                "status": "started",
            }

            x_uncoupled, isolated_diagnostics = isolated_states_on_noise_branch(
                factors, baseline_params, baseline_state
            )
            record["isolated_diagnostics"] = isolated_diagnostics
            if x_uncoupled is None:
                record["status"] = "isolated_continuation_failed"
                trial_records.append(record)
                continue

            J_uncoupled = ring_jacobian(
                x_uncoupled,
                params_list,
                np.zeros_like(Ldiff),
                n_cells,
            )
            alpha_uncoupled = spectral_abscissa(J_uncoupled)
            record["x_uncoupled"] = x_uncoupled
            record["alpha_uncoupled"] = alpha_uncoupled

            coupling_result, coupling_stability = coupling_continuation(
                params_list, x_uncoupled, Ldiff, n_cells
            )
            record["coupling_continuation"] = {
                "converged": coupling_result.converged,
                "reached": coupling_result.reached,
                "reason": coupling_result.reason,
                **coupling_stability,
            }
            if not coupling_result.converged:
                record["status"] = "coupling_continuation_failed"
                trial_records.append(record)
                continue

            x_coupled = coupling_result.x
            J_coupled = ring_jacobian(x_coupled, params_list, Ldiff, n_cells)
            spectrum = heterogeneous_spectrum(J_coupled, n_cells)
            alpha_coupled = spectrum["alpha"]
            no_coupling_stable = alpha_uncoupled < -stability_tol
            coupled_unstable = alpha_coupled > stability_tol
            diffusion_driven = bool(no_coupling_stable and coupled_unstable)

            record.update(
                {
                    "status": "classified",
                    "x_coupled": x_coupled,
                    "coupled_scaled_residual": scaled_residual(
                        ring_rhs(x_coupled, params_list, Ldiff, n_cells),
                        x_coupled,
                    ),
                    "spectrum": spectrum,
                    "alpha_coupled": alpha_coupled,
                    "no_coupling_stable": bool(no_coupling_stable),
                    "coupled_unstable": bool(coupled_unstable),
                    "diffusion_driven_instability": diffusion_driven,
                }
            )

            if do_nonlinear:
                nonlinear_runs = []
                for init_index in range(n_initial_conditions):
                    direction = common_initial_directions[trial, init_index]
                    x_init = x_coupled * np.exp(perturbation_size * direction)
                    simulation = simulate_ring(
                        params_list, Ldiff, n_cells, x_init
                    )
                    nonlinear_record = {
                        key: value
                        for key, value in simulation.items()
                        if key != "x"
                    }
                    if simulation["success"]:
                        nonlinear_record["x_final"] = simulation["x"]
                        nonlinear_record["metrics"] = nonlinear_pattern_metrics(
                            simulation["x"], x_coupled, n_cells
                        )
                        nonlinear_record["pattern_grew"] = bool(
                            simulation["settled"]
                            and nonlinear_record["metrics"]["delta_amplitude"]
                            > nonlinear_amplitude_tol
                            and nonlinear_record["metrics"]["dominant_mode"] > 0
                        )
                        nonlinear_record["correct_baseline_mode"] = bool(
                            nonlinear_record["pattern_grew"]
                            and nonlinear_record["metrics"]["dominant_mode"]
                            == baseline_mode
                        )
                    else:
                        nonlinear_record["pattern_grew"] = False
                        nonlinear_record["correct_baseline_mode"] = False
                    nonlinear_runs.append(nonlinear_record)

                successful = [r for r in nonlinear_runs if r["success"]]
                required = n_initial_conditions // 2 + 1
                record["nonlinear_runs"] = nonlinear_runs
                record["nonlinear_valid"] = len(successful) >= required
                record["nonlinear_pattern"] = bool(
                    record["nonlinear_valid"]
                    and sum(r["pattern_grew"] for r in successful)
                    >= (len(successful) // 2 + 1)
                )
                record["nonlinear_diffusion_driven_pattern"] = bool(
                    diffusion_driven and record["nonlinear_pattern"]
                )
                record["nonlinear_correct_mode"] = bool(
                    diffusion_driven
                    and record["nonlinear_valid"]
                    and sum(r["correct_baseline_mode"] for r in successful)
                    >= (len(successful) // 2 + 1)
                )

            trial_records.append(record)

        classified = [r for r in trial_records if r["status"] == "classified"]
        n_isolated_failed = sum(
            r["status"] == "isolated_continuation_failed" for r in trial_records
        )
        n_coupling_failed = sum(
            r["status"] == "coupling_continuation_failed" for r in trial_records
        )
        n_uncoupled_stable = sum(r["no_coupling_stable"] for r in classified)
        n_coupled_unstable = sum(r["coupled_unstable"] for r in classified)
        n_diffusion_driven = sum(
            r["diffusion_driven_instability"] for r in classified
        )
        alpha_values = np.array([r["alpha_coupled"] for r in classified])

        summary = {
            "noise_level": level,
            "noise_model": noise_model,
            "n_trials": n_trials,
            "n_classified": len(classified),
            "n_isolated_continuation_failed": int(n_isolated_failed),
            "n_coupling_continuation_failed": int(n_coupling_failed),
            "n_uncoupled_stable": int(n_uncoupled_stable),
            "n_coupled_unstable": int(n_coupled_unstable),
            "n_diffusion_driven": int(n_diffusion_driven),
            "diffusion_driven_percent_of_all": 100.0 * n_diffusion_driven / n_trials,
            "diffusion_driven_percent_of_classified": (
                100.0 * n_diffusion_driven / len(classified)
                if classified
                else np.nan
            ),
            "alpha_coupled_mean": (
                float(np.mean(alpha_values)) if alpha_values.size else np.nan
            ),
            "alpha_coupled_median": (
                float(np.median(alpha_values)) if alpha_values.size else np.nan
            ),
            "alpha_coupled_q025": (
                float(np.quantile(alpha_values, 0.025))
                if alpha_values.size
                else np.nan
            ),
            "alpha_coupled_q975": (
                float(np.quantile(alpha_values, 0.975))
                if alpha_values.size
                else np.nan
            ),
        }

        if do_nonlinear:
            nonlinear_valid = [r for r in classified if r.get("nonlinear_valid", False)]
            n_nonlinear_pattern = sum(
                r.get("nonlinear_diffusion_driven_pattern", False)
                for r in classified
            )
            n_correct_mode = sum(
                r.get("nonlinear_correct_mode", False) for r in classified
            )
            summary.update(
                {
                    "n_nonlinear_valid": len(nonlinear_valid),
                    "n_nonlinear_diffusion_driven_pattern": int(n_nonlinear_pattern),
                    "nonlinear_pattern_percent_of_all": (
                        100.0 * n_nonlinear_pattern / n_trials
                    ),
                    "n_nonlinear_correct_mode": int(n_correct_mode),
                }
            )

        results.append({"summary": summary, "trials": trial_records})
        if verbose:
            print(
                f"noise={level:.2f}: diffusion-driven "
                f"{n_diffusion_driven}/{n_trials}; "
                f"classified={len(classified)}, "
                f"failures={n_isolated_failed + n_coupling_failed}; "
                f"median max Re(lambda)={summary['alpha_coupled_median']:+.5g}",
                flush=True,
            )

    metadata = {
        "seed": seed,
        "n_cells": n_cells,
        "n_trials": n_trials,
        "noise_levels": tuple(noise_levels),
        "noise_model": noise_model,
        "stability_tolerance": stability_tol,
        "baseline": baseline_info,
        "hopping": hopping,
        "baseline_params": baseline_params,
        "baseline_state": baseline_state,
        "method_note": (
            "Heterogeneous trials are classified by stability without versus with "
            "coupling. Fourier power is descriptive because heterogeneous Fourier "
            "modes are coupled."
        ),
    }
    return results, metadata


# ---------------------------------------------------------------------------
# Input/output
# ---------------------------------------------------------------------------

PARAMETER_COLUMNS = (
    "alpha_u",
    "beta_u",
    "K_uu",
    "K_vu",
    "delta_u",
    "alpha_v",
    "beta_v",
    "K_uv",
    "K_wv",
    "delta_v",
    "alpha_w",
    "beta_w",
    "K_ww",
    "K_uw",
    "K_vw",
    "delta_w",
)


def load_configuration(
    csv_path: Path,
    config_id: int,
    param_rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    matches = [
        row
        for row in rows
        if row.get("classification") == "Type-I"
        and int(float(row["config_id"])) == config_id
        and int(float(row["param_rank"])) == param_rank
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Type-I row for config_id={config_id}, "
            f"param_rank={param_rank}; found {len(matches)}."
        )

    row = matches[0]
    params = np.array([float(row[name]) for name in PARAMETER_COLUMNS])
    state = np.array([float(row["u_star"]), float(row["v_star"]), float(row["w_star"])])
    hopping = {
        "h_u": float(row["dU"]),
        "h_v": float(row["dV"]),
        "h_w": float(row["dW"]),
    }
    return params, state, hopping, row


def write_summary_csv(results: list[dict], path: Path) -> None:
    rows = [item["summary"] for item in results]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv"),
        help="Topology 3954 parameter-results CSV.",
    )
    parser.add_argument("--config-id", type=int, default=21)
    parser.add_argument("--param-rank", type=int, default=1)
    parser.add_argument("--config-label", default="high")
    parser.add_argument("--n-cells", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-model",
        choices=("lognormal_cv", "uniform_fraction"),
        default="lognormal_cv",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEVELS),
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Skip nonlinear integrations; linear stability remains fully analysed.",
    )
    parser.add_argument("--n-initial-conditions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params, state, hopping, source_row = load_configuration(
        args.csv, args.config_id, args.param_rank
    )
    results, metadata = run_experiment(
        params,
        state,
        hopping,
        n_cells=args.n_cells,
        n_trials=args.n_trials,
        noise_levels=tuple(args.noise_levels),
        noise_model=args.noise_model,
        seed=args.seed,
        do_nonlinear=not args.linear_only,
        n_initial_conditions=args.n_initial_conditions,
    )
    metadata.update(
        {
            "config_id": args.config_id,
            "param_rank": args.param_rank,
            "config_name": source_row.get("config_name", ""),
            "source_csv": str(args.csv.resolve()),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"3954_ring_{args.config_label}_config{args.config_id}"
        f"_N{args.n_cells}_{args.noise_model}"
    )
    pickle_path = args.output_dir / f"{stem}.pkl"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    with pickle_path.open("wb") as handle:
        pickle.dump({"results": results, "metadata": metadata}, handle)
    write_summary_csv(results, summary_path)
    print(f"Saved full trial data: {pickle_path}")
    print(f"Saved summary table:   {summary_path}")


if __name__ == "__main__":
    main()
