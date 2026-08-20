#!/usr/bin/env python3
"""Final heterogeneous-ring robustness experiment for topology 3954.

MODEL
-----
Ten cells (configurable) are arranged in a periodic nearest-neighbour ring.
Every cell contains topology 3954 and receives its own 16 kinetic parameters.
The three hopping/diffusion coefficients are kept fixed unless the model is
explicitly changed.

NOISE
-----
The default noise is identical in meaning to the original draft: each kinetic
parameter is multiplied by an independent, mean-one lognormal random variable
with the requested coefficient of variation (CV).  CV=0.40 is therefore a 40%
coefficient of variation, not a hard +/-40% bound.

STABILITY DEFINITION
--------------------
For a heterogeneous ring there is no exact dispersion relation lambda(k):
heterogeneity couples the Fourier modes.  This script therefore calls a trial
a *pure stationary diffusion-driven (Turing) instability* only when:

1. all ten noisy cells are stable when uncoupled;
2. the coupled heterogeneous equilibrium has a positive stationary eigenvalue;
3. it has no positive oscillatory eigenvalue.

Mixed stationary/oscillatory and purely oscillatory instabilities are reported
separately.  The full eigenvalue spectrum and the Fourier power of every
eigenvector are saved for descriptive mode-resolved plots.

EQUILIBRIUM BRANCHES
--------------------
Multistability can make the result depend on how the equilibrium is prepared.
Two paths are therefore calculated:

A. Primary/noise path: begin with the homogeneous coupled ring and gradually
   introduce the cell-specific parameter noise.
B. Coupling path: find the ten noisy isolated equilibria and gradually turn on
   cell-cell coupling.

If the two paths reach different final equilibria, the trial is explicitly
marked path-dependent.  It is not included in the conservative
"unambiguous Turing" count, although both endpoint results are retained.

OUTPUT
------
The program writes:

* a pickle containing all trial-level parameters, equilibria, eigenvalues,
  eigenvector Fourier powers, continuation diagnostics and optional nonlinear
  results;
* a flat CSV summary with one row per CV.

The homogeneous CV=0 reference includes the exact discrete dispersion relation
at k_m = 2 sin(pi*m/N)/a.  A ten-cell ring has six unique wavenumbers.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


N_HILL = 2
N_PARAMETERS = 16
DEFAULT_CV_VALUES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


# ===========================================================================
# Reaction kinetics
# ===========================================================================

def hill_activation(x, K):
    return x**N_HILL / (K**N_HILL + x**N_HILL)


def hill_inhibition(x, K):
    return K**N_HILL / (K**N_HILL + x**N_HILL)


def dH_act(x, K):
    return (
        N_HILL
        * K**N_HILL
        * x ** (N_HILL - 1)
        / (K**N_HILL + x**N_HILL) ** 2
    )


def dH_inh(x, K):
    return (
        -N_HILL
        * K**N_HILL
        * x ** (N_HILL - 1)
        / (K**N_HILL + x**N_HILL) ** 2
    )


def reaction_rhs(state, params):
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


def reaction_jacobian(state, params):
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


# ===========================================================================
# Periodic ring
# ===========================================================================

def build_diffusion_operator(n_cells, hopping, lattice_spacing=1.0):
    """Return the periodic discrete diffusion operator L (x) D/a^2."""
    if n_cells < 3:
        raise ValueError("n_cells must be at least 3.")
    if lattice_spacing <= 0:
        raise ValueError("lattice_spacing must be positive.")

    rates = np.array(
        [hopping["h_u"], hopping["h_v"], hopping["h_w"]],
        dtype=float,
    ) / lattice_spacing**2
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Hopping/diffusion coefficients must be finite and non-negative.")

    Ldiff = np.zeros((3 * n_cells, 3 * n_cells), dtype=float)
    for cell in range(n_cells):
        left = (cell - 1) % n_cells
        right = (cell + 1) % n_cells
        for species in range(3):
            row = 3 * cell + species
            Ldiff[row, row] -= 2.0 * rates[species]
            Ldiff[row, 3 * left + species] += rates[species]
            Ldiff[row, 3 * right + species] += rates[species]
    return Ldiff


def effective_wavenumbers(n_cells, lattice_spacing=1.0):
    modes = np.arange(n_cells // 2 + 1)
    return 2.0 * np.sin(np.pi * modes / n_cells) / lattice_spacing


def ring_rhs(x, params_list, Ldiff, n_cells):
    X = np.asarray(x).reshape(n_cells, 3)
    reactions = np.concatenate(
        [reaction_rhs(X[cell], params_list[cell]) for cell in range(n_cells)]
    )
    return reactions + Ldiff @ x


def ring_jacobian(x, params_list, Ldiff, n_cells):
    X = np.asarray(x).reshape(n_cells, 3)
    J = Ldiff.copy()
    for cell in range(n_cells):
        block = slice(3 * cell, 3 * cell + 3)
        J[block, block] += reaction_jacobian(X[cell], params_list[cell])
    return J


# ===========================================================================
# Positive equilibrium solver and continuation
# ===========================================================================

def scaled_residual(f, x):
    return float(
        np.linalg.norm(f, ord=np.inf)
        / (1.0 + np.linalg.norm(x, ord=np.inf))
    )


def least_squares_log_fallback(
    x0,
    params_list,
    Ldiff,
    n_cells,
    tolerance,
    preceding_reason,
):
    """Independent fallback when damped Newton fails or stagnates."""
    if np.any(np.asarray(x0) <= 0):
        return {
            "converged": False,
            "x": None,
            "residual": np.inf,
            "iterations": 0,
            "reason": f"{preceding_reason};invalid_fallback_state",
        }

    def objective(y):
        x = np.exp(y)
        return ring_rhs(x, params_list, Ldiff, n_cells)

    def jacobian(y):
        x = np.exp(y)
        return (
            ring_jacobian(x, params_list, Ldiff, n_cells)
            * x[None, :]
        )

    try:
        solution = least_squares(
            objective,
            np.log(np.asarray(x0, dtype=float)),
            jac=jacobian,
            method="trf",
            x_scale="jac",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=1000,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        return {
            "converged": False,
            "x": None,
            "residual": np.inf,
            "iterations": 0,
            "reason": f"{preceding_reason};fallback_exception:{type(error).__name__}",
        }

    x = np.exp(solution.x)
    residual = scaled_residual(
        ring_rhs(x, params_list, Ldiff, n_cells),
        x,
    )
    converged = bool(
        np.all(np.isfinite(x))
        and np.all(x > 0)
        and residual < tolerance
    )
    return {
        "converged": converged,
        "x": x if converged else None,
        "residual": residual,
        "iterations": int(solution.nfev),
        "reason": (
            f"least_squares_fallback_after_{preceding_reason}"
            if converged
            else f"{preceding_reason};least_squares_failed"
        ),
    }


def newton_log(
    x0,
    params_list,
    Ldiff,
    n_cells,
    tolerance=1e-9,
    max_iterations=80,
    maximum_log_step=0.5,
):
    """Damped Newton solve in y=log(x), preserving positive concentrations."""
    x0 = np.asarray(x0, dtype=float)
    if np.any(x0 <= 0) or not np.all(np.isfinite(x0)):
        return {
            "converged": False,
            "x": None,
            "residual": np.inf,
            "iterations": 0,
            "reason": "invalid_initial_state",
        }

    y = np.log(x0)
    for iteration in range(max_iterations + 1):
        x = np.exp(y)
        f = ring_rhs(x, params_list, Ldiff, n_cells)
        residual = scaled_residual(f, x)
        if residual < tolerance:
            return {
                "converged": True,
                "x": x,
                "residual": residual,
                "iterations": iteration,
                "reason": "converged",
            }
        if iteration == max_iterations:
            break

        # dF/dy = dF/dx @ diag(x): multiply each column by its x value.
        Jy = ring_jacobian(x, params_list, Ldiff, n_cells) * x[None, :]
        try:
            dy = np.linalg.solve(Jy, -f)
        except np.linalg.LinAlgError:
            return least_squares_log_fallback(
                x,
                params_list,
                Ldiff,
                n_cells,
                tolerance,
                "singular_jacobian",
            )
        if not np.all(np.isfinite(dy)):
            return least_squares_log_fallback(
                x,
                params_list,
                Ldiff,
                n_cells,
                tolerance,
                "nonfinite_step",
            )

        step = min(
            1.0,
            maximum_log_step / max(np.max(np.abs(dy)), 1e-300),
        )
        accepted = False
        for _ in range(30):
            y_trial = y + step * dy
            x_trial = np.exp(y_trial)
            f_trial = ring_rhs(x_trial, params_list, Ldiff, n_cells)
            trial_residual = scaled_residual(f_trial, x_trial)
            if np.isfinite(trial_residual) and trial_residual < residual:
                y = y_trial
                accepted = True
                break
            step *= 0.5

        if not accepted:
            return least_squares_log_fallback(
                x,
                params_list,
                Ldiff,
                n_cells,
                tolerance,
                "line_search_failed",
            )

    return least_squares_log_fallback(
        x,
        params_list,
        Ldiff,
        n_cells,
        tolerance,
        "iteration_limit",
    )


def continue_branch(
    x_start,
    system_at,
    n_cells,
    initial_step=0.05,
    minimum_step=1.0 / 4096.0,
    keep_states=False,
):
    """Adaptive natural-parameter continuation from coordinate 0 to 1."""
    params0, L0 = system_at(0.0)
    start = newton_log(x_start, params0, L0, n_cells)
    if not start["converged"]:
        return {
            "converged": False,
            "x": None,
            "reached": 0.0,
            "reason": f"start_{start['reason']}",
            "coordinate": np.array([0.0]),
            "residual": np.array([start["residual"]]),
            "states": [] if keep_states else None,
        }

    x = start["x"]
    coordinate = 0.0
    step = initial_step
    coordinate_path = [0.0]
    residual_path = [start["residual"]]
    states = [x.copy()] if keep_states else None

    while coordinate < 1.0 - 1e-14:
        target = min(1.0, coordinate + step)
        params, Ldiff = system_at(target)
        attempt = newton_log(x, params, Ldiff, n_cells)
        if attempt["converged"]:
            x = attempt["x"]
            coordinate = target
            coordinate_path.append(coordinate)
            residual_path.append(attempt["residual"])
            if keep_states:
                states.append(x.copy())
            step = min(initial_step, 1.5 * step)
            continue

        step *= 0.5
        if step < minimum_step:
            return {
                "converged": False,
                "x": None,
                "reached": coordinate,
                "reason": attempt["reason"],
                "coordinate": np.asarray(coordinate_path),
                "residual": np.asarray(residual_path),
                "states": states,
            }

    return {
        "converged": True,
        "x": x,
        "reached": 1.0,
        "reason": "converged",
        "coordinate": np.asarray(coordinate_path),
        "residual": np.asarray(residual_path),
        "states": states,
    }


def isolated_noise_branches(
    noise_factors,
    baseline_params,
    baseline_state,
):
    """Follow each isolated cell from baseline to its noisy parameter vector."""
    zero_diffusion = np.zeros((3, 3), dtype=float)
    states = []
    diagnostics = []

    for cell, factor in enumerate(noise_factors):
        def system_at(s, cell_factor=factor):
            return [baseline_params * cell_factor**s], zero_diffusion

        result = continue_branch(
            baseline_state.copy(),
            system_at,
            n_cells=1,
            initial_step=0.10,
        )
        diagnostics.append(
            {
                "cell": cell,
                "converged": result["converged"],
                "reached": result["reached"],
                "reason": result["reason"],
                "final_residual": float(result["residual"][-1]),
            }
        )
        if not result["converged"]:
            return {
                "converged": False,
                "states": None,
                "diagnostics": diagnostics,
            }
        states.append(result["x"])

    return {
        "converged": True,
        "states": np.concatenate(states),
        "diagnostics": diagnostics,
    }


def coupled_noise_branch(
    noise_factors,
    baseline_params,
    baseline_state,
    Ldiff,
    n_cells,
):
    """Primary path: add heterogeneous parameter noise to the coupled ring."""
    x_uniform = np.tile(baseline_state, n_cells)

    def system_at(s):
        params = [
            baseline_params * noise_factors[cell] ** s
            for cell in range(n_cells)
        ]
        return params, Ldiff

    return continue_branch(
        x_uniform,
        system_at,
        n_cells=n_cells,
        initial_step=0.05,
    )


# ===========================================================================
# Spectrum and spatial-mode diagnostics
# ===========================================================================

def split_growth_rates(eigenvalues, imaginary_tolerance):
    eigenvalues = np.asarray(eigenvalues)
    stationary = np.abs(np.imag(eigenvalues)) <= imaginary_tolerance
    stationary_growth = (
        float(np.max(np.real(eigenvalues[stationary])))
        if np.any(stationary)
        else -np.inf
    )
    oscillatory_growth = (
        float(np.max(np.real(eigenvalues[~stationary])))
        if np.any(~stationary)
        else -np.inf
    )
    return stationary_growth, oscillatory_growth


def folded_fourier_power(eigenvectors, n_cells):
    """Return normalized +/- folded Fourier power for every eigenvector."""
    fields = eigenvectors.reshape(n_cells, 3, -1)
    power = np.abs(np.fft.fft(fields, axis=0)) ** 2
    power = power.sum(axis=1)  # (N, n_eigenvalues)

    n_unique = n_cells // 2 + 1
    folded = power[:n_unique].copy()
    for mode in range(1, (n_cells + 1) // 2):
        folded[mode] += power[n_cells - mode]

    denominator = folded.sum(axis=0, keepdims=True)
    folded = np.divide(
        folded,
        denominator,
        out=np.zeros_like(folded),
        where=denominator > 0,
    )
    return folded.T  # (n_eigenvalues, n_unique_modes)


def full_spectrum(J, n_cells, k_effective, imaginary_tolerance):
    """Exact full spectrum with descriptive Fourier and localization metrics."""
    eigenvalues, eigenvectors = np.linalg.eig(J)
    fourier_power = folded_fourier_power(eigenvectors, n_cells)
    dominant_mode = np.argmax(fourier_power, axis=1)
    mode_purity = fourier_power[
        np.arange(fourier_power.shape[0]),
        dominant_mode,
    ]
    k_centroid = fourier_power @ k_effective

    fields = eigenvectors.reshape(n_cells, 3, -1)
    cell_power = np.abs(fields) ** 2
    cell_power = cell_power.sum(axis=1).T  # (n_eigenvalues, n_cells)
    cell_power /= cell_power.sum(axis=1, keepdims=True)
    inverse_participation_ratio = np.sum(cell_power**2, axis=1)

    stationary_growth, oscillatory_growth = split_growth_rates(
        eigenvalues,
        imaginary_tolerance,
    )
    leading_index = int(np.argmax(np.real(eigenvalues)))

    # This envelope is a descriptive dominant-mode binning only.  It is not
    # used to classify stability and is not an exact heterogeneous dispersion.
    descriptive_envelope = np.full(k_effective.size, np.nan)
    bin_counts = np.zeros(k_effective.size, dtype=int)
    for mode in range(k_effective.size):
        selected = dominant_mode == mode
        bin_counts[mode] = int(np.sum(selected))
        if np.any(selected):
            descriptive_envelope[mode] = float(
                np.max(np.real(eigenvalues[selected]))
            )

    return {
        "eigenvalues": eigenvalues,
        "fourier_power": fourier_power,
        "dominant_mode": dominant_mode,
        "mode_purity": mode_purity,
        "k_centroid": k_centroid,
        "inverse_participation_ratio": inverse_participation_ratio,
        "stationary_growth": stationary_growth,
        "oscillatory_growth": oscillatory_growth,
        "spectral_abscissa": float(np.max(np.real(eigenvalues))),
        "leading_index": leading_index,
        "leading_eigenvalue": eigenvalues[leading_index],
        "leading_fourier_power": fourier_power[leading_index],
        "leading_dominant_mode": int(dominant_mode[leading_index]),
        "leading_mode_purity": float(mode_purity[leading_index]),
        "leading_k_centroid": float(k_centroid[leading_index]),
        "leading_ipr": float(inverse_participation_ratio[leading_index]),
        "descriptive_mode_envelope": descriptive_envelope,
        "descriptive_mode_bin_counts": bin_counts,
    }


def homogeneous_dispersion(
    baseline_state,
    baseline_params,
    hopping,
    n_cells,
    lattice_spacing,
    imaginary_tolerance,
):
    """Exact discrete homogeneous dispersion relation."""
    D = np.diag(
        [hopping["h_u"], hopping["h_v"], hopping["h_w"]]
    )
    k_effective = effective_wavenumbers(n_cells, lattice_spacing)
    J0 = reaction_jacobian(baseline_state, baseline_params)
    eigenvalues = np.array(
        [np.linalg.eigvals(J0 - k**2 * D) for k in k_effective]
    )
    stationary_growth = []
    oscillatory_growth = []
    for values in eigenvalues:
        stationary, oscillatory = split_growth_rates(
            values,
            imaginary_tolerance,
        )
        stationary_growth.append(stationary)
        oscillatory_growth.append(oscillatory)
    return {
        "k_effective": k_effective,
        "eigenvalues": eigenvalues,
        "spectral_abscissa": np.max(np.real(eigenvalues), axis=1),
        "stationary_growth": np.asarray(stationary_growth),
        "oscillatory_growth": np.asarray(oscillatory_growth),
    }


def classify_spectrum(
    uncoupled_stable,
    spectrum,
    stability_tolerance,
):
    stationary_unstable = spectrum["stationary_growth"] > stability_tolerance
    oscillatory_unstable = spectrum["oscillatory_growth"] > stability_tolerance

    if not uncoupled_stable:
        label = "uncoupled_unstable"
    elif stationary_unstable and oscillatory_unstable:
        label = "mixed_stationary_oscillatory"
    elif stationary_unstable:
        label = "pure_stationary_turing"
    elif oscillatory_unstable:
        label = "pure_oscillatory_coupling_instability"
    else:
        label = "stable_coupled_ring"

    return {
        "label": label,
        "stationary_diffusion_driven": bool(
            uncoupled_stable and stationary_unstable
        ),
        "oscillatory_diffusion_driven": bool(
            uncoupled_stable and oscillatory_unstable
        ),
        "pure_stationary_turing": label == "pure_stationary_turing",
        "mixed_instability": label == "mixed_stationary_oscillatory",
    }


def first_negative_to_positive_crossing(coordinate, growth, tolerance):
    coordinate = np.asarray(coordinate)
    growth = np.asarray(growth)
    for index in range(1, coordinate.size):
        left = growth[index - 1]
        right = growth[index]
        if left < -tolerance and right > tolerance:
            fraction = -left / (right - left)
            return float(
                coordinate[index - 1]
                + fraction * (coordinate[index] - coordinate[index - 1])
            )
    return np.nan


def coupling_branch(
    params_list,
    isolated_states,
    Ldiff,
    n_cells,
    k_effective,
    imaginary_tolerance,
    stability_tolerance,
):
    """Secondary path: turn on coupling between already-noisy isolated cells."""
    def system_at(gamma):
        return params_list, gamma * Ldiff

    continuation = continue_branch(
        isolated_states,
        system_at,
        n_cells=n_cells,
        initial_step=0.05,
        keep_states=True,
    )
    if not continuation["converged"]:
        return {
            "converged": False,
            "x": None,
            "reached": continuation["reached"],
            "reason": continuation["reason"],
            "gamma": continuation["coordinate"],
            "residual": continuation["residual"],
        }

    alpha_path = []
    stationary_path = []
    oscillatory_path = []
    for gamma, state in zip(
        continuation["coordinate"],
        continuation["states"],
    ):
        J = ring_jacobian(
            state,
            params_list,
            gamma * Ldiff,
            n_cells,
        )
        eigenvalues = np.linalg.eigvals(J)
        stationary, oscillatory = split_growth_rates(
            eigenvalues,
            imaginary_tolerance,
        )
        alpha_path.append(float(np.max(np.real(eigenvalues))))
        stationary_path.append(stationary)
        oscillatory_path.append(oscillatory)

    stationary_path = np.asarray(stationary_path)
    oscillatory_path = np.asarray(oscillatory_path)
    endpoint_J = ring_jacobian(
        continuation["x"],
        params_list,
        Ldiff,
        n_cells,
    )
    endpoint_spectrum = full_spectrum(
        endpoint_J,
        n_cells,
        k_effective,
        imaginary_tolerance,
    )

    return {
        "converged": True,
        "x": continuation["x"],
        "reached": 1.0,
        "reason": "converged",
        "gamma": continuation["coordinate"],
        "residual": continuation["residual"],
        "spectral_abscissa_path": np.asarray(alpha_path),
        "stationary_growth_path": stationary_path,
        "oscillatory_growth_path": oscillatory_path,
        "stationary_crossing_gamma": first_negative_to_positive_crossing(
            continuation["coordinate"],
            stationary_path,
            stability_tolerance,
        ),
        "oscillatory_crossing_gamma": first_negative_to_positive_crossing(
            continuation["coordinate"],
            oscillatory_path,
            stability_tolerance,
        ),
        "endpoint_spectrum": endpoint_spectrum,
    }


# ===========================================================================
# Optional nonlinear simulations
# ===========================================================================

def simulate_ring(
    params_list,
    Ldiff,
    n_cells,
    x_initial,
    time_end=5000.0,
    stages=(1000.0, 2500.0, 5000.0),
    rhs_tolerance=1e-8,
):
    rhs = lambda _t, x: ring_rhs(x, params_list, Ldiff, n_cells)
    jac = lambda _t, x: ring_jacobian(x, params_list, Ldiff, n_cells)
    x = np.asarray(x_initial, dtype=float)
    time = 0.0
    rhs_residual = scaled_residual(rhs(time, x), x)

    endpoints = sorted(set([t for t in stages if 0 < t <= time_end] + [time_end]))
    for endpoint in endpoints:
        solution = solve_ivp(
            rhs,
            (time, endpoint),
            x,
            method="LSODA",
            jac=jac,
            rtol=1e-7,
            atol=1e-10,
            t_eval=[endpoint],
        )
        if not solution.success:
            return {
                "success": False,
                "settled": False,
                "reason": solution.message,
                "state": None,
                "time": time,
                "rhs_residual": np.nan,
            }
        x = solution.y[:, -1]
        time = endpoint
        if not np.all(np.isfinite(x)) or np.any(x <= 0):
            return {
                "success": False,
                "settled": False,
                "reason": "nonpositive_or_nonfinite_state",
                "state": None,
                "time": time,
                "rhs_residual": np.nan,
            }
        rhs_residual = scaled_residual(rhs(time, x), x)
        if rhs_residual < rhs_tolerance:
            break

    return {
        "success": True,
        "settled": bool(rhs_residual < rhs_tolerance),
        "reason": "steady" if rhs_residual < rhs_tolerance else "time_end",
        "state": x,
        "time": time,
        "rhs_residual": rhs_residual,
    }


def nonlinear_metrics(final_state, equilibrium, n_cells, species=0):
    final_profile = final_state.reshape(n_cells, 3)[:, species]
    equilibrium_profile = equilibrium.reshape(n_cells, 3)[:, species]
    delta = final_profile - equilibrium_profile
    scale = max(abs(np.mean(equilibrium_profile)), 1e-12)

    delta_amplitude = float((np.max(delta) - np.min(delta)) / scale)
    forced_amplitude = float(
        (np.max(equilibrium_profile) - np.min(equilibrium_profile))
        / max(abs(np.mean(equilibrium_profile)), 1e-12)
    )
    final_amplitude = float(
        (np.max(final_profile) - np.min(final_profile))
        / max(abs(np.mean(final_profile)), 1e-12)
    )

    centered = delta - np.mean(delta)
    power = np.abs(np.fft.fft(centered)) ** 2
    n_unique = n_cells // 2 + 1
    folded = power[:n_unique].copy()
    for mode in range(1, (n_cells + 1) // 2):
        folded[mode] += power[n_cells - mode]
    folded = folded / folded.sum() if folded.sum() > 0 else np.zeros_like(folded)
    dominant_mode = int(np.argmax(folded[1:]) + 1) if folded[1:].sum() > 0 else 0

    return {
        "delta_amplitude": delta_amplitude,
        "forced_equilibrium_amplitude": forced_amplitude,
        "final_amplitude": final_amplitude,
        "delta_fourier_power": folded,
        "dominant_mode": dominant_mode,
    }


# ===========================================================================
# Noise, validation and statistics
# ===========================================================================

def lognormal_cv_factors(CV, standard_normal_draws, center_within_ring=False):
    """Mean-one lognormal multipliers with the requested ensemble CV."""
    sigma = np.sqrt(np.log1p(CV**2))
    mu = -0.5 * sigma**2
    factors = np.exp(mu + sigma * standard_normal_draws)
    if center_within_ring:
        # Optional alternative design: each kinetic parameter has arithmetic
        # mean multiplier exactly one across the cells in each ring.
        factors = factors / factors.mean(axis=1, keepdims=True)
    return factors


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return 100.0 * (center - half_width), 100.0 * (center + half_width)


def validate_baseline(
    baseline_params,
    baseline_state,
    hopping,
    n_cells,
    lattice_spacing,
    stability_tolerance,
    imaginary_tolerance,
):
    baseline_params = np.asarray(baseline_params, dtype=float)
    baseline_state = np.asarray(baseline_state, dtype=float)
    if baseline_params.shape != (N_PARAMETERS,):
        raise ValueError("The baseline parameter vector must contain 16 values.")
    if baseline_state.shape != (3,):
        raise ValueError("The baseline state must contain [u*, v*, w*].")
    if np.any(baseline_params <= 0) or np.any(baseline_state <= 0):
        raise ValueError("Baseline parameters and concentrations must be positive.")

    reaction_residual = scaled_residual(
        reaction_rhs(baseline_state, baseline_params),
        baseline_state,
    )
    if reaction_residual > 1e-7:
        raise ValueError(
            "The tabulated baseline state is not an accurate reaction equilibrium: "
            f"scaled residual={reaction_residual:.3e}."
        )

    dispersion = homogeneous_dispersion(
        baseline_state,
        baseline_params,
        hopping,
        n_cells,
        lattice_spacing,
        imaginary_tolerance,
    )
    local_stable = dispersion["spectral_abscissa"][0] < -stability_tolerance
    stationary_finite_mode = (
        np.max(dispersion["stationary_growth"][1:]) > stability_tolerance
    )
    oscillatory_finite_mode = (
        np.max(dispersion["oscillatory_growth"][1:]) > stability_tolerance
    )
    if not local_stable:
        raise ValueError(
            "The homogeneous reaction equilibrium is not stable at mode 0."
        )
    if not stationary_finite_mode:
        raise ValueError(
            "The selected baseline has no stationary instability on the discrete "
            f"{n_cells}-cell ring. The continuum instability may fall between "
            "the ring's accessible wavenumbers."
        )
    if oscillatory_finite_mode:
        raise ValueError(
            "The baseline also has an oscillatory finite-mode instability and is "
            "not a clean stationary Type-I reference."
        )

    dominant_mode = int(
        np.argmax(dispersion["stationary_growth"][1:]) + 1
    )
    return {
        "reaction_scaled_residual": reaction_residual,
        "dispersion": dispersion,
        "dominant_mode": dominant_mode,
    }


def endpoint_log_distance(x_a, x_b):
    return float(np.max(np.abs(np.log(np.asarray(x_a) / np.asarray(x_b)))))


# ===========================================================================
# Main Monte Carlo experiment
# ===========================================================================

def run_experiment(
    baseline_params,
    baseline_state,
    hopping,
    n_cells=10,
    n_trials=200,
    CV_values=DEFAULT_CV_VALUES,
    seed=42,
    lattice_spacing=1.0,
    stability_tolerance=1e-8,
    imaginary_tolerance=1e-7,
    branch_match_tolerance=1e-5,
    center_within_ring=False,
    run_nonlinear=False,
    n_initial_conditions=3,
    initial_perturbation=1e-3,
    nonlinear_amplitude_tolerance=0.05,
    verbose=True,
):
    """Run paired heterogeneous-ring trials over all requested CV values."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if n_initial_conditions <= 0:
        raise ValueError("n_initial_conditions must be positive.")

    Ldiff = build_diffusion_operator(
        n_cells,
        hopping,
        lattice_spacing,
    )
    baseline = validate_baseline(
        baseline_params,
        baseline_state,
        hopping,
        n_cells,
        lattice_spacing,
        stability_tolerance,
        imaginary_tolerance,
    )
    k_effective = baseline["dispersion"]["k_effective"]

    # Separate reproducible streams.  The same underlying draws are reused at
    # each CV, producing a paired noise-response experiment.
    seed_sequence = np.random.SeedSequence(seed)
    parameter_seed, initial_condition_seed = seed_sequence.spawn(2)
    parameter_rng = np.random.default_rng(parameter_seed)
    initial_rng = np.random.default_rng(initial_condition_seed)
    common_parameter_draws = parameter_rng.standard_normal(
        (n_trials, n_cells, N_PARAMETERS)
    )
    common_initial_directions = initial_rng.standard_normal(
        (n_trials, n_initial_conditions, 3 * n_cells)
    )

    results = []
    for CV in CV_values:
        all_factors = lognormal_cv_factors(
            CV,
            common_parameter_draws,
            center_within_ring=center_within_ring,
        )
        trial_records = []

        for trial_index in range(n_trials):
            # if verbose and trial_index and trial_index % 25 == 0:
            #     print(
            #         f"CV={CV:.2f}: trial {trial_index}/{n_trials}",
            #         flush=True,
            #     )

            factors = all_factors[trial_index]
            params_list = [
                baseline_params * factors[cell]
                for cell in range(n_cells)
            ]
            record = {
                "trial": trial_index,
                "CV": CV,
                "status": "started",
                "noise_factors": factors,
                "parameter_vectors": np.asarray(params_list),
                "realized_parameter_mean_factors": factors.mean(axis=0),
                "realized_parameter_CVs_across_cells": (
                    factors.std(axis=0, ddof=0) / factors.mean(axis=0)
                ),
            }

            # Ten baseline-connected isolated equilibria.
            isolated = isolated_noise_branches(
                factors,
                baseline_params,
                baseline_state,
            )
            record["isolated_diagnostics"] = isolated["diagnostics"]
            if not isolated["converged"]:
                record["status"] = "isolated_continuation_failed"
                trial_records.append(record)
                continue

            isolated_states = isolated["states"]
            zero_diffusion = np.zeros_like(Ldiff)
            J_uncoupled = ring_jacobian(
                isolated_states,
                params_list,
                zero_diffusion,
                n_cells,
            )
            uncoupled_eigenvalues = np.linalg.eigvals(J_uncoupled)
            alpha_uncoupled = float(np.max(np.real(uncoupled_eigenvalues)))
            uncoupled_stable = alpha_uncoupled < -stability_tolerance
            record.update(
                {
                    "isolated_states": isolated_states,
                    "uncoupled_eigenvalues": uncoupled_eigenvalues,
                    "alpha_uncoupled": alpha_uncoupled,
                    "uncoupled_stable": bool(uncoupled_stable),
                }
            )

            # Primary endpoint: the already-coupled ring receives noise.
            primary = coupled_noise_branch(
                factors,
                baseline_params,
                baseline_state,
                Ldiff,
                n_cells,
            )
            record["noise_path_diagnostics"] = {
                "converged": primary["converged"],
                "reached": primary["reached"],
                "reason": primary["reason"],
                "coordinate": primary["coordinate"],
                "residual": primary["residual"],
            }
            if not primary["converged"]:
                record["status"] = "coupled_noise_continuation_failed"
                trial_records.append(record)
                continue

            primary_state = primary["x"]
            primary_J = ring_jacobian(
                primary_state,
                params_list,
                Ldiff,
                n_cells,
            )
            primary_spectrum = full_spectrum(
                primary_J,
                n_cells,
                k_effective,
                imaginary_tolerance,
            )
            primary_classification = classify_spectrum(
                uncoupled_stable,
                primary_spectrum,
                stability_tolerance,
            )
            record.update(
                {
                    "primary_coupled_state": primary_state,
                    "primary_coupled_residual": float(primary["residual"][-1]),
                    "primary_spectrum": primary_spectrum,
                    "primary_classification": primary_classification,
                }
            )

            # Secondary endpoint and stability path: noisy cells are coupled.
            secondary = coupling_branch(
                params_list,
                isolated_states,
                Ldiff,
                n_cells,
                k_effective,
                imaginary_tolerance,
                stability_tolerance,
            )
            record["coupling_path"] = {
                key: value
                for key, value in secondary.items()
                if key not in ("x", "endpoint_spectrum")
            }
            if secondary["converged"]:
                secondary_classification = classify_spectrum(
                    uncoupled_stable,
                    secondary["endpoint_spectrum"],
                    stability_tolerance,
                )
                branch_distance = endpoint_log_distance(
                    primary_state,
                    secondary["x"],
                )
                branch_consistent = branch_distance <= branch_match_tolerance
                record.update(
                    {
                        "coupling_path_endpoint_state": secondary["x"],
                        "coupling_path_endpoint_spectrum": secondary["endpoint_spectrum"],
                        "coupling_path_classification": secondary_classification,
                        "branch_log_distance": branch_distance,
                        "branch_consistent": bool(branch_consistent),
                    }
                )
            else:
                record.update(
                    {
                        "coupling_path_endpoint_state": None,
                        "coupling_path_endpoint_spectrum": None,
                        "coupling_path_classification": None,
                        "branch_log_distance": np.nan,
                        "branch_consistent": False,
                    }
                )

            record["unambiguous_pure_stationary_turing"] = bool(
                primary_classification["pure_stationary_turing"]
                and record["branch_consistent"]
                and record["coupling_path_classification"] is not None
                and record["coupling_path_classification"]["pure_stationary_turing"]
            )
            record["status"] = "classified"

            if run_nonlinear:
                nonlinear_runs = []
                for initial_index in range(n_initial_conditions):
                    direction = common_initial_directions[
                        trial_index,
                        initial_index,
                    ]
                    initial_state = primary_state * np.exp(
                        initial_perturbation * direction
                    )
                    simulation = simulate_ring(
                        params_list,
                        Ldiff,
                        n_cells,
                        initial_state,
                    )
                    nonlinear_record = simulation.copy()
                    if simulation["success"]:
                        metrics = nonlinear_metrics(
                            simulation["state"],
                            primary_state,
                            n_cells,
                        )
                        nonlinear_record["metrics"] = metrics
                        nonlinear_record["pattern_grew"] = bool(
                            simulation["settled"]
                            and metrics["delta_amplitude"]
                            > nonlinear_amplitude_tolerance
                            and metrics["dominant_mode"] > 0
                        )
                    else:
                        nonlinear_record["metrics"] = None
                        nonlinear_record["pattern_grew"] = False
                    nonlinear_runs.append(nonlinear_record)

                successful_runs = [
                    run for run in nonlinear_runs if run["success"]
                ]
                required = n_initial_conditions // 2 + 1
                nonlinear_valid = len(successful_runs) >= required
                majority_pattern = bool(
                    nonlinear_valid
                    and sum(run["pattern_grew"] for run in successful_runs)
                    >= (len(successful_runs) // 2 + 1)
                )
                record.update(
                    {
                        "nonlinear_runs": nonlinear_runs,
                        "nonlinear_valid": nonlinear_valid,
                        "nonlinear_pattern_grew": majority_pattern,
                        "nonlinear_confirmed_unambiguous_turing": bool(
                            record["unambiguous_pure_stationary_turing"]
                            and majority_pattern
                        ),
                    }
                )

            trial_records.append(record)

        classified = [
            record
            for record in trial_records
            if record["status"] == "classified"
        ]
        n_failures = n_trials - len(classified)
        n_uncoupled_unstable = sum(
            not record["uncoupled_stable"]
            for record in classified
        )
        n_primary_pure_turing = sum(
            record["primary_classification"]["pure_stationary_turing"]
            for record in classified
        )
        n_primary_mixed = sum(
            record["primary_classification"]["mixed_instability"]
            for record in classified
        )
        n_primary_oscillatory_only = sum(
            record["primary_classification"]["label"]
            == "pure_oscillatory_coupling_instability"
            for record in classified
        )
        n_path_dependent = sum(
            record["coupling_path"]["converged"]
            and not record["branch_consistent"]
            for record in classified
        )
        n_coupling_path_failed = sum(
            not record["coupling_path"]["converged"]
            for record in classified
        )
        n_unambiguous_turing = sum(
            record["unambiguous_pure_stationary_turing"]
            for record in classified
        )
        confidence_low, confidence_high = wilson_interval(
            n_unambiguous_turing,
            n_trials,
        )
        alpha_values = np.array(
            [
                record["primary_spectrum"]["spectral_abscissa"]
                for record in classified
            ]
        )

        summary = {
            "CV": CV,
            "n_trials": n_trials,
            "n_classified": len(classified),
            "n_failures": n_failures,
            "n_uncoupled_unstable": int(n_uncoupled_unstable),
            "n_primary_pure_stationary_turing": int(n_primary_pure_turing),
            "n_primary_mixed_instability": int(n_primary_mixed),
            "n_primary_oscillatory_only": int(n_primary_oscillatory_only),
            "n_path_dependent": int(n_path_dependent),
            "n_coupling_path_failed": int(n_coupling_path_failed),
            "n_unambiguous_pure_stationary_turing": int(n_unambiguous_turing),
            "unambiguous_turing_percent_of_all": (
                100.0 * n_unambiguous_turing / n_trials
            ),
            "unambiguous_turing_Wilson95_low": confidence_low,
            "unambiguous_turing_Wilson95_high": confidence_high,
            "max_Re_lambda_mean": (
                float(np.mean(alpha_values)) if alpha_values.size else np.nan
            ),
            "max_Re_lambda_median": (
                float(np.median(alpha_values)) if alpha_values.size else np.nan
            ),
            "max_Re_lambda_q025": (
                float(np.quantile(alpha_values, 0.025))
                if alpha_values.size
                else np.nan
            ),
            "max_Re_lambda_q975": (
                float(np.quantile(alpha_values, 0.975))
                if alpha_values.size
                else np.nan
            ),
        }
        if run_nonlinear:
            n_nonlinear_valid = sum(
                record.get("nonlinear_valid", False)
                for record in classified
            )
            n_nonlinear_confirmed = sum(
                record.get("nonlinear_confirmed_unambiguous_turing", False)
                for record in classified
            )
            summary.update(
                {
                    "n_nonlinear_valid": int(n_nonlinear_valid),
                    "n_nonlinear_confirmed_unambiguous_turing": int(
                        n_nonlinear_confirmed
                    ),
                }
            )

        results.append(
            {
                "summary": summary,
                "trials": trial_records,
            }
        )
        if verbose:
            print(
                f"CV={CV:.2f}: unambiguous stationary Turing "
                f"{n_unambiguous_turing}/{n_trials} "
                f"({summary['unambiguous_turing_percent_of_all']:.1f}%); "
                f"primary pure={n_primary_pure_turing}, mixed={n_primary_mixed}, "
                f"oscillatory={n_primary_oscillatory_only}, "
                f"path-dependent={n_path_dependent}, "
                f"coupling-path-failed={n_coupling_path_failed}, "
                f"primary failures={n_failures}; "
                f"median max Re(lambda)="
                f"{summary['max_Re_lambda_median']:+.5g}",
                flush=True,
            )

    metadata = {
        "method_version": "final-2026-07-25",
        "seed": seed,
        "n_cells": n_cells,
        "n_trials": n_trials,
        "CV_values": tuple(CV_values),
        "noise_model": "independent mean-one lognormal multipliers",
        "center_within_ring": center_within_ring,
        "noisy_parameters": "16 kinetic parameters only",
        "hopping_is_noisy": False,
        "lattice_spacing": lattice_spacing,
        "stability_tolerance": stability_tolerance,
        "imaginary_tolerance": imaginary_tolerance,
        "branch_match_tolerance_log_concentration": branch_match_tolerance,
        "run_nonlinear": run_nonlinear,
        "baseline": baseline,
        "baseline_params": np.asarray(baseline_params),
        "baseline_state": np.asarray(baseline_state),
        "hopping": hopping,
        "classification_note": (
            "The conservative primary count requires stable isolated noisy cells, "
            "a pure stationary instability at full coupling, agreement between "
            "the noise and coupling continuation endpoints, and the same pure "
            "stationary classification on both paths."
        ),
        "heterogeneous_mode_note": (
            "Heterogeneous Fourier modes are coupled. Saved Fourier powers and "
            "dominant-mode envelopes are descriptive and are not used as an exact "
            "dispersion relation or stability criterion."
        ),
    }
    return results, metadata


# ===========================================================================
# CSV input and output
# ===========================================================================

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


def load_parameter_row(csv_path, config_id, param_rank):
    with Path(csv_path).open(newline="") as handle:
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
    params = np.array(
        [float(row[column]) for column in PARAMETER_COLUMNS],
        dtype=float,
    )
    state = np.array(
        [float(row["u_star"]), float(row["v_star"]), float(row["w_star"])],
        dtype=float,
    )
    hopping = {
        "h_u": float(row["dU"]),
        "h_v": float(row["dV"]),
        "h_w": float(row["dW"]),
    }
    return params, state, hopping, row


def write_summary_csv(results, path):
    rows = [entry["summary"] for entry in results]
    if not rows:
        raise ValueError("No results were generated.")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "../TopologyRanking/Topology3954/"
            "3954_FINAL_lhs_results_parameters.csv"
        ),
        help="Topology 3954 parameter-results CSV.",
    )
    parser.add_argument("--config-id", type=int, default=21)
    parser.add_argument("--param-rank", type=int, default=1)
    parser.add_argument("--config-label", default="high")
    parser.add_argument("--n-cells", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--CV-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_CV_VALUES),
    )
    parser.add_argument("--lattice-spacing", type=float, default=1.0)
    parser.add_argument(
        "--center-within-ring",
        action="store_true",
        help=(
            "Normalize every parameter's multiplier to arithmetic mean one "
            "within each ring. Off by default to reproduce the original noise."
        ),
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="Also run nonlinear integrations from perturbed coupled equilibria.",
    )
    parser.add_argument("--n-initial-conditions", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    baseline_params, baseline_state, hopping, source_row = load_parameter_row(
        arguments.csv,
        arguments.config_id,
        arguments.param_rank,
    )
    results, metadata = run_experiment(
        baseline_params,
        baseline_state,
        hopping,
        n_cells=arguments.n_cells,
        n_trials=arguments.n_trials,
        CV_values=tuple(arguments.CV_values),
        seed=arguments.seed,
        lattice_spacing=arguments.lattice_spacing,
        center_within_ring=arguments.center_within_ring,
        run_nonlinear=arguments.nonlinear,
        n_initial_conditions=arguments.n_initial_conditions,
    )
    metadata.update(
        {
            "config_id": arguments.config_id,
            "param_rank": arguments.param_rank,
            "config_name": source_row.get("config_name", ""),
            "source_csv": str(arguments.csv.resolve()),
        }
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"3954_heterogeneous_ring_{arguments.config_label}"
        f"_config{arguments.config_id}_N{arguments.n_cells}_FINAL"
    )
    pickle_path = arguments.output_dir / f"{stem}.pkl"
    summary_path = arguments.output_dir / f"{stem}_summary.csv"

    with pickle_path.open("wb") as handle:
        pickle.dump(
            {"metadata": metadata, "results": results},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    write_summary_csv(results, summary_path)
    print(f"Saved complete trial data: {pickle_path}")
    print(f"Saved CV summary:          {summary_path}")


if __name__ == "__main__":
    main()
