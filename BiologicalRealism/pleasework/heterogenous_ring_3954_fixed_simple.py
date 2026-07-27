#!/usr/bin/env python3
"""Simple corrected version of the original heterogeneous-ring experiment.

This intentionally keeps the structure and lognormal CV sampling of the first
draft.  The important correction is that the 30x30 Jacobian is evaluated at a
steady state of the *coupled ring*, not at ten isolated steady states pasted
together.

For a heterogeneous ring, Fourier modes are mixed, so there is no exact
dispersion relation lambda(k).  We therefore use this generalized criterion:

    1. all ten uncoupled noisy cells are stable;
    2. the fully coupled ring is unstable.

At CV=0 this reduces to the usual homogeneous diffusion-driven test.  For
CV>0, the leading eigenvector's Fourier mode is saved only as a description of
the spatial structure; it is not treated as an exact independent mode.
"""

import pickle

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_HILL = 2
CONFIG_TO_TEST = 49
CONFIG_LABEL = "high"
N_CELLS = 20
N_TRIALS = 200
SEED = 42
CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
STABILITY_TOL = 1e-8

CSV_PATH = "../TopologyRanking/Topology3954/3954_FINAL_lhs_results_parameters.csv"


# ---------------------------------------------------------------------------
# Topology 3954 reaction kinetics
# ---------------------------------------------------------------------------

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


def ode_system(state, params):
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
    return np.array([du, dv, dw])


def compute_jacobian(state, params):
    u, v, w = state
    _, beta_u, K_uu, K_vu, delta_u = params[0:5]
    _, beta_v, K_uv, K_wv, delta_v = params[5:10]
    _, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]

    J = np.zeros((3, 3))
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
# Correct periodic ring equations
# ---------------------------------------------------------------------------

def build_diffusion_operator(n_cells, hopping):
    """Periodic nearest-neighbour diffusion for [u0,v0,w0,u1,v1,w1,...]."""
    h = np.array([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    Ldiff = np.zeros((3 * n_cells, 3 * n_cells))

    for i in range(n_cells):
        left = (i - 1) % n_cells
        right = (i + 1) % n_cells
        for species in range(3):
            row = 3 * i + species
            Ldiff[row, row] -= 2.0 * h[species]
            Ldiff[row, 3 * left + species] += h[species]
            Ldiff[row, 3 * right + species] += h[species]
    return Ldiff


def ring_rhs(x, params_list, Ldiff, n_cells):
    X = x.reshape(n_cells, 3)
    reactions = np.concatenate(
        [ode_system(X[i], params_list[i]) for i in range(n_cells)]
    )
    return reactions + Ldiff @ x


def ring_jacobian(x, params_list, Ldiff, n_cells):
    J = Ldiff.copy()
    X = x.reshape(n_cells, 3)
    for i in range(n_cells):
        block = slice(3 * i, 3 * i + 3)
        J[block, block] += compute_jacobian(X[i], params_list[i])
    return J


# ---------------------------------------------------------------------------
# Positive steady-state solver
# ---------------------------------------------------------------------------

def scaled_residual(f, x):
    return np.linalg.norm(f, ord=np.inf) / (
        1.0 + np.linalg.norm(x, ord=np.inf)
    )


def newton_log(
    x0,
    params_list,
    Ldiff,
    n_cells,
    tolerance=1e-9,
    max_iterations=80,
):
    """Solve ring_rhs(x)=0 using y=log(x), which keeps concentrations positive."""
    if np.any(x0 <= 0):
        return None, np.inf

    y = np.log(x0)
    for _ in range(max_iterations):
        x = np.exp(y)
        f = ring_rhs(x, params_list, Ldiff, n_cells)
        residual = scaled_residual(f, x)
        if residual < tolerance:
            return x, residual

        # dF/dy = dF/dx @ diag(x)
        Jy = ring_jacobian(x, params_list, Ldiff, n_cells) * x[None, :]
        try:
            step_direction = np.linalg.solve(Jy, -f)
        except np.linalg.LinAlgError:
            return None, residual

        # Limit the largest log-concentration change.
        step_size = min(
            1.0,
            0.5 / max(np.max(np.abs(step_direction)), 1e-300),
        )

        accepted = False
        for _ in range(30):
            y_trial = y + step_size * step_direction
            x_trial = np.exp(y_trial)
            f_trial = ring_rhs(x_trial, params_list, Ldiff, n_cells)
            trial_residual = scaled_residual(f_trial, x_trial)
            if np.isfinite(trial_residual) and trial_residual < residual:
                y = y_trial
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            return None, residual

    x = np.exp(y)
    return None, scaled_residual(
        ring_rhs(x, params_list, Ldiff, n_cells),
        x,
    )


def continue_to_noisy_parameters(
    noise_factors,
    baseline_params,
    baseline_state,
    Ldiff,
    n_cells,
    initial_step=0.05,
):
    """Track the ring equilibrium from CV=0 parameters to one noisy sample.

    At continuation coordinate s:
        params_i(s) = baseline_params * noise_factors[i]**s

    s=0 is the known homogeneous equilibrium and s=1 is the requested sample.
    """
    x = np.tile(baseline_state, n_cells)
    s = 0.0
    step = initial_step
    last_residual = np.nan

    while s < 1.0 - 1e-14:
        target = min(1.0, s + step)
        params_list = [
            baseline_params * noise_factors[i] ** target
            for i in range(n_cells)
        ]
        x_new, residual = newton_log(
            x,
            params_list,
            Ldiff,
            n_cells,
        )

        if x_new is not None:
            x = x_new
            s = target
            last_residual = residual
            step = min(initial_step, 1.5 * step)
        else:
            step *= 0.5
            if step < 1.0 / 2048.0:
                return None, {
                    "status": "coupled_equilibrium_failed",
                    "s_reached": s,
                    "residual": residual,
                }

    final_params = [
        baseline_params * noise_factors[i]
        for i in range(n_cells)
    ]
    return x, {
        "status": "success",
        "s_reached": 1.0,
        "residual": last_residual,
        "params_list": final_params,
    }


def continue_isolated_cell(
    noise_factor,
    baseline_params,
    baseline_state,
    initial_step=0.1,
):
    """Track one isolated cell to its noisy steady state on the same branch."""
    zero_diffusion = np.zeros((3, 3))
    x = baseline_state.copy()
    s = 0.0
    step = initial_step

    while s < 1.0 - 1e-14:
        target = min(1.0, s + step)
        params = baseline_params * noise_factor**target
        x_new, residual = newton_log(
            x,
            [params],
            zero_diffusion,
            n_cells=1,
        )
        if x_new is not None:
            x = x_new
            s = target
            step = min(initial_step, 1.5 * step)
        else:
            step *= 0.5
            if step < 1.0 / 2048.0:
                return None, residual

    return x, residual


# ---------------------------------------------------------------------------
# Stability and spatial-content measurements
# ---------------------------------------------------------------------------

def max_real_eigenvalue(J):
    return float(np.max(np.real(np.linalg.eigvals(J))))


def uncoupled_stability(
    noise_factors,
    baseline_params,
    baseline_state,
):
    """Return the largest reaction-only growth rate among the ten noisy cells."""
    cell_states = []
    cell_growth_rates = []

    for i in range(noise_factors.shape[0]):
        state, residual = continue_isolated_cell(
            noise_factors[i],
            baseline_params,
            baseline_state,
        )
        if state is None:
            return None
        params = baseline_params * noise_factors[i]
        cell_states.append(state)
        cell_growth_rates.append(
            max_real_eigenvalue(compute_jacobian(state, params))
        )

    return {
        "states": np.asarray(cell_states),
        "cell_growth_rates": np.asarray(cell_growth_rates),
        "alpha_uncoupled": float(np.max(cell_growth_rates)),
    }


def leading_eigenvector_spatial_content(J_ring, n_cells):
    """Descriptive Fourier content of the full ring's leading eigenvector."""
    eigenvalues, eigenvectors = np.linalg.eig(J_ring)
    leading_index = int(np.argmax(np.real(eigenvalues)))
    leading_vector = eigenvectors[:, leading_index].reshape(n_cells, 3)

    power = np.abs(np.fft.fft(leading_vector, axis=0)) ** 2
    power = power.sum(axis=1)
    n_unique = n_cells // 2 + 1
    folded = power[:n_unique].copy()
    for mode in range(1, (n_cells + 1) // 2):
        folded[mode] += power[n_cells - mode]
    folded /= folded.sum()

    dominant_mode = int(np.argmax(folded))
    return {
        "leading_eigenvalue": eigenvalues[leading_index],
        "fourier_power": folded,
        "dominant_mode": dominant_mode,
        "mode_purity": float(folded[dominant_mode]),
    }


def homogeneous_ring_dispersion(
    steady_state,
    params,
    hopping,
    n_cells,
):
    """Exact discrete dispersion relation, valid only for the homogeneous ring."""
    J_local = compute_jacobian(steady_state, params)
    D = np.diag([hopping["h_u"], hopping["h_v"], hopping["h_w"]])
    modes = np.arange(n_cells // 2 + 1)
    k_effective = 2.0 * np.sin(np.pi * modes / n_cells)
    max_re = np.array(
        [
            np.max(np.real(np.linalg.eigvals(J_local - k**2 * D)))
            for k in k_effective
        ]
    )
    return k_effective, max_re


# ---------------------------------------------------------------------------
# Monte Carlo experiment
# ---------------------------------------------------------------------------

def run_cv_sweep(
    baseline_params,
    baseline_state,
    hopping,
    n_cells=N_CELLS,
    n_trials=N_TRIALS,
    cv_values=CV_VALUES,
    seed=SEED,
):
    Ldiff = build_diffusion_operator(n_cells, hopping)
    x_uniform = np.tile(baseline_state, n_cells)

    baseline_ring_residual = scaled_residual(
        ring_rhs(
            x_uniform,
            [baseline_params] * n_cells,
            Ldiff,
            n_cells,
        ),
        x_uniform,
    )
    if baseline_ring_residual > 1e-7:
        raise ValueError(
            "The tabulated baseline state is not an accurate ring equilibrium: "
            f"scaled residual={baseline_ring_residual:.3e}"
        )

    k_baseline, dispersion_baseline = homogeneous_ring_dispersion(
        baseline_state,
        baseline_params,
        hopping,
        n_cells,
    )
    if not (
        dispersion_baseline[0] < -STABILITY_TOL
        and np.max(dispersion_baseline[1:]) > STABILITY_TOL
    ):
        raise ValueError(
            "This parameter set is not Turing unstable on the discrete "
            f"{n_cells}-cell ring. max Re(lambda) by mode = "
            f"{dispersion_baseline}"
        )

    print("Baseline discrete wavenumbers:", np.round(k_baseline, 5))
    print("Baseline max Re(lambda):       ", np.round(dispersion_baseline, 6))

    # Reuse the same underlying random samples at every CV.  This does not
    # change the lognormal distribution; it makes comparisons between CVs paired.
    rng = np.random.default_rng(seed)
    standard_normal_draws = rng.standard_normal(
        (n_trials, n_cells, 16)
    )

    results_by_cv = []
    for CV in cv_values:
        sigma = np.sqrt(np.log(1.0 + CV**2))
        mu = -0.5 * sigma**2
        all_noise_factors = np.exp(
            mu + sigma * standard_normal_draws
        )

        trial_results = []
        for trial in range(n_trials):
            # if trial and trial % 25 == 0:
            #     print(f"CV={CV:.2f}: trial {trial}/{n_trials}", flush=True)

            noise_factors = all_noise_factors[trial]

            uncoupled = uncoupled_stability(
                noise_factors,
                baseline_params,
                baseline_state,
            )
            if uncoupled is None:
                trial_results.append(
                    {
                        "trial": trial,
                        "status": "isolated_equilibrium_failed",
                        "noise_factors": noise_factors,
                    }
                )
                continue

            x_coupled, coupled_info = continue_to_noisy_parameters(
                noise_factors,
                baseline_params,
                baseline_state,
                Ldiff,
                n_cells,
            )
            if x_coupled is None:
                trial_results.append(
                    {
                        "trial": trial,
                        "status": coupled_info["status"],
                        "s_reached": coupled_info["s_reached"],
                        "residual": coupled_info["residual"],
                        "alpha_uncoupled": uncoupled["alpha_uncoupled"],
                        "noise_factors": noise_factors,
                    }
                )
                continue

            params_list = coupled_info["params_list"]
            J_ring = ring_jacobian(
                x_coupled,
                params_list,
                Ldiff,
                n_cells,
            )
            spatial = leading_eigenvector_spatial_content(
                J_ring,
                n_cells,
            )

            alpha_uncoupled = uncoupled["alpha_uncoupled"]
            alpha_coupled = float(np.real(spatial["leading_eigenvalue"]))
            uncoupled_stable = alpha_uncoupled < -STABILITY_TOL
            coupled_unstable = alpha_coupled > STABILITY_TOL
            diffusion_driven = bool(
                uncoupled_stable and coupled_unstable
            )

            trial_results.append(
                {
                    "trial": trial,
                    "status": "success",
                    "noise_factors": noise_factors,
                    "params": np.asarray(params_list),
                    "isolated_states": uncoupled["states"],
                    "coupled_state": x_coupled,
                    "coupled_residual": coupled_info["residual"],
                    "cell_growth_rates": uncoupled["cell_growth_rates"],
                    "alpha_uncoupled": alpha_uncoupled,
                    "alpha_coupled": alpha_coupled,
                    "uncoupled_stable": uncoupled_stable,
                    "coupled_unstable": coupled_unstable,
                    "diffusion_driven_instability": diffusion_driven,
                    # Descriptive only: heterogeneous modes are mixed.
                    "leading_dominant_mode": spatial["dominant_mode"],
                    "leading_mode_purity": spatial["mode_purity"],
                    "leading_fourier_power": spatial["fourier_power"],
                }
            )

        valid = [
            record
            for record in trial_results
            if record["status"] == "success"
        ]
        n_failed = n_trials - len(valid)
        n_diffusion_driven = sum(
            record["diffusion_driven_instability"]
            for record in valid
        )
        n_uncoupled_unstable = sum(
            not record["uncoupled_stable"]
            for record in valid
        )
        alpha_coupled_values = np.array(
            [record["alpha_coupled"] for record in valid]
        )

        summary = {
            "CV": CV,
            "n_trials": n_trials,
            "n_valid": len(valid),
            "n_failed": n_failed,
            "n_uncoupled_unstable": int(n_uncoupled_unstable),
            "n_diffusion_driven": int(n_diffusion_driven),
            # Primary robustness percentage: failures remain in denominator.
            "robustness_percent_of_all": (
                100.0 * n_diffusion_driven / n_trials
            ),
            "mean_max_re_lambda": (
                float(np.mean(alpha_coupled_values))
                if len(valid)
                else np.nan
            ),
            "median_max_re_lambda": (
                float(np.median(alpha_coupled_values))
                if len(valid)
                else np.nan
            ),
            "std_max_re_lambda": (
                float(np.std(alpha_coupled_values))
                if len(valid)
                else np.nan
            ),
            "all_max_re_lambda": alpha_coupled_values,
        }

        results_by_cv.append(
            {
                "summary": summary,
                "trials": trial_results,
            }
        )
        print(
            f"CV={CV:.2f}: diffusion-driven "
            f"{n_diffusion_driven}/{n_trials} "
            f"({summary['robustness_percent_of_all']:.1f}%), "
            f"failed={n_failed}, "
            f"mean max Re(lambda)="
            f"{summary['mean_max_re_lambda']:+.6f}"
        )

    baseline_info = {
        "k_effective": k_baseline,
        "max_re_lambda_by_mode": dispersion_baseline,
        "dominant_mode": int(
            np.argmax(dispersion_baseline[1:]) + 1
        ),
    }
    return results_by_cv, baseline_info


def main():
    df = pd.read_csv(CSV_PATH)
    selected = df[
        (df["classification"] == "Type-I")
        & (df["config_id"] == CONFIG_TO_TEST)
        & (df["param_rank"] == 1)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one matching Type-I parameter row; "
            f"found {len(selected)}."
        )
    row = selected.iloc[0]

    baseline_params = np.array(
        [
            row["alpha_u"],
            row["beta_u"],
            row["K_uu"],
            row["K_vu"],
            row["delta_u"],
            row["alpha_v"],
            row["beta_v"],
            row["K_uv"],
            row["K_wv"],
            row["delta_v"],
            row["alpha_w"],
            row["beta_w"],
            row["K_ww"],
            row["K_uw"],
            row["K_vw"],
            row["delta_w"],
        ]
    )
    baseline_state = np.array(
        [row["u_star"], row["v_star"], row["w_star"]]
    )
    hopping = {
        "h_u": row["dU"],
        "h_v": row["dV"],
        "h_w": row["dW"],
    }

    results, baseline_info = run_cv_sweep(
        baseline_params,
        baseline_state,
        hopping,
    )

    output = {
        "results": results,
        "baseline_info": baseline_info,
        "baseline_params": baseline_params,
        "baseline_state": baseline_state,
        "hopping": hopping,
        "n_trials": N_TRIALS,
        "n_cells": N_CELLS,
        "config_id": CONFIG_TO_TEST,
        "config_name": row["config_name"],
        "noise_definition": (
            "Mean-one lognormal multiplicative noise; CV is the "
            "coefficient of variation, not a hard +/- bound."
        ),
    }

    output_file = (
        f"3954_cv_sweep_{CONFIG_LABEL}_config"
        f"{CONFIG_TO_TEST}_N{N_CELLS}_fixed.pkl"
    )
    with open(output_file, "wb") as handle:
        pickle.dump(output, handle)
    print("Saved", output_file)


if __name__ == "__main__":
    main()
