import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Import only the functions we need (no side effects from CV sweep etc.)
from heterogenous_ring_3954 import (
    ode_system,
    compute_jacobian,
    find_steady_state,
    build_ring_jacobian_heterogeneous,
)

# Two plots:
#   Plot A: Basic dispersion relation λ(k) for both configs, with discrete k_m overlay
#   Plot B: Heterogeneous dispersion — how the peak shifts under parameter noise

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'

def load_config(config_id):
    df = pd.read_csv(CSV_PATH)
    row = df[(df['config_id'] == config_id) & (df['param_rank'] == 1)].iloc[0]
    params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    hopping = {'h_u': row['dU'], 'h_v': row['dV'], 'h_w': row['dW']}
    return params, ss, hopping

print("Loading configs...")
params_13, ss_13, hopping_13 = load_config(45) #rename so its jsut robust and fragile instead of the numbers itself...
params_2,  ss_2,  hopping_2  = load_config(4)

def compute_dispersion(local_jacobian, hopping_dict, k_values):
    """Compute max Re(λ) of [J - k²·D] for each k in k_values."""
    D = np.diag([hopping_dict['h_u'], hopping_dict['h_v'], hopping_dict['h_w']])
    max_reals = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        M = local_jacobian - (k**2) * D
        eigs = np.linalg.eigvals(M)
        max_reals[i] = np.max(np.real(eigs))
    return max_reals


def discrete_k_values(N):
    """Allowed wavenumbers k_m = 2πm/N for a periodic ring of N cells."""
    return np.array([2 * np.pi * m / N for m in range(N // 2 + 1)])


# ============================================================================
# PLOT A: BASIC DISPERSION WITH DISCRETE-K OVERLAY
# ============================================================================

def plot_dispersion_basic(ax, params, ss, hopping, title, max_growth_rate=None):
    """One panel of the basic dispersion figure."""
    J = compute_jacobian(ss, params)

    # Continuous dispersion curve
    k_continuous = np.arange(0.01, 5.0, 0.01)
    disp_continuous = compute_dispersion(J, hopping, k_continuous)

    # Discrete k_m markers
    k_N10 = discrete_k_values(10)
    k_N20 = discrete_k_values(20)
    disp_N10 = compute_dispersion(J, hopping, k_N10)
    disp_N20 = compute_dispersion(J, hopping, k_N20)

    # Plot continuous curve
    ax.plot(k_continuous, disp_continuous, '-', color='black', linewidth=2.5,
            label='Continuous dispersion λ(k)', zorder=3)

    # Turing threshold
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Turing threshold (Re(λ)=0)', zorder=2)

    # Discrete markers
    ax.scatter(k_N10, disp_N10, s=130, color='darkblue', marker='o',
               edgecolors='black', linewidths=1, zorder=5,
               label='Allowed k_m at N=10')
    ax.scatter(k_N20, disp_N20, s=80, facecolors='white',
               edgecolors='darkblue', linewidths=1.5, marker='s', zorder=4,
               label='Allowed k_m at N=20')

    # Continuous-tissue peak reference line
    if max_growth_rate is not None:
        ax.axhline(y=max_growth_rate, color='green', linestyle=':',
                   linewidth=1.2, alpha=0.6,
                   label=f'Continuous-tissue max (Obj 1)')

    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Max Re(λ)', fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)


print("\n" + "="*70)
print("PLOT A: BASIC DISPERSION WITH DISCRETE-K OVERLAY")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

plot_dispersion_basic(ax1, params_13, ss_13, hopping_13,
                      'Config 13 (robust): peak aligns with discrete k_m',
                      max_growth_rate=0.143)

plot_dispersion_basic(ax2, params_2, ss_2, hopping_2,
                      'Config 2 (fragile): peak falls between discrete k_m',
                      max_growth_rate=0.173)

fig.suptitle('Dispersion Relation with Discrete Wavenumber Sampling',
             fontsize=14, y=1.00)

plt.tight_layout()
plt.savefig('fig_dispersion_basic.png', dpi=300, bbox_inches='tight')
print("Saved: fig_dispersion_basic.png")
plt.close()


# ============================================================================
# PLOT B: HETEROGENEOUS DISPERSION — HOW THE PEAK SHIFTS UNDER NOISE
# ============================================================================
#
# For each Monte Carlo trial at a given CV:
#   1. Generate perturbed parameters for N cells
#   2. Find each cell's steady state independently
#   3. Build the 30×30 ring Jacobian
#   4. Find the eigenvector of the most unstable mode
#   5. Take its spatial FFT to identify dominant wavenumber k*
#   6. Record (k*, max Re(λ)) for this trial
#
# Then plot the homogeneous dispersion curve with overlaid scatter of
# (k*, λ_max) points from heterogeneous trials.

N_CELLS = 10
N_TRIALS = 200  # fewer than CV sweep — we just need the distribution shape


def get_heterogeneous_dispersion_peaks(baseline_params, baseline_ss,
                                       hopping_dict, CV, n_trials=N_TRIALS,
                                       seed=42):
    """For n_trials heterogeneous realisations, extract (k_peak, max_re_lambda)
    using the existing build_ring_jacobian_heterogeneous from homogenous_ring.
    
    Note: baseline_ss is not used here (the function finds steady states 
    internally), but we keep it in the signature for consistency.
    """
    np.random.seed(seed)
    
    # Discrete k_m for FFT bin mapping
    k_m = discrete_k_values(N_CELLS)
    
    k_peaks = []
    lambda_peaks = []
    discarded = 0
    
    for trial in range(n_trials):
        # Build heterogeneous ring (function handles noise + steady states internally)
        J_ring, steady_states, params_list = build_ring_jacobian_heterogeneous(
            N_CELLS, baseline_params, hopping_dict, CV
        )
        
        # Function returns (None, None, None) if any cell failed steady-state
        if J_ring is None:
            discarded += 1
            continue
        
        # Eigendecomposition of the 30×30 ring Jacobian
        eigvals, eigvecs = np.linalg.eig(J_ring)
        real_parts = np.real(eigvals)
        max_idx = np.argmax(real_parts)
        max_real = real_parts[max_idx]
        
        # Extract dominant wavenumber via spatial FFT of u-component
        v_max = np.real(eigvecs[:, max_idx])
        v_reshaped = v_max.reshape(N_CELLS, 3)
        u_spatial = v_reshaped[:, 0]
        fft_amps = np.abs(np.fft.fft(u_spatial))[:N_CELLS // 2 + 1]
        dominant_bin = np.argmax(fft_amps[1:]) + 1  # skip k=0 (uniform mode)
        k_dominant = k_m[dominant_bin]
        
        k_peaks.append(k_dominant)
        lambda_peaks.append(max_real)
    
    return np.array(k_peaks), np.array(lambda_peaks), discarded



def plot_dispersion_heterogeneous(ax, params, ss, hopping, title,
                                   CV_values=(0.0, 0.10, 0.20)):
    """One panel of the heterogeneous dispersion figure."""
    J = compute_jacobian(ss, params)

    # Homogeneous continuous dispersion curve as reference
    k_continuous = np.arange(0.01, 5.0, 0.01)
    disp_continuous = compute_dispersion(J, hopping, k_continuous)

    ax.plot(k_continuous, disp_continuous, '-', color='black', linewidth=2.5,
            label='Continuous λ(k) (homogeneous baseline)', zorder=3)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Turing threshold', zorder=2)

    # Overlay scatter of (k_peak, λ_peak) from heterogeneous trials
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(CV_values)))
    markers = ['o', 's', '^', 'D']

    for i, CV in enumerate(CV_values):
        if CV == 0.0:
            # Homogeneous case: just one point at the deterministic peak
            disp_at_k_m = compute_dispersion(J, hopping, discrete_k_values(N_CELLS))
            best_idx = np.argmax(disp_at_k_m)
            ax.scatter([discrete_k_values(N_CELLS)[best_idx]], [disp_at_k_m[best_idx]],
                       s=200, color=colors[i], marker='*', edgecolors='black',
                       linewidths=1.5, zorder=10,
                       label=f'CV=0.00 (homogeneous, N={N_CELLS})')
            continue

        print(f"  Sampling CV={CV}...")
        k_pk, l_pk, n_disc = get_heterogeneous_dispersion_peaks(
            params, ss, hopping, CV
        )
        print(f"    {len(k_pk)} valid trials, {n_disc} discarded")

        # Scatter with low alpha to show density
        ax.scatter(k_pk, l_pk, s=40, color=colors[i],
                   alpha=0.4, marker=markers[i % len(markers)],
                   edgecolors='none', zorder=5,
                   label=f'CV={CV:.2f}  ({len(k_pk)} trials)')

    ax.set_xlabel('Dominant wavenumber k* of unstable mode', fontsize=12)
    ax.set_ylabel('Max Re(λ) of ring', fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)


print("\n" + "="*70)
print("PLOT B: HETEROGENEOUS DISPERSION (peak shift under noise)")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

print("\nConfig 13 (robust):")
plot_dispersion_heterogeneous(ax1, params_13, ss_13, hopping_13,
                              'Config 13 (robust): peak distribution under noise',
                              CV_values=(0.0, 0.10, 0.20))

print("\nConfig 2 (fragile):")
plot_dispersion_heterogeneous(ax2, params_2, ss_2, hopping_2,
                              'Config 2 (fragile): peak distribution under noise',
                              CV_values=(0.0, 0.05, 0.10))

fig.suptitle('Heterogeneous Dispersion: How Parameter Noise Shifts the Peak',
             fontsize=14, y=1.00)

plt.tight_layout()
plt.savefig('fig_dispersion_heterogeneous.png', dpi=300, bbox_inches='tight')
print("\nSaved: fig_dispersion_heterogeneous.png")
plt.close()

print("\n" + "="*70)
print("DONE")
print("="*70)