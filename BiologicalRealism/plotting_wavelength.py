import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# from heterogenous_ring_3954 import compute_jacobian

# FUNCTIONS FROM OBJECTIVE 1 (HILL FUNCTIONS, ODE SYSTEM, STEADY STATE FINDING, JACOBIAN)

n = 2

def hill_activation(X, K):
    return X**n / (K**n + X**n)

def hill_inhibition(X, K):
    return K**n / (K**n + X**n)

def dH_act(x, K):
    return n * K**n * x**(n-1) / (K**n + x**n)**2

def dH_inh(x, K):
    return -n * K**n * x**(n-1) / (K**n + x**n)**2

def ode_system(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    
    du = alpha_u + beta_u * hill_activation(u, K_uu) * hill_inhibition(v, K_vu) - delta_u * u
    dv = alpha_v + beta_v * hill_activation(u, K_uv) * hill_inhibition(w, K_wv) - delta_v * v
    dw = alpha_w + beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w * w
    
    return [du, dv, dw]

def find_steady_state(params, n_attempts=100): # was 10 
    for _ in range(n_attempts):
        initial_guess = np.random.uniform(0.01, 10.0, 3) # np.random.uniform(0.001, 50.0, 3) bc broader range
        sol = fsolve(ode_system, initial_guess, args=(params,), full_output=True)
        steady_state, info, ier, msg = sol
        residuals = ode_system(steady_state, params)
        if (ier == 1 and np.max(np.abs(residuals)) < 1e-8 and np.all(steady_state > 0)):
            return steady_state
    return None

def compute_jacobian(state, params):
    u, v, w = state
    alpha_u, beta_u, K_uu, K_vu, delta_u = params[0:5]
    alpha_v, beta_v, K_uv, K_wv, delta_v = params[5:10]
    alpha_w, beta_w, K_ww, K_uw, K_vw, delta_w = params[10:16]
    
    J = np.zeros((3, 3))
    J[0, 0] = beta_u * dH_act(u, K_uu) * hill_inhibition(v, K_vu) - delta_u
    J[0, 1] = beta_u * hill_activation(u, K_uu) * dH_inh(v, K_vu)
    J[0, 2] = 0
    J[1, 0] = beta_v * dH_act(u, K_uv) * hill_inhibition(w, K_wv)
    J[1, 1] = -delta_v
    J[1, 2] = beta_v * hill_activation(u, K_uv) * dH_inh(w, K_wv)
    J[2, 0] = beta_w * hill_activation(w, K_ww) * dH_inh(u, K_uw) * hill_inhibition(v, K_vw)
    J[2, 1] = beta_w * hill_activation(w, K_ww) * hill_inhibition(u, K_uw) * dH_inh(v, K_vw)
    J[2, 2] = beta_w * dH_act(w, K_ww) * hill_inhibition(u, K_uw) * hill_inhibition(v, K_vw) - delta_w
    
    return J

# CONFIG
CSV_PATH = '../TopologyRanking/Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv'

# Pick one config per classification — choose configs you know have good
# examples of that type. Adjust these based on your CSV's content.

CONFIG_PER_CATEGORY = {
    'Type-I':  44,
    'Type-II': 44,
    'Hopf':    49,
    'Filter':  54,
}

COLOR_PER_CATEGORY = {
    'Type-I':  'steelblue',
    'Type-II': 'mediumvioletred',
    'Hopf':    'darkorange',
    'Filter':  'seagreen',
}

N_SAMPLES_PER_TYPE = 2
SEED = 42
df = pd.read_csv(CSV_PATH)

categories = list(CONFIG_PER_CATEGORY.keys())

# samples_per_category = {}
# for category in categories:
#     config_id = CONFIG_PER_CATEGORY[category]
#     subset = df[(df['config_id'] == config_id) &
#                 (df['classification'] == category)]
    
#     if len(subset) == 0:
#         print(f"  {category} (config {config_id}): NO samples found! Skipping.")
#         samples_per_category[category] = None
#         continue
    
#     # Random sample of N_SAMPLES_PER_TYPE from this category
#     rng = np.random.default_rng(SEED + categories.index(category))  # different seed per category
#     n_to_sample = min(N_SAMPLES_PER_TYPE, len(subset))
#     sample_indices = rng.choice(len(subset), size=n_to_sample, replace=False)
#     random_samples = subset.iloc[sample_indices]
#     samples_per_category[category] = random_samples


samples_per_category = {}
K_PEAK_FILTER = 0.3
MAX_SEARCH_DEPTH = 100

k_values = np.arange(0.01, 20.01, 0.05)

def find_peak_k(row, dU, dV, dW):
    params = np.array([
        row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
        row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
        row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
    ])
    ss = np.array([row['u_star'], row['v_star'], row['w_star']])
    J = compute_jacobian(ss, params)
    D = np.diag([dU, dV, dW])
    
    max_reals = []
    for k in k_values:
        M = J - k**2 * D
        max_reals.append(np.max(np.real(np.linalg.eigvals(M))))
    
    return k_values[np.argmax(max_reals)]


for category in categories:
    config_id = CONFIG_PER_CATEGORY[category]
    subset = df[(df['config_id'] == config_id) &
                (df['classification'] == category)]
    
    if len(subset) == 0:
        samples_per_category[category] = None
        continue
    
    dU = subset['dU'].iloc[0]
    dV = subset['dV'].iloc[0]
    dW = subset['dW'].iloc[0]
    config_name = subset['config_name'].iloc[0]
    
    if category in ('Type-I', 'Type-II'):
        # Filter for samples with peak k < K_PEAK_FILTER
        sorted_subset = subset.sort_values('param_rank').head(MAX_SEARCH_DEPTH)
        selected = []
        
        for _, row in sorted_subset.iterrows():
            peak_k = find_peak_k(row, dU, dV, dW)
            if peak_k < K_PEAK_FILTER:
                selected.append(row)
                if len(selected) >= N_SAMPLES_PER_TYPE:
                    break
        
        if selected:
            samples_per_category[category] = pd.DataFrame(selected)
        else:
            samples_per_category[category] = subset.sort_values('param_rank').head(N_SAMPLES_PER_TYPE)

    else:
        # Hopf or Filter: just take top samples by rank
        samples_per_category[category] = subset.sort_values('param_rank').head(N_SAMPLES_PER_TYPE)

# PLOT

fig, axes = plt.subplots(N_SAMPLES_PER_TYPE, len(categories),figsize=(4 * len(categories), 3 * N_SAMPLES_PER_TYPE))

for col_idx, category in enumerate(categories):
    samples = samples_per_category[category]
    
    if samples is None:
        # No samples available — blank out this column
        for row_idx in range(N_SAMPLES_PER_TYPE):
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_title(f"{category}\n(no samples)",fontsize=11)
        continue
    
    config_id = CONFIG_PER_CATEGORY[category]
    dU = samples['dU'].iloc[0]
    dV = samples['dV'].iloc[0]
    dW = samples['dW'].iloc[0]
    D = np.diag([dU, dV, dW])
    
    for row_idx in range(N_SAMPLES_PER_TYPE):
        ax = axes[row_idx, col_idx]
        
        if row_idx >= len(samples):
            ax.axis('off')
            continue
        
        row = samples.iloc[row_idx]
        params = np.array([
            row['alpha_u'], row['beta_u'], row['K_uu'], row['K_vu'], row['delta_u'],
            row['alpha_v'], row['beta_v'], row['K_uv'], row['K_wv'], row['delta_v'],
            row['alpha_w'], row['beta_w'], row['K_ww'], row['K_uw'], row['K_vw'], row['delta_w']
        ])
        ss = np.array([row['u_star'], row['v_star'], row['w_star']])
        
        # Compute dispersion
        J = compute_jacobian(ss, params)
        max_reals = np.zeros(len(k_values))
        for i, k in enumerate(k_values):
            M = J - k**2 * D
            max_reals[i] = np.max(np.real(np.linalg.eigvals(M)))
        
        # Plot
        #ax.plot(k_values, max_reals, 'b-', linewidth=2) # just one colour blue
        ax.plot(k_values, max_reals, color=COLOR_PER_CATEGORY[category], linewidth=2)
        ax.axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.6)
        
        # Mark classifier boundary at k=10
        ax.axvline(x=10, color='gray', linestyle=':', linewidth=1.4,alpha=0.7)
        
        # Mark peak
        peak_idx = np.argmax(max_reals)
        peak_k = k_values[peak_idx]
        peak_val = max_reals[peak_idx]
        # ax.plot(peak_k, peak_val, 'o', color='black', markersize=8, zorder=5)
        
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Re(λ)', fontsize=10) # QUESTION: is it max Re or just Re???
        # title = (f"{category} (rank {int(row['param_rank'])})\n"f"config {config_id}, peak k={peak_k:.2f}, λ_max={peak_val:.3f}")
        title = (f"{category} (sample {row_idx + 1})\n"f"config {config_id}, peak k={peak_k:.2f}, lambda max={peak_val:.3f}")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.5)

fig.suptitle(
    'Dispersion Relations Across the Four Turing Classifications\n''Two samples per class, with classifier boundary at k=10',
    fontsize=13, y=1.00
)

plt.tight_layout()
plt.savefig('classifier_validation_final.png', dpi=150, bbox_inches='tight')
print("\nSaved: classifier_validation_final.png")
plt.close()