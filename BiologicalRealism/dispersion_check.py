import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heterogenous_ring_3954 import compute_jacobian

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

# ============================================================================
# CONFIG: change these to validate different configs
# ============================================================================

CSV_PATH = '../TopologyRanking/Topology3954/3954_ALLPARAMSNEW_lhs_results_parameters.csv'
CONFIG_ID = 44           # 49 for all, The config to validate (54 = VWFreeze, only U diffuses)
N_SAMPLES_PER_TYPE = 4   # How many samples per classification to plot
K_MAX = 30               # How far to sweep k (wide enough to see asymptote)
K_STEP = 0.01            # Resolution of k sweep
SEED = 42                # For reproducible sample selection

# ============================================================================
# LOAD AND FILTER
# ============================================================================

df_file = pd.read_csv('../TopologyRanking/Topology3954/3954_ALLPARAMSNEW_lhs_results_parameters.csv')
print(f"CSV columns: {df_file.columns.tolist()}")
print(f"Total rows: {len(df_file)}")
print(f"First row: {df_file.iloc[0]}")

print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

df_config = df[df['config_id'] == CONFIG_ID]
if len(df_config) == 0:
    print(f"ERROR: No rows found for config_id={CONFIG_ID}")
    raise SystemExit

# Get diffusion rates (same for all rows of this config)
dU = df_config['dU'].iloc[0]
dV = df_config['dV'].iloc[0]
dW = df_config['dW'].iloc[0]
config_name = df_config['config_name'].iloc[0]

print(f"\nConfig {CONFIG_ID}: {config_name}")
print(f"  Diffusion: dU={dU}, dV={dV}, dW={dW}")

# Show classification breakdown
print("\nClassification breakdown for this config:")
print(df_config['classification'].value_counts())

# Categories to plot (only include ones that have samples)
all_categories = ['Type-I', 'Type-II', 'Hopf','Filter']
available_categories = [c for c in all_categories 
                        if len(df_config[df_config['classification'] == c]) > 0]
print(f"\nWill plot: {available_categories}")


# ============================================================================
# PLOT GRID
# ============================================================================

D = np.diag([dU, dV, dW])
k_values = np.arange(0.01, K_MAX + K_STEP, K_STEP)

n_rows = len(available_categories)
n_cols = N_SAMPLES_PER_TYPE
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

# Make axes 2D even if n_rows == 1
if n_rows == 1:
    axes = axes.reshape(1, -1)

rng = np.random.default_rng(SEED)

for row_idx, category in enumerate(available_categories):
    subset = df_config[df_config['classification'] == category]
    
    # Random sample of N_SAMPLES_PER_TYPE from this category
    n_available = len(subset)
    n_to_plot = min(N_SAMPLES_PER_TYPE, n_available)
    sample_indices = rng.choice(n_available, size=n_to_plot, replace=False)
    samples = subset.iloc[sample_indices]
    
    print(f"\n{category}: plotting {n_to_plot} of {n_available} samples")
    
    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]
        
        if col_idx >= n_to_plot:
            ax.axis('off')
            continue
        
        row = samples.iloc[col_idx]
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
            eigs = np.linalg.eigvals(M)
            max_reals[i] = np.max(np.real(eigs))
        
        # Plot
        ax.plot(k_values, max_reals, 'b-', linewidth=2)
        ax.axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Mark the peak
        peak_idx = np.argmax(max_reals)
        peak_k = k_values[peak_idx]
        peak_val = max_reals[peak_idx]
        ax.plot(peak_k, peak_val, 'o', color='darkred', markersize=8, zorder=5)
        
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('max Re(λ)', fontsize=10)
        ax.set_title(f"{category}\npeak at k={peak_k:.2f}, λ_max={peak_val:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)

fig.suptitle(f'Classifier validation: config {CONFIG_ID} ({config_name})\n'
             f'Diffusion: dU={dU}, dV={dV}, dW={dW}',
             fontsize=13, y=1.00)

plt.tight_layout()
plt.savefig(f'classifier_validation_config{CONFIG_ID}.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: classifier_validation_config{CONFIG_ID}.png")
plt.close()