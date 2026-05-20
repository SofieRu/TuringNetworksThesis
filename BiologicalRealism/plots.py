#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

# Load results
with open('objective2_cv_sweep_config13.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']

# Extract data
CV_values = [r['CV'] for r in results]
mean_eigs = [r['mean_eig'] for r in results]
min_eigs = [r['min_eig'] for r in results]
max_eigs = [r['max_eig'] for r in results]
std_eigs = [r['std_eig'] for r in results]

CV_values = np.array(CV_values)
mean_eigs = np.array(mean_eigs)
min_eigs = np.array(min_eigs)
max_eigs = np.array(max_eigs)
std_eigs = np.array(std_eigs)

# ============================================================================
# FIGURE 1: SIMPLIFIED: Mean with full range
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(CV_values, min_eigs, max_eigs, alpha=0.2, color='blue', label='Full range (min-max)', zorder=1)

# Plot mean line
ax.plot(CV_values, mean_eigs, 'o-', color='darkblue', linewidth=2.5, 
        markersize=8, label='Mean Re(λ)', zorder=3)

# Horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Turing threshold', zorder=2)

# Labels
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('Turing Growth Rate vs Parameter Heterogeneity\n'
             'Config 13: dU=1.0, dV=0.1, dW=0.0', fontsize=13, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("Saved: fig1_mean_range_vs_cv.png")
plt.show()


#############

# ============================================================================
# FIGURE 2: BOXPLOT: Distribution of Re(λ) at each CV
# ============================================================================

# Extract data
CV_values = [r['CV'] for r in results]
all_eigenvalues = [r['all_eigenvalues'] for r in results]
robustness = [r['robustness'] for r in results]

fig, ax = plt.subplots(figsize=(12, 6))

# Create boxplot
bp = ax.boxplot(all_eigenvalues, 
                positions=range(len(CV_values)),
                widths=0.6,
                patch_artist=True,
                showfliers=True,  # Show outliers
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

# Color boxes
for patch in bp['boxes']:
    patch.set_facecolor('purple')
    patch.set_alpha(0.3)

# Horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
           label='Turing threshold (Re(λ)=0)', zorder=10)

# X-axis labels
ax.set_xticks(range(len(CV_values)))
ax.set_xticklabels([f'{cv:.2f}' for cv in CV_values])
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('Distribution of Turing Growth Rates Under Parameter Heterogeneity\n'
             'Config 13: dU=1.0, dV=0.1, dW=0.0 (1000 trials per CV)', 
             fontsize=13, pad=15)

ax.legend(fontsize=9, framealpha=0.9)  # Was fontsize=11
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig2_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: fig2_boxplot_cv_sweep.png")

################



# ============================================================================
# FIGURE 3: MEAN VS STD PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Plot data points
ax.plot(mean_eigs, std_eigs, 'o-', markersize=10, linewidth=2, 
        color='darkblue', label='Observed')

# Reference line: std = mean (Poisson-like)
max_val = max(mean_eigs.max(), std_eigs.max())
ax.plot([0, max_val], [0, max_val], '--', color='red', linewidth=2,
        label='Std = Mean (Poisson)', alpha=0.7)

# Annotate each point with CV value
for i, cv in enumerate(CV_values):
    ax.annotate(f'CV={cv:.2f}', 
                xy=(mean_eigs[i], std_eigs[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.7)

# Labels
ax.set_xlabel('Mean Re(λ)', fontsize=13)
ax.set_ylabel('Std Re(λ)', fontsize=13)
ax.set_title('Mean-Variance Relationship\n'
             'Config 13: dU=1.0, dV=0.1, dW=0.0', fontsize=13, pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')  # Makes it easier to see if slope = 1

# Compute slope (fit line through origin)
slope = np.sum(mean_eigs * std_eigs) / np.sum(mean_eigs**2)
ax.plot([0, max_val], [0, slope*max_val], ':', color='green', linewidth=2,
        label=f'Best fit: Std = {slope:.2f} × Mean', alpha=0.7)

# Update legend
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('fig3_mean_vs_std.png', dpi=300, bbox_inches='tight')
print("Saved: fig3_mean_vs_std.png")
plt.show()

# Print relationship
# print("\n" + "="*70)
# print("MEAN-VARIANCE RELATIONSHIP:")
# print("="*70)
# for i, cv in enumerate(CV_values):
#     ratio = std_eigs[i] / mean_eigs[i] if mean_eigs[i] > 0 else 0
#     print(f"CV={cv:.2f}: Mean={mean_eigs[i]:.4f}, Std={std_eigs[i]:.4f}, "
#           f"Ratio={ratio:.4f}")