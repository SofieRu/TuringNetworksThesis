#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

# # Load results
# with open('objective2_cv_sweep_results.pkl', 'rb') as f:
#     data = pickle.load(f)

# results = data['results']

# # Extract data
# CV_values = [r['CV'] for r in results]
# mean_eigs = [r['mean_eig'] for r in results]
# std_eigs = [r['std_eig'] for r in results]
# robustness = [r['robustness'] for r in results]

# # Convert to arrays
# CV_values = np.array(CV_values)
# mean_eigs = np.array(mean_eigs)
# std_eigs = np.array(std_eigs)
# robustness = np.array(robustness)

# # ============================================================================
# # FIGURE 1: Mean Re(λ) ± std vs CV
# # ============================================================================

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot mean line
# ax.plot(CV_values, mean_eigs, 'o-', color='darkblue', linewidth=2.5, 
#         markersize=8, label='Mean Re(λ)', zorder=3)

# # Shaded region for ±1 std
# ax.fill_between(CV_values, 
#                 mean_eigs - std_eigs, 
#                 mean_eigs + std_eigs,
#                 alpha=0.3, color='blue', label='±1 std', zorder=2)

# # Horizontal line at Re(λ) = 0 (Turing threshold)
# ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
#            label='Turing threshold (Re(λ)=0)', zorder=1)

# # Labels and styling
# ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
# ax.set_ylabel('Max Re(λ)', fontsize=13)
# ax.set_title('Turing Instability vs Parameter Heterogeneity\n'
#              'Config 13: dU=1.0, dV=0.1, dW=0.0', fontsize=14, pad=15)
# ax.legend(fontsize=11, loc='upper left')
# ax.grid(True, alpha=0.3)

# # Add text annotation
# ax.text(0.3, 0.5, 
#         f'Baseline (CV=0): {mean_eigs[0]:.3f}\n'
#         f'At CV=0.4: {mean_eigs[-1]:.3f}\n'
#         f'Increase: {((mean_eigs[-1]/mean_eigs[0])-1)*100:.0f}%',
#         transform=ax.transData,
#         fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plt.tight_layout()
# plt.savefig('objective2_mean_std_vs_cv.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: objective2_mean_std_vs_cv.png")
# plt.show()

# print("\n" + "="*70)
# print("SUMMARY:")
# print(f"CV=0.00: {mean_eigs[0]:.6f} (homogeneous)")
# print(f"CV=0.40: {mean_eigs[-1]:.6f} ({((mean_eigs[-1]/mean_eigs[0])-1)*100:+.1f}% change)")
# print(f"Std at CV=0.40: {std_eigs[-1]:.6f} (huge spread!)")
# print("="*70)


# Load results
with open('objective2_cv_sweep_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']

# Extract data
CV_values = [r['CV'] for r in results]
mean_eigs = [r['mean_eig'] for r in results]
std_eigs = [r['std_eig'] for r in results]
min_eigs = [r['min_eig'] for r in results]
max_eigs = [r['max_eig'] for r in results]
robustness = [r['robustness'] for r in results]

# Convert to arrays
CV_values = np.array(CV_values)
mean_eigs = np.array(mean_eigs)
std_eigs = np.array(std_eigs)
min_eigs = np.array(min_eigs)
max_eigs = np.array(max_eigs)

# ============================================================================
# FIGURE 1: Mean Re(λ) ± std vs CV
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot mean line
ax.plot(CV_values, mean_eigs, 'o-', color='darkblue', linewidth=2.5, 
        markersize=8, label='Mean Re(λ)', zorder=3)

# Shaded region for ±1 std
ax.fill_between(CV_values, 
                mean_eigs - std_eigs, 
                mean_eigs + std_eigs,
                alpha=0.25, color='blue', label='±1 std (68% of data)', zorder=2)

# Min/max range (shows ALL data including outliers)
ax.plot(CV_values, min_eigs, ':', color='navy', linewidth=2, 
        label='Min (worst case)', alpha=0.8, zorder=2)
ax.plot(CV_values, max_eigs, ':', color='navy', linewidth=2, 
        label='Max (best case)', alpha=0.8, zorder=2)

# Horizontal line at Re(λ) = 0 (Turing threshold)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
           label='Turing threshold', zorder=4)

# Highlight where some realizations fail
failed_cv = [cv for cv, min_val in zip(CV_values, min_eigs) if min_val < 0]
if failed_cv:
    ax.axvspan(min(failed_cv), CV_values[-1], alpha=0.1, color='red', zorder=1)
    ax.text(0.35, -0.05, 'Some realizations\nlose Turing', 
            fontsize=9, color='darkred', style='italic')

# Labels and styling
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('Turing Instability vs Parameter Heterogeneity\n'
             'Config 13: dU=1.0, dV=0.1, dW=0.0', fontsize=14, pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# Add summary stats box
ax.text(0.02, 0.5, 
        f'Baseline: {mean_eigs[0]:.3f}\n'
        f'At CV=0.4: {mean_eigs[-1]:.3f}\n'
        f'Change: +{((mean_eigs[-1]/mean_eigs[0])-1)*100:.0f}%\n'
        f'Min at CV=0.4: {min_eigs[-1]:.3f}\n'
        f'11/1000 fail at CV=0.4',
        transform=ax.transData,
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('objective2_mean_std_vs_cv.png', dpi=300, bbox_inches='tight')
print("✓ Saved: objective2_mean_std_vs_cv.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY:")
for i, cv in enumerate(CV_values):
    print(f"CV={cv:.2f}: mean={mean_eigs[i]:.3f}, std={std_eigs[i]:.3f}, "
          f"range=[{min_eigs[i]:.3f}, {max_eigs[i]:.3f}], "
          f"robustness={robustness[i]:.1f}%")
print("="*70)



#############

# Extract data
CV_values = [r['CV'] for r in results]
mean_eigs = [r['mean_eig'] for r in results]
min_eigs = [r['min_eig'] for r in results]
max_eigs = [r['max_eig'] for r in results]

CV_values = np.array(CV_values)
mean_eigs = np.array(mean_eigs)
min_eigs = np.array(min_eigs)
max_eigs = np.array(max_eigs)

# ============================================================================
# SIMPLIFIED: Mean with full range
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Fill FULL range (min to max)
ax.fill_between(CV_values, min_eigs, max_eigs,
                alpha=0.2, color='purple', 
                label='Full range (min-max)', zorder=1)

# Plot mean line
ax.plot(CV_values, mean_eigs, 'o-', color='darkblue', linewidth=2.5, 
        markersize=8, label='Mean Re(λ)', zorder=3)

# Horizontal line at Re(λ) = 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
           label='Turing threshold', zorder=2)

# Labels
ax.set_xlabel('CV (Coefficient of Variation)', fontsize=13)
ax.set_ylabel('Max Re(λ)', fontsize=13)
ax.set_title('Turing Growth Rate vs Parameter Heterogeneity\n'
             'Config 13: dU=1.0, dV=0.1, dW=0.0', fontsize=13, pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simpler_objective2_mean_range_vs_cv.png', dpi=300, bbox_inches='tight')
print("✓ Saved: objective2_mean_range_vs_cv.png")
plt.show()


#############

# ============================================================================
# BOXPLOT: Distribution of Re(λ) at each CV
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
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

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
plt.savefig('objective2_boxplot_cv_sweep.png', dpi=300, bbox_inches='tight')
print("✓ Saved: objective2_boxplot_cv_sweep.png")
plt.show()

# Print summary
print("\n" + "="*70)
print("SUMMARY:")
for i, cv in enumerate(CV_values):
    data_cv = all_eigenvalues[i]
    print(f"CV={cv:.2f}: median={np.median(data_cv):.3f}, "
          f"range=[{np.min(data_cv):.3f}, {np.max(data_cv):.3f}], "
          f"robustness={robustness[i]:.1f}%")
print("="*70)