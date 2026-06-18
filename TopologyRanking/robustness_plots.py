"""
Plot per-config robustness distributions for both topologies.
Shows mean, median, max, and full distribution — not just summary statistics.

Two side-by-side panels:
  Left:  Genuine Turing robustness (Type-I + Type-II + Hopf)
  Right: Type-I robustness only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

# ============================================================================
# CONFIG
# ============================================================================

SUMMARY_CSVS = {
    3954: 'Topology3954/3954_NEWTURINGCLASS_lhs_results_summary.csv',
    1754: 'Topology1754/1754_NEWTURINGCLASS_lhs_results_summary.csv',
}

TOPOLOGY_COLORS = {3954: 'darkblue', 1754: 'crimson'}


# ============================================================================
# LOAD AND COMPUTE PER-CONFIG ROBUSTNESS
# ============================================================================

data = {}

for topo_id, csv_path in SUMMARY_CSVS.items():
    df = pd.read_csv(csv_path)
    
    # Type-I robustness per config (already a column)
    type_I_rob = df['rob_shaberi_type_I'].values * 100  # convert to %
    
    # Genuine Turing robustness per config: (Type-I + Type-II + Hopf) / n_samples
    genuine_rob = ((df['shaberi_type_I'] + df['shaberi_type_II'] + df['shaberi_hopf']) 
                   / df['n_samples']).values * 100
    
    data[topo_id] = {
        'type_I': type_I_rob,
        'genuine': genuine_rob,
        'n_configs': len(df),
    }
    
    print(f"\nTopology #{topo_id}: {len(df)} configurations")
    print(f"  Type-I robustness  — mean={np.mean(type_I_rob):.3f}%, "
          f"median={np.median(type_I_rob):.3f}%, max={np.max(type_I_rob):.3f}%")
    print(f"  Genuine Turing rob — mean={np.mean(genuine_rob):.3f}%, "
          f"median={np.median(genuine_rob):.3f}%, max={np.max(genuine_rob):.3f}%")


# ============================================================================
# PLOTTING HELPER
# ============================================================================

def plot_distribution_panel(ax, values_per_topo, title, ylabel):
    """Plot a boxplot + scatter for two topologies side by side."""
    
    topo_ids = list(values_per_topo.keys())
    positions = [1, 2]
    
    # Boxplot
    bp = ax.boxplot(
        [values_per_topo[t] for t in topo_ids],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='^', markerfacecolor='gold',
                       markeredgecolor='black', markersize=10),
        medianprops=dict(color='black', linewidth=2),
        showfliers=False,  # we'll add scatter for individual points
    )
    
    # Color each box by topology
    for patch, topo_id in zip(bp['boxes'], topo_ids):
        patch.set_facecolor(TOPOLOGY_COLORS[topo_id])
        patch.set_alpha(0.4)
    
    # Add scatter points for each config (so you see the full distribution)
    np.random.seed(42)
    for pos, topo_id in zip(positions, topo_ids):
        values = values_per_topo[topo_id]
        x_jitter = np.random.normal(pos, 0.06, size=len(values))
        ax.scatter(x_jitter, values, alpha=0.6, s=30,
                   color=TOPOLOGY_COLORS[topo_id], edgecolors='black', linewidths=0.5,
                   zorder=3)
    
    # Annotate stats above each box
    for pos, topo_id in zip(positions, topo_ids):
        values = values_per_topo[topo_id]
        text = (f"max = {np.max(values):.2f}%\n"
                f"mean = {np.mean(values):.3f}%\n"
                f"median = {np.median(values):.3f}%\n"
                f"n configs = {len(values)}")
        # Position text in the upper area but slightly inside the plot
        ymin, ymax = ax.get_ylim() if ax.get_ylim()[1] > 0 else (0, np.max(values) * 1.2)
        # We'll set ylim explicitly later, so just place at relative top for now
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'#{t}' for t in topo_ids], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(alpha=0.3, axis='y')
    # Force y-axis to start at 0 — robustness is a percentage, can't be negative
    ymax = max(np.max(values_per_topo[t]) for t in topo_ids)
    ax.set_ylim(0, ymax * 1.15)  # tiny padding below 0 so the bottom whisker doesn't get clipped



# ============================================================================
# BUILD THE FIGURE
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Genuine Turing robustness
plot_distribution_panel(
    axes[0],
    {t: data[t]['genuine'] for t in data.keys()},
    title='Genuine Turing Robustness Distribution\n(Type-I + Type-II + Hopf, per config)',
    ylabel='Per-config robustness (%)',
)

# Right: Type-I only
plot_distribution_panel(
    axes[1],
    {t: data[t]['type_I'] for t in data.keys()},
    title='Type-I Robustness Distribution\n(per config)',
    ylabel='Per-config robustness (%)',
)

# Add stats annotations to each panel
for ax, values_per_topo in zip(axes,
                                [{t: data[t]['genuine'] for t in data},
                                 {t: data[t]['type_I'] for t in data}]):
    topo_ids = list(values_per_topo.keys())
    for pos, topo_id in zip([1, 2], topo_ids):
        values = values_per_topo[topo_id]
        text = (f"max: {np.max(values):.2f}%\n"
                f"mean: {np.mean(values):.3f}%\n"
                f"median: {np.median(values):.3f}%\n"
                f"n: {len(values)}")
        ax.text(pos, ax.get_ylim()[1] * 0.85, text,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.9))

# Shared legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gold',
           markeredgecolor='black', markersize=10, label='Mean'),
    Line2D([0], [0], color='black', linewidth=2, label='Median'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=7, label='Individual configs'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Per-Configuration Robustness Distributions: #3954 vs #1754',
             fontsize=14, y=1.00)

plt.tight_layout(rect=[0, 0.08, 1, 0.97])
plt.savefig('robustness_distribution.png', dpi=200, bbox_inches='tight')
print("\nSaved: robustness_distribution.png")
plt.close()