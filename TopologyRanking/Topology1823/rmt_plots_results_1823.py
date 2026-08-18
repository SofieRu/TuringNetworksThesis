from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV     = "1823_FINAL_rmt_results_summary.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 10,
    "axes.edgecolor"   : "#444444",
    "axes.linewidth"   : 0.8,
    "grid.color"       : "#dddddd",
    "grid.linewidth"   : 0.7,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.png")

def load_data():
    df = pd.read_csv(CSV)
    #df = df[~df["config_name"].str.contains("OneFast|Control|Limit")]
    df["topology"]    = df["config_name"].str.extract(r"RMT_1823_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")
    return df

#TYPE_COLORS = {"Type1": 'lightseagreen', "Type2": 'teal', "Type3": 'mediumpurple',} 
TYPE_COLORS = {"Type1": "#2E9F6E", "Type2": "#2B72DB", "Type3": "#E34D93"}


def complete_robustness_figure(df, sigma_val):

    random.seed(42)
    fig = plt.figure(figsize=(12.4, 7.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax_stable = fig.add_subplot(gs[0, :])
    ax_avg = fig.add_subplot(gs[1, 0])
    ax_dot = fig.add_subplot(gs[1, 1])

    # PANEL A: SIGMA VS ROBUSTNESS OVERVIEW (TOP ROW)
    stable = df.groupby('sigma')['stable_without_diffusion'].first()
    ax_stable.plot(stable.index, stable.values, color='black', linewidth=2, linestyle='--', zorder=5,)
    ax_stable.set_xticks(range(int(stable.index.min()), int(stable.index.max()) + 1))
    ax_stable.set_xlabel('Sigma (σ)', fontsize=15)
    ax_stable.tick_params(axis='x', labelsize=14)

    # Show 100,000, 200,000, ... as 1, 2, ... and put the scale in the label once.
    ax_stable.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f'{value / 1e5:g}'))
    ax_stable.set_ylabel(r'Stable steady states ($\times 10^5$)', fontsize=14.5)
    ax_stable.tick_params(axis='y', labelsize=14)
    ax_stable.xaxis.grid(False)

    ax_rob = ax_stable.twinx()
    for cfg in df['config_name'].unique():
        subset = df[df['config_name'] == cfg].sort_values('sigma')
        t_type = subset['turing_type'].iloc[0]
        line_color = TYPE_COLORS.get(t_type, 'black')
        ax_rob.plot(subset['sigma'],subset['rob_shaberi_total'],color=line_color,linewidth=1.5,zorder=3,)

    ax_rob.set_ylabel('Robustness Score (in %)', fontsize=15)
    ax_rob.tick_params(axis='y', labelsize=14)
    ax_rob.spines['right'].set_visible(True)
    ax_rob.spines['top'].set_visible(False)
    ax_rob.yaxis.grid(False)
    ax_rob.xaxis.grid(False)
    ax_stable.set_title('(A) Sigma vs Robustness and Stability Profile',fontsize=16,loc='left',pad=10,)

    # PANEL B: FOCUSED AVERAGE ROBUSTNESS LINE (BOTTOM LEFT)
    focused_df = df[(df['sigma'] >= 0.2) & (df['sigma'] <= 0.8)].copy()
    avg_df = (focused_df.groupby(['sigma', 'turing_type'])['rob_shaberi_total'].mean().reset_index())
    types = ['Type1', 'Type2', 'Type3']
    labels = ['Type 1 Average', 'Type 2 Average', 'Type 3 Average']
    for t, label in zip(types, labels):
        subset = avg_df[avg_df['turing_type'] == t].sort_values('sigma')
        if subset.empty:
            continue
        color = TYPE_COLORS.get(t, 'black')
        ax_avg.plot(subset['sigma'],subset['rob_shaberi_total'],color=color,linewidth=2.5,)

    ax_avg.set_xlabel('Sigma (σ)', fontsize=15)
    ax_avg.tick_params(axis='x', labelsize=14)
    ax_avg.set_ylabel('Robustness Score (in %)', fontsize=15)
    ax_avg.tick_params(axis='y', labelsize=14)
    ax_avg.set_xlim(0.2, 0.8)
    ax_avg.spines[['top', 'right']].set_visible(False)
    ax_avg.xaxis.grid(False)
    ax_avg.yaxis.grid(True)
    ax_avg.set_title('(B) Average Robustness by Type (σ = 0.2-0.8)',fontsize=16,loc='left',pad=10,)

    # PANEL C: RAINCLOUD DOT PLOT AT FIXED SIGMA (BOTTOM RIGHT)
    subset_df = df[df['sigma'] == sigma_val].copy()
    dot_labels = ['Type 1', 'Type 2', 'Type 3']
    for i, t in enumerate(types):
        t_data = subset_df[subset_df['turing_type'] == t]['rob_shaberi_total'].dropna().values
        if len(t_data) == 0:
            continue

        color = TYPE_COLORS[t]
        # kde = gaussian_kde(t_data, bw_method=0.3)
        # y_range = np.linspace(t_data.min() - t_data.std() * 0.3,t_data.max() + t_data.std() * 0.3,200,)
        # kde_vals = kde(y_range)
        # kde_vals = kde_vals / kde_vals.max() * 0.35
        # ax_dot.fill_betweenx(y_range,i - kde_vals,i,color=color,alpha=1.0,linewidth=0,zorder=2,)

        # mean_val = t_data.mean()
        # closest_idx = np.argmin(np.abs(y_range - mean_val))
        # kde_at_mean = kde_vals[closest_idx]
        # ax_dot.hlines(mean_val,i - kde_at_mean,i,color='black',linewidth=1.0,zorder=4,)

        for val in t_data:
            matching_rows = subset_df[(subset_df['turing_type'] == t) & (subset_df['rob_shaberi_total'] == val)]
            marker = ('^'
                if any(
                    'Unequal' in row['config_name']
                    for _, row in matching_rows.iterrows()
                )
                else 'o'
            )
            jitter = i + random.uniform(0.08, 0.35)
            ax_dot.scatter(jitter,val,color=color,marker=marker,s=120,edgecolors='white',linewidths=0.4,zorder=3,)

    ax_dot.set_xticks(range(len(types)))
    ax_dot.set_xticklabels(dot_labels, fontsize=14)
    ax_dot.set_ylabel('Robustness Score (in %)', fontsize=15, labelpad=10)
    ax_dot.tick_params(axis='y', labelsize=14)
    ax_dot.xaxis.grid(False)
    ax_dot.yaxis.grid(True)
    ax_dot.set_xlim(-0.5, len(types) - 0.5)
    ax_dot.spines[['top', 'right']].set_visible(False)
    ax_dot.set_title(f'(C) Robustness Distribution at σ = {sigma_val}',fontsize=16,loc='left',pad=10,)

    global_handles = [
        mlines.Line2D([], [], color='black', linewidth=2, linestyle='--',label='Stable-state count'),
        mlines.Line2D([], [], color=TYPE_COLORS['Type1'], linewidth=2.5,label='Type 1',),
        mlines.Line2D([], [], color=TYPE_COLORS['Type2'], linewidth=2.5,label='Type 2',),
        mlines.Line2D([], [], color=TYPE_COLORS['Type3'], linewidth=2.5,label='Type 3',),
        mlines.Line2D([], [], color='#313131', marker='o', linestyle='None',markersize=8, markeredgecolor='white', label='Equal Diffusion',),
        mlines.Line2D([], [], color='#313131', marker='^', linestyle='None',markersize=8, markeredgecolor='white', label='Unequal Diffusion',),]

    fig.legend(handles=global_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=6, fontsize=14.5, handlelength=1.5, columnspacing=1.4,)
    fig.suptitle('Random Matrix Theory Results, $5 \\times 10^5$ simulations\n' 'Robustness of different diffusion rate configurations for Topology #1823', fontsize=18, y=1.0,)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.12, hspace=0.45, wspace=0.32,) # hspace is gap between Panel A and Panels B/C,  wspace is gap between Panels B and C
    save(fig, 'final_1823_rmt_overview_sigma')



########### RUN THE WHOLE THING ############

df = load_data()
complete_robustness_figure(df, sigma_val=0.58)