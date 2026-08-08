from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV     = "3954_FINAL_rmt_results_summary.csv"
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
    df = df[~df["config_name"].str.contains("Control")]
    df["topology"]    = df["config_name"].str.extract(r"NEW_RMT_3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")
    return df

#TYPE_COLORS = {"Type1": 'lightseagreen', "Type2": 'teal', "Type3": 'mediumpurple',} 
TYPE_COLORS = {"Type1": "#2E9F6E", "Type2": "#2B72DB", "Type3": "#E34D93"}


########## Robustness vs Sigma overview ##########

def complete_robustness_figure(df, sigma_val):
    import matplotlib.ticker as mticker

    random.seed(42)
    fig = plt.figure(figsize=(12.4, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax_stable = fig.add_subplot(gs[0, :])
    ax_avg = fig.add_subplot(gs[1, 0])
    ax_dot = fig.add_subplot(gs[1, 1])

    # PANEL A: SIGMA VS ROBUSTNESS OVERVIEW (TOP ROW)
    stable = df.groupby('sigma')['stable_without_diffusion'].first()
    ax_stable.plot(stable.index, stable.values, color='black', linewidth=2, linestyle='--', zorder=5,)
    ax_stable.set_xticks(range(int(stable.index.min()), int(stable.index.max()) + 1))
    ax_stable.set_xlabel('Sigma (σ)', fontsize=13)

    # Show 100,000, 200,000, ... as 1, 2, ... and put the scale in the label once.
    ax_stable.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f'{value / 1e5:g}'))
    ax_stable.set_ylabel(r'Stable steady states ($\times 10^5$)', fontsize=13)
    ax_stable.xaxis.grid(False)

    ax_rob = ax_stable.twinx()
    for cfg in df['config_name'].unique():
        subset = df[df['config_name'] == cfg].sort_values('sigma')
        t_type = subset['turing_type'].iloc[0]
        line_color = TYPE_COLORS.get(t_type, 'black')
        ax_rob.plot(subset['sigma'],subset['rob_shaberi_total'],color=line_color,linewidth=1.5,zorder=3,)

    ax_rob.set_ylabel('Robustness Score (in %)', fontsize=13)
    ax_rob.spines['right'].set_visible(True)
    ax_rob.spines['top'].set_visible(False)
    ax_rob.yaxis.grid(False)
    ax_rob.xaxis.grid(False)
    ax_stable.set_title('(A) Sigma vs Robustness and Stability Profile',fontsize=13,loc='left',pad=10,)

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

    ax_avg.set_xlabel('Sigma (σ)', fontsize=13)
    ax_avg.set_ylabel('Average Robustness Score (in %)', fontsize=13)
    ax_avg.set_xlim(0.2, 0.8)
    ax_avg.spines[['top', 'right']].set_visible(False)
    ax_avg.xaxis.grid(False)
    ax_avg.yaxis.grid(True)
    ax_avg.set_title('(B) Average Robustness by Diffusion Type (σ = 0.2 to 0.8)',fontsize=13,loc='left',pad=10,)

    # PANEL C: RAINCLOUD DOT PLOT AT FIXED SIGMA (BOTTOM RIGHT)
    subset_df = df[df['sigma'] == sigma_val].copy()
    dot_labels = ['Type 1', 'Type 2', 'Type 3']
    for i, t in enumerate(types):
        t_data = subset_df[subset_df['turing_type'] == t]['rob_shaberi_total'].dropna().values
        if len(t_data) == 0:
            continue

        color = TYPE_COLORS[t]
        kde = gaussian_kde(t_data, bw_method=0.3)
        y_range = np.linspace(t_data.min() - t_data.std() * 0.3,t_data.max() + t_data.std() * 0.3,200,)
        kde_vals = kde(y_range)
        kde_vals = kde_vals / kde_vals.max() * 0.35
        ax_dot.fill_betweenx(y_range,i - kde_vals,i,color=color,alpha=1.0,linewidth=0,zorder=2,)

        mean_val = t_data.mean()
        closest_idx = np.argmin(np.abs(y_range - mean_val))
        kde_at_mean = kde_vals[closest_idx]
        ax_dot.hlines(mean_val,i - kde_at_mean,i,color='black',linewidth=1.0,zorder=4,)

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
    ax_dot.set_xticklabels(dot_labels, fontsize=13)
    ax_dot.set_ylabel('Robustness Score (in %)', fontsize=13, labelpad=10)
    ax_dot.xaxis.grid(False)
    ax_dot.yaxis.grid(True)
    ax_dot.set_xlim(-0.5, len(types) - 0.5)
    ax_dot.spines[['top', 'right']].set_visible(False)
    ax_dot.set_title(f'(C) Robustness Distribution by Diffusion Type at σ = {sigma_val}',fontsize=13,loc='left',pad=10,)

    global_handles = [
        mlines.Line2D([], [], color='black', linewidth=2, linestyle='--',label='Stable-state count'),
        mlines.Line2D([], [], color=TYPE_COLORS['Type1'], linewidth=2.5,label='Type 1',),
        mlines.Line2D([], [], color=TYPE_COLORS['Type2'], linewidth=2.5,label='Type 2',),
        mlines.Line2D([], [], color=TYPE_COLORS['Type3'], linewidth=2.5,label='Type 3',),
        mlines.Line2D([], [], color='#313131', marker='o', linestyle='None',markersize=8, markeredgecolor='white', label='Equal Diffusion',),
        mlines.Line2D([], [], color='#313131', marker='^', linestyle='None',markersize=8, markeredgecolor='white', label='Unequal Diffusion',),]

    fig.legend(handles=global_handles, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=6, fontsize=13, handlelength=2.4, columnspacing=1.6,)
    fig.suptitle('Random Matrix Theory Results, 1 million simulations\n' 'Robustness of different diffusion rate configurations for Topology #3954', fontsize=16, y=0.98,)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.12, hspace=0.4, wspace=0.2,) # hspace is gap between Panel A and Panels B/C,  wspace is gap between Panels B and C

    # Increase all y-axis tick-label sizes
    # for ax in [ax_stable, ax_rob, ax_avg, ax_dot]:
    #     ax.tick_params(axis='y', labelsize=13)
    
    save(fig, 'final_3954_rmt_overview_sigma')


########### RUN THE WHOLE THING ############

df = load_data()
complete_robustness_figure(df, sigma_val=0.58)







########## CODE FOR SINGLE PANELS ##########

# def fig1_sigma_vs_robustness(df):
#     fig, ax_stable = plt.subplots(figsize=(12, 6))

#     stable = df.groupby("sigma")["stable_without_diffusion"].first()
#     ax_stable.plot(stable.index, stable.values, color="black", linewidth=1.5, linestyle="--", label="Stable steady state count")
#     ax_stable.set_xlabel("Sigma (σ)", fontsize=11)
#     ax_stable.set_ylabel("Number of stable steady states", fontsize=11)
#     ax_stable.xaxis.grid(False)
#     ax_rob = ax_stable.twinx()
    
#     # Track which types have been labeled in the legend to avoid duplicates
#     labeled_types = set()

#     for cfg in df["config_name"].unique():
#         subset = df[df["config_name"] == cfg].sort_values("sigma")
        
#         # Get the turing_type for this specific configuration
#         t_type = subset["turing_type"].iloc[0]
        
#         # Fallback to black if a type is missing or not in TYPE_COLORS
#         line_color = TYPE_COLORS.get(t_type, "black") 
        
#         # Format a clean label name (e.g., "Type 1 Robustness")
#         formatted_label = f"Type {t_type[-1]} Robustness" if t_type else "Unknown Type"

#         # Only apply a label if this type hasn't been added to the legend yet
#         if t_type not in labeled_types:
#             label = formatted_label
#             labeled_types.add(t_type)
#         else:
#             label = ""

#         ax_rob.plot(
#             subset["sigma"],
#             subset["rob_shaberi_total"],
#             color=line_color,
#             linewidth=1.5,
#             label=label,
#         )
        
#     ax_rob.set_ylabel("Robustness (rob_shaberi_total)", fontsize=11)
#     ax_rob.spines["right"].set_visible(True)
#     ax_rob.spines["top"].set_visible(False)
#     ax_rob.yaxis.grid(False)
#     ax_rob.xaxis.grid(False)

#     ax_stable.set_title("Topology #3954 RMT, Sigma vs Robustness and Stability", fontsize=12, loc="center", pad=10,)

#     # Combine legend handles from both axes dynamically
#     lines1, labels1 = ax_stable.get_legend_handles_labels()
#     lines2, labels2 = ax_rob.get_legend_handles_labels()
    
#     # Adjusted ncol to 4 since there are now 4 legend items total (1 stable count + 3 types)
#     ax_stable.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

#     fig.tight_layout()
#     save(fig, "test_3954_rmt_fig1_sigma_vs_robustness_all")



# def fig1_sigma_vs_robustness_average(df):
#     # Filter dataframe upfront for the sigma range 0 to 1
#     focused_df = df[(df["sigma"] >= 0.2) & (df["sigma"] <= 0.8)].copy()
    
#     fig, ax_rob = plt.subplots(figsize=(7, 5))

#     # Calculate the mean robustness score for each type at each unique sigma value
#     avg_df = (focused_df.groupby(["sigma", "turing_type"])["rob_shaberi_total"].mean().reset_index())
    
#     types = ["Type1", "Type2", "Type3"]
#     labels = ["Type 1 Average", "Type 2 Average", "Type 3 Average"]

#     for t, label in zip(types, labels):
#         # Isolate the pre-aggregated trajectory for this specific type
#         subset = avg_df[avg_df["turing_type"] == t].sort_values("sigma")
        
#         # Skip if the dataset contains no records for this category in the range
#         if subset.empty:
#             continue
            
#         color = TYPE_COLORS.get(t, "black")
        
#         # Removed marker and markersize arguments for a smooth, clean line style
#         ax_rob.plot(
#             subset["sigma"],
#             subset["rob_shaberi_total"],
#             color=color,
#             linewidth=2.5,  # Stronger line weight stands out nicely
#             label=label,
#         )
        
#     # Formatting layout elements
#     ax_rob.set_xlabel("Sigma (σ)", fontsize=11)
#     ax_rob.set_ylabel("Average Robustness (rob_shaberi_total)", fontsize=11)
#     ax_rob.set_xlim(0.2, 0.8)
    
#     ax_rob.spines[["top", "right"]].set_visible(False)
#     ax_rob.xaxis.grid(False)
#     ax_rob.yaxis.grid(True, linestyle=":", alpha=0.6)

#     ax_rob.set_title(
#         "Topology #3954 RMT – Average Robustness by Turing Type (σ: 0 to 1)",
#         fontsize=12, loc="left", pad=10,
#     )

#     # Clean centered legend
#     ax_rob.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

#     fig.tight_layout()
#     save(fig, "test_3954_rmt_fig1_sigma_vs_robustness_average")



# # DOT PLOT!!!

# def dotplot_fixed_sigma(df, sigma_val):
#     random.seed(42)
#     subset_df = df[df["sigma"] == sigma_val].copy()
#     fig, ax = plt.subplots(figsize=(6, 5))
    
#     types = ["Type1", "Type2", "Type3"]
#     labels = ["Type 1", "Type 2", "Type 3"]

#     for i, t in enumerate(types):
#         t_data = subset_df[subset_df["turing_type"] == t]["rob_shaberi_total"].dropna().values
#         color = TYPE_COLORS[t]
        
#         kde = gaussian_kde(t_data, bw_method=0.3)
#         y_range = np.linspace(t_data.min() - t_data.std()*0.3, t_data.max() + t_data.std()*0.3, 200)
#         kde_vals = kde(y_range)
#         kde_vals = kde_vals / kde_vals.max() * 0.35
#         ax.fill_betweenx(y_range, i - kde_vals, i, color=color, alpha=1.0, linewidth=0, zorder=2)

#         mean_val = t_data.mean()
#         closest_idx = np.argmin(np.abs(y_range - mean_val))
#         kde_at_mean = kde_vals[closest_idx]
#         ax.hlines(mean_val, i - kde_at_mean, i, color="black", linewidth=1.0, zorder=4)

#         for val in t_data:
#             jitter = i + random.uniform(0.08, 0.35)
            
#             marker = ("^"
#                 if any(
#                     "Unequal" in row["config_name"]
#                     for _, row in subset_df[(subset_df["turing_type"] == t) & (subset_df["rob_shaberi_total"] == val)].iterrows()
#                 )
#                 else "o")

#             ax.scatter(jitter, val, color=color, marker=marker, s=120, edgecolors="white", linewidths=0.4, zorder=3)

#     ax.set_xticks(range(len(types)))
#     ax.set_xticklabels(labels, fontsize=12)
#     ax.set_ylabel("Robustness Score (in %)", fontsize=11, labelpad=10)
#     ax.set_title(f"Topology #3954 RMT, Robustness by Turing Type at σ = {sigma_val}", fontsize=12, loc="center", pad=10)
    
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(True)
#     ax.set_xlim(-0.5, len(types) - 0.5)
#     ax.spines[["top", "right"]].set_visible(False)

#     #bar_handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
#     rain_handles = [
#         mlines.Line2D([], [], color="#313131", marker="o", linestyle="None", markersize=8, markeredgecolor="white", label="Equal Diffusion"),
#         mlines.Line2D([], [], color="#313131", marker="^", linestyle="None", markersize=8, markeredgecolor="white", label="Unequal Diffusion")]

#     # Render legends safely underneath the main figure area
#     #fig.legend(handles=bar_handles, title="Turing Type", frameon=False, loc="lower center", bbox_to_anchor=(0.35, 0.04), ncol=3, fontsize=11)
#     fig.legend(handles=rain_handles, title="Diffusion Variant", frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=11)
    
#     # Clear margins for the bottom legends
#     fig.subplots_adjust(left=0.07, right=0.95, top=0.94, bottom=0.21, hspace=0.5)
    
#     save(fig, f"3954_rmt_fig4_dotplot_sigma{sigma_val}")

