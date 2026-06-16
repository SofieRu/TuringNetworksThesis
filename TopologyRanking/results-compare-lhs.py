from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

CSVS = {
    "#1754": "Topology1754/1754_NEWTURINGCLASS_lhs_results_summary.csv",
    "#1823": "Topology1823/1823_PREFINAL_lhs_results_summary.csv",
    "#1838": "Topology1838/1838_PREFINAL_lhs_results_summary.csv",
    "#3954": "Topology3954/3954_NEWTURINGCLASS_lhs_results_summary.csv",
}

PARAMS_CSV = "Topology3954/3954_NEWTURINGCLASS_lhs_results_parameters.csv"

# for both 3954 and 1754 i got rid of _CCD_Type1_Var1/2 bc the values were really high and kinda did not match the rest but later if it does match then we can put it back in and see if it changes the results

OUT_DIR = Path("ResultPlots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

TYPE_COLORS = {
    "Type1": "#444EA6",
    "Type2": "#AE2BA1",
    "Type3": "#3FA051",
} 

TOPO_MARKERS = {
    "#1754": "o",
    "#1823": "s",
    "#1838": "^",
    "#3954": "D",
}

PATTERN_COLORS = {
    "Type I"  : 'steelblue',
    "Type II" : 'mediumvioletred',
    "Hopf"    : 'darkorange',
    "Turing Filter" : 'seagreen',
}

def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/{name}.png")

def load_all():
    dfs = []
    for topo_id, path in CSVS.items():
        df = pd.read_csv(path)
        df = df[~df["config_name"].str.contains("OneFast|Control|Limit")]
        df["topology_id"]  = topo_id
        df["turing_type"]  = df["config_name"].str.extract(r"(Type[123])")
        #parsed = df["diffusion"].apply(parse_diff)
        #df["dU"] = parsed.apply(lambda d: d.get("dU"))
        #df["dV"] = parsed.apply(lambda d: d.get("dV"))
        #df["dW"] = parsed.apply(lambda d: d.get("dW"))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def parse_diff(s):
    try:
        return ast.literal_eval(s.replace("\u2018", "'").replace("\u2019", "'"))
    except:
        return {"dU": None, "dV": None, "dW": None}






########## FIGURE 1: Heatmap – Max robustness per topology × Turing Type ##########

def fig1_heatmap(df):
    pivot = (
        df.groupby(["topology_id", "turing_type"])["rob_shaberi_total"]
        .max()
        .unstack("topology_id")
        .reindex(index=["Type1", "Type2", "Type3"])
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Max robustness (rob_shaberi_total)"},
    )
    ax.set_xlabel("Topology", fontsize=11)
    ax.set_ylabel("Turing Type", fontsize=11)
    ax.set_title(
        "Max robustness per topology and Turing Type LHS comparison",
        fontsize=12, loc="left", pad=10,
    )
    fig.tight_layout()
    save(fig, "new_compare_fig1_heatmap")


def fig1_heatmap_typeI(df):
    pivot = (
        df.groupby(["topology_id", "turing_type"])["rob_shaberi_type_I"]
        .max()
        .unstack("topology_id")
        .reindex(index=["Type1", "Type2", "Type3"])
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="Greens",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Type I robustness (rob_shaberi_type_I)"},
    )
    ax.set_xlabel("Topology", fontsize=11)
    ax.set_ylabel("Turing Type", fontsize=11)
    ax.set_title(
        "Type I robustness per topology and Turing Type LHS comparison",
        fontsize=12, loc="left", pad=10,
    )
    fig.tight_layout()
    save(fig, "new_compare_typeI_fig1_heatmap")


def fig1_combined_heatmaps(df):
    # 1. Prepare data for the first heatmap (Total Robustness)
    pivot_total = (
        df.groupby(["topology_id", "turing_type"])["rob_shaberi_total"]
        .max()
        .unstack("topology_id")
        .reindex(index=["Type3", "Type2", "Type1"])
    )

    # 2. Prepare data for the second heatmap (Type I Robustness)
    pivot_typeI = (
        df.groupby(["topology_id", "turing_type"])["rob_shaberi_type_I"]
        .max()
        .unstack("topology_id")
        .reindex(index=["Type3", "Type2", "Type1"])
    )

    # 3. Create a 1-row, 2-column subplot structure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 4. Plot first heatmap (Left Panel)
    sns.heatmap(
        pivot_total,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax1,
        cbar_kws={"label": "Max robustness (rob_shaberi_total)"},
    )
    ax1.set_xlabel("Topology", fontsize=11)
    ax1.set_ylabel("Turing Type", fontsize=11)
    ax1.set_title(
        "Max robustness per topology and Turing Type LHS comparison",
        fontsize=11,
        loc="left",
        pad=10,
    )

    # 5. Plot second heatmap (Right Panel)
    sns.heatmap(
        pivot_typeI,
        annot=True,
        fmt=".4f",
        cmap="Purples",
        linewidths=0.5,
        linecolor="white",
        ax=ax2,
        cbar_kws={"label": "Type I robustness (rob_shaberi_type_I)"},
    )
    ax2.set_xlabel("Topology", fontsize=11)
    ax2.set_ylabel("Turing Type", fontsize=11)
    ax2.set_title(
        "Type I robustness per topology and Turing Type LHS comparison",
        fontsize=11,
        loc="left",
        pad=10,
    )

    fig.tight_layout()
    save(fig, "thesis_combined_fig1_heatmaps")








########## FIGURE 6: Stacked absolute bar – Type I vs II vs Hopf composition ##########


def fig_filter_distribution_all_configs(df):
    df = df.copy()

    # Target topologies requested
    topos = ["#1754", "#3954"]
    types = ["Type1", "Type2", "Type3"]

    # 1. Aggregate sums across ALL configs globally by topology and turing type
    grouped = (
        df.groupby(["topology_id", "turing_type"])
        .agg(
            type_I=("shaberi_type_I", "sum"),
            type_II=("shaberi_type_II", "sum"),
            hopf=("shaberi_hopf", "sum"),
            turing_filter=("filter_count", "sum"),
        )
        .reset_index()
    )

    # 2. Set up 2 panels side-by-side (one for each topology)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, topo in zip(axes, topos):
        topo_data = grouped[grouped["topology_id"] == topo]

        # X positions for Type1, Type2, Type3
        x_pos = range(len(types))
        bar_width = 0.5

        for idx, t_type in enumerate(types):
            # Fetch data or fall back to zeros if type is missing
            row = topo_data[topo_data["turing_type"] == t_type]
            if row.empty:
                continue

            t1 = row["type_I"].values[0]
            t2 = row["type_II"].values[0]
            th = row["hopf"].values[0]
            tf = row["turing_filter"].values[0]

            total = t1 + t2 + th + tf
            if total == 0:
                continue

            # Convert to percentages to normalise across all configurations
            p1 = (t1 / total) * 100
            p2 = (t2 / total) * 100
            ph = (th / total) * 100
            pf = (tf / total) * 100

            # Stacked bars
            ax.bar(
                idx,
                p1,
                width=bar_width,
                color=PATTERN_COLORS["Type I"],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar(
                idx,
                p2,
                width=bar_width,
                color=PATTERN_COLORS["Type II"],
                edgecolor="white",
                linewidth=0.5,
                bottom=p1,
            )
            ax.bar(
                idx,
                ph,
                width=bar_width,
                color=PATTERN_COLORS["Hopf"],
                edgecolor="white",
                linewidth=0.5,
                bottom=p1 + p2,
            )
            ax.bar(
                idx,
                pf,
                width=bar_width,
                color=PATTERN_COLORS["Turing Filter"],
                edgecolor="white",
                linewidth=0.5,
                bottom=p1 + p2 + ph,
            )

            # Display raw filter count text on top of the bars if filters exist
            if tf > 0:
                ax.text(
                    idx,
                    p1 + p2 + ph + (pf / 2),
                    f"n={int(tf)}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=9,
                )

        # Panel styling
        ax.set_title(f"Topology {topo}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(types, fontsize=11)
        ax.xaxis.grid(False)

    # Global formatting
    axes[0].set_ylabel(
        "Proportion of Total Counts across All Configs (%)", fontsize=11
    )
    plt.suptitle(
        "Turing Filters are Exclusively Present within Turing Type III Configurations",
        fontsize=14,
        y=0.98,
        fontweight="semibold",
    )

    # Unified clean legend placement on the right border
    handles = [
        mpatches.Patch(color=c, label=l) for l, c in PATTERN_COLORS.items()
    ]
    axes[1].legend(
        handles=handles,
        title="Pattern Type",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.08, right=0.85, top=0.88, bottom=0.12, wspace=0.2
    )
    save(fig, "turing_filter_distribution_all_configs")




def fig_all_patterns_profile_trends(df):
    df = df.copy()

    topos = ["#1754", "#3954"]
    types = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]  # Clean display names for X-axis

    # 1. Aggregate global sums for ALL configurations combined per topology × turing type
    grouped = (
        df.groupby(["topology_id", "turing_type"])
        .agg(
            type_I=("shaberi_type_I", "sum"),
            type_II=("shaberi_type_II", "sum"),
            hopf=("shaberi_hopf", "sum"),
            turing_filter=("filter_count", "sum"),
        )
        .reset_index()
    )

    # 2. Set up a less high, sleeker 2-panel layout (Height reduced to 4.8)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for i, (ax, topo) in enumerate(zip(axes, topos)):
        topo_data = grouped[grouped["topology_id"] == topo]

        # Initialize lists to store profile coordinates for plotting
        percentages = {
            "Type I": [],
            "Type II": [],
            "Hopf": [],
            "Turing Filter": [],
        }

        # Calculate the 100% composition breakdown for each step sequentially
        for t_type in types:
            row = topo_data[topo_data["turing_type"] == t_type]

            if not row.empty:
                t1 = row["type_I"].values[0]
                t2 = row["type_II"].values[0]
                th = row["hopf"].values[0]
                tf = row["turing_filter"].values[0]
            else:
                t1 = t2 = th = tf = 0

            total = t1 + t2 + th + tf

            # Calculate and append relative percentages
            if total > 0:
                percentages["Type I"].append((t1 / total) * 100)
                percentages["Type II"].append((t2 / total) * 100)
                percentages["Hopf"].append((th / total) * 100)
                percentages["Turing Filter"].append((tf / total) * 100)
            else:
                for key in percentages:
                    percentages[key].append(0)

        # 3. Plot the structural profile trendline for each pattern type
        for pattern_name, color in PATTERN_COLORS.items():
            y_values = percentages[pattern_name]

            # Main sleek trendline
            ax.plot(
                labels,  # Use clean display labels
                y_values,
                color=color,
                linewidth=3.0,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=pattern_name,
                zorder=3,
            )

            # Smooth shaded area under each line to emphasize density transitions
            ax.fill_between(
                labels, y_values, 0, color=color, alpha=0.1, zorder=2
            )

        # Panel styling and visual polish
        ax.set_title(
            f"Topology {topo} Pattern Dynamics",
            fontsize=12,
            #fontweight="semibold",  # Changed from bold to semibold
            color="#222222",
            pad=14,
        )
        ax.set_xlabel("Turing Type (Diego et al. 2018)", fontsize=11, color="#333333", labelpad=8)
        ax.set_ylabel("Pattern Composition Proportion (%)", fontsize=11, color="#333333", labelpad=8)
        
        # Enforce all axes showing numbers/ticks clearly
        ax.tick_params(axis="both", which="major", labelsize=10, labelleft=True, colors="#444444")
        ax.set_ylim(-3, 103)
        
        # Ultra-clean faint layout grids
        ax.grid(True, axis="both", linestyle=":", alpha=0.5, color="#cccccc", zorder=0)
        
        # Remove top and right borders for a premium, lightweight look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

    # Global Main Header
    plt.suptitle(
        "Distribution of Turing Instability Types Across Topological Network Types", # Changes in Turing Instability Composition When Moving from Type I to Type III
        fontsize=13,
        y=0.96,
        fontweight="semibold",
        color="#111111"
    )

    # 4. Construct bottom center legend using the exact line styles plotted
    legend_handles = [
        mlines.Line2D(
            [], [], 
            color=c, 
            linewidth=3.0, 
            marker="o", 
            markersize=7, 
            markeredgecolor="white", 
            markeredgewidth=1.5, 
            label=l
        ) for l, c in PATTERN_COLORS.items()
    ]
    
    # Places legend perfectly aligned horizontally underneath both panels
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,                 # Arranges legend items in a clean single row
        frameon=False,
        fontsize=10.5
    )

    # Manual tight padding adjustments to preserve the bottom legend space
    fig.subplots_adjust(
        left=0.07, 
        right=0.95, 
        top=0.80, 
        bottom=0.22,  # Added padding at the bottom for the new legend layout
        wspace=0.22
    )
    
    save(fig, "thesis_type_profile_trends")






def fig_pseudo_phase_39542(df):
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    sub = df[df["topology_id"] == "#3954"].copy()
    norm = mcolors.Normalize(
        vmin=sub["rob_shaberi_type_I"].min(), # previously rob_shaberi_total but maybe just focus on type I
        vmax=sub["rob_shaberi_type_I"].max(),
    )

    pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]
    # Reduced figure width to 14 to pull the subplots physically closer together
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (xvar, yvar) in zip(axes, pairs):
        sc = ax.scatter(
            sub[xvar],
            sub[yvar],
            c=sub["rob_shaberi_type_I"],
            cmap="BuPu",
            norm=norm,
            s=300,
            edgecolors="#444444",
            linewidths=0.5,
        )
        ax.set_xlabel(xvar, fontsize=11)
        ax.set_ylabel(yvar, fontsize=11)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.xaxis.grid(False)
        ax.set_title(f"{xvar} vs {yvar}", fontsize=11)

    # 1. Manually build a dedicated axis box for the colour bar on the far right edge
    # Syntax: [left_position, bottom_position, width, height] relative to the whole canvas
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.65])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Robustness (rob_shaberi_type_I)", fontsize=10)

    fig.suptitle(
        "Topology #3954 – Pseudo phase diagram across diffusion rate combinations",
        fontsize=12,
        x=0.04,
        y=0.96,
        ha="left",
    )

    # 2. Tightened wspace from 0.3 to 0.18 to bring the 3 main panels close together
    # Shrunk right to 0.84 to completely insulate the plots from hitting the colorbar axis at 0.88
    fig.subplots_adjust(
        left=0.04, right=0.86, top=0.82, bottom=0.15, wspace=0.2
    )

    save(fig, "new2_3954_pseudo_phase_diagram")




def fig_pseudo_phase_combined(df):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Filter data for both topologies
    topos = ["#3954", "#1754"]
    sub_all = df[df["topology_id"].isin(topos)].copy()
    # Normalize globally across both topologies so colors are directly comparable
    norm = mcolors.Normalize(
        vmin=sub_all["rob_shaberi_type_I"].min(), # was before rob_shaberi_total
        vmax=sub_all["rob_shaberi_type_I"].max(),
    )

    pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]

    # 2 rows (one per topology), 3 columns (pairs)
    # Added sharex and sharey to keep the panels aligned and clean
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7.5), sharex=True, sharey=True
    )

    #ax.tick_params(labelbottom=True)

    # Loop through each row (topology) and column (variable pair)
    for row_idx, topo in enumerate(topos):
        sub = sub_all[sub_all["topology_id"] == topo]

        for col_idx, (xvar, yvar) in enumerate(pairs):
            ax = axes[row_idx, col_idx]

            sc = ax.scatter(
                sub[xvar],
                sub[yvar],
                c=sub["rob_shaberi_type_I"],
                cmap="PuRd",
                norm=norm,
                s=250,  # Slightly smaller to prevent overcrowding in stacked views
                edgecolors="#444444",
                linewidths=0.5,
            )

            # Apply scales to all axes
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.xaxis.grid(False)

            # Only set pair titles on the top row to avoid duplication
            if row_idx == 0:
                ax.set_title(f"{xvar} vs {yvar}", fontsize=11, pad=8)

            # Only set x-labels on the bottom row to prevent overcrowding
            if row_idx == 1:
                ax.set_xlabel(xvar, fontsize=11)
            
            # ax.set_xlabel(xvar, fontsize=11) to add label to the first row

            # Only set y-labels on the left-most column
            if col_idx == 0:
                ax.set_ylabel(f"{topo}\n{yvar}", fontsize=11) # fontweight="bold"
            else:
                ax.set_ylabel(yvar, fontsize=11)

    # 1. Dedicated vertical axis box for the single global colour bar
    # Height adjusted to 0.70 to scale nicely with the taller 2-row figure
    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.70])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Robustness of Turing Type I", fontsize=11)

    fig.suptitle(
        "Phase Diagram Across Diffusion Rate Combinations for Topologies #3954 and #1754",
        fontsize=13,
        x=0.04,
        y=0.96,
        ha="left",
        fontweight="semibold",
    )

    # 2. Tight manual spacing adjustments for the 2D grid matrix
    # hspace handles the vertical gap between row 1 and row 2
    fig.subplots_adjust(
        left=0.06, right=0.84, top=0.88, bottom=0.12, wspace=0.18, hspace=0.15
    )

    save(fig, "thesis_combined_pseudo_phase_diagram")

############

# Add this function with your other plotting functions

# def fig_3d_parameter_space():
#     # Load detailed parameter data
#     df_params = pd.read_csv(PARAMS_CSV)
    
#     # Choose which config(s) to plot
#     # Option 1: Plot just one config (the most robust)
#     config_to_plot = 13  # Your most robust config
#     df_plot = df_params[df_params['config_id'] == config_to_plot]
    
#     # Extract the 3 production rates
#     beta_u = df_plot['beta_u'].values
#     beta_v = df_plot['beta_v'].values
#     beta_w = df_plot['beta_w'].values
    
#     print(f"\nConfig {config_to_plot}: {len(df_plot)} points")
#     print(f"  beta_u range: [{beta_u.min():.2f}, {beta_u.max():.2f}]")
#     print(f"  beta_v range: [{beta_v.min():.2f}, {beta_v.max():.2f}]")
#     print(f"  beta_w range: [{beta_w.min():.2f}, {beta_w.max():.2f}]")
    
#     # Create 3D plot
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Scatter plot of all Turing parameter sets
#     scatter = ax.scatter(beta_u, beta_v, beta_w,
#                         c=df_plot['max_growth_rate'],  # Color by Turing strength
#                         cmap='YlOrRd',
#                         s=50,
#                         alpha=0.6,
#                         edgecolors='k',
#                         linewidth=0.5)
    
#     # Optional: Draw convex hull if you have enough points (>10)
#     if len(df_plot) > 10:
#         points = np.column_stack([beta_u, beta_v, beta_w])
#         try:
#             hull = ConvexHull(points)
#             # Plot hull surface
#             for simplex in hull.simplices:
#                 triangle = points[simplex]
#                 ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
#                                color='orange', alpha=0.2, linewidth=0)
#         except:
#             print("  (Not enough points for convex hull)")
    
#     # Labels and styling
#     ax.set_xlabel('β$_u$ (u production rate)', fontsize=12, labelpad=10)
#     ax.set_ylabel('β$_v$ (v production rate)', fontsize=12, labelpad=10)
#     ax.set_zlabel('β$_w$ (w production rate)', fontsize=12, labelpad=10)
    
#     config_name = df_plot['config_name'].iloc[0]
#     ax.set_title(f'Turing Parameter Space: {config_name}\n'
#                  f'(dU={df_plot["dU"].iloc[0]}, dV={df_plot["dV"].iloc[0]}, dW={df_plot["dW"].iloc[0]})',
#                  fontsize=13, pad=20)
    
#     # Colorbar
#     cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
#     cbar.set_label('Max Growth Rate (Turing strength)', fontsize=11)
    
#     # Better viewing angle
#     ax.view_init(elev=20, azim=45)
    
#     save(fig, "fig_3d_turing_island_single_config")


# think about maybe comparing different parametres so not necessarily beta u, beta v and beta w
def fig_3d_parameter_space_comparison():
    df_params = pd.read_csv(PARAMS_CSV)
    
    # Choose three configs to compare
    config_a = 4    # type I config 
    config_b = 32   # type II config
    config_c = 50   # type III config
    
    df_a = df_params[df_params['config_id'] == config_a] # type I config 
    df_b = df_params[df_params['config_id'] == config_b] # type II config
    df_c = df_params[df_params['config_id'] == config_c] # type III config
    
    def format_title(df_sub):
        if df_sub.empty:
            return "No Data"
        
        # 1. Extract the raw string (e.g., "NEW_LHS_1754_Type1_Control_Slow")
        raw_name = str(df_sub["config_name"].iloc[0])
        parts = raw_name.split("_")
        
        # Find the part containing "Type" and standardise it to "Type X"
        type_str = "Type"
        for part in parts:
            if "Type" in part:
                # Converts 'Type1' -> 'Type 1', 'TypeI' -> 'Type 1', etc.
                if "1" in part or "I" in part and "III" not in part:
                    type_str = "Type 1"
                elif "2" in part or "II" in part and "III" not in part:
                    type_str = "Type 2"
                elif "3" in part or "III" in part:
                    type_str = "Type 3"
                else:
                    type_str = part
                break
        
        # 2. Pull the numerical diffusion rates from their specific columns
        du = df_sub["dU"].iloc[0]
        dv = df_sub["dV"].iloc[0]
        dw = df_sub["dW"].iloc[0]
        
        # 3. Combine into a clean layout with plain dU, dV, dW text
        return f"{type_str} (dU={du}, dV={dv}, dW={dw})\n{len(df_sub)} Turing sets"

    # Create plot
    fig = plt.figure(figsize=(18, 6))
    
    # Plot A
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(df_a['beta_u'], df_a['beta_v'], df_a['beta_w'],
               c='blue', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_a}')
    ax1.set_xlabel('β$_u$', fontsize=11)
    ax1.set_ylabel('β$_v$', fontsize=11)
    ax1.set_zlabel('β$_w$', fontsize=11)
    ax1.set_title(format_title(df_a), fontsize=11)
    ax1.view_init(elev=30, azim=45) # Changed to look down from above, prev azim = -60
    
    # Plot B
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(df_b['beta_u'], df_b['beta_v'], df_b['beta_w'],
               c='fuchsia', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_b}')
    ax2.set_xlabel('β$_u$', fontsize=11)
    ax2.set_ylabel('β$_v$', fontsize=11)
    ax2.set_zlabel('β$_w$', fontsize=11)
    ax2.set_title(format_title(df_b), fontsize=11)
    ax2.view_init(elev=30, azim=45) # Changed to look down from above, prev azim = -60

    # Plot C
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(df_c['beta_u'], df_c['beta_v'], df_c['beta_w'],
               c='green', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_c}')
    ax3.set_xlabel('β$_u$', fontsize=11)
    ax3.set_ylabel('β$_v$', fontsize=11)
    ax3.set_zlabel('β$_w$', fontsize=11)
    ax3.set_title(format_title(df_c), fontsize=11)
    ax3.view_init(elev=30, azim=45) # Changed to look down from above
    
    plt.suptitle('3954 Turing Parameter Space Comparison for Turing Instabilities Type I', fontsize=14, y=0.98)
    save(fig, "new_fig_3d_turing_island_comparison")






########### RUN THE WHOLE THING ############

df = load_all()
# fig1_heatmap(df)
# fig1_heatmap_typeI(df)
# fig_pseudo_phase_39542(df)

fig1_combined_heatmaps(df)
fig_filter_distribution_all_configs(df)
fig_all_patterns_profile_trends(df)
fig_pseudo_phase_combined(df)
fig_3d_parameter_space_comparison()