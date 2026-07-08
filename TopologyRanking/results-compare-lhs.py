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
    "#1754": "Topology1754/1754_FINAL_lhs_results_summary.csv",
    "#1823": "Topology1823/1823_PREFINAL_lhs_results_summary.csv",
    "#1838": "Topology1838/1838_PREFINAL_lhs_results_summary.csv",
    "#3954": "Topology3954/3954_FINAL_lhs_results_summary.csv",
}

PARAMS_CSV = "Topology3954/3954_FINAL_lhs_results_parameters.csv"

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
        #df = df[~df["config_name"].str.contains("OneFast|Control|Limit")]
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






def fig_all_patterns_profile_trends_complete(df):
    df = df.copy()

    # Reordered topos: row 1 -> #3954, row 2 -> #1754
    topos = ["#3954", "#1754"]
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

    # 2. Changed layout to 2x2 grid. sharey=False because absolute numbers vs percents need different scales.
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), sharey=False)

    for row_idx, topo in enumerate(topos):
        topo_data = grouped[grouped["topology_id"] == topo]

        # Initialize tracking dictionaries for both data modes
        percentages = {"Type I": [], "Type II": [], "Hopf": [], "Turing Filter": []}
        absolutes = {"Type I": [], "Type II": [], "Hopf": [], "Turing Filter": []}

        # Calculate metrics for each step sequentially
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

            # Track absolute metrics
            absolutes["Type I"].append(t1)
            absolutes["Type II"].append(t2)
            absolutes["Hopf"].append(th)
            absolutes["Turing Filter"].append(tf)

            # Track relative percentage metrics
            if total > 0:
                percentages["Type I"].append((t1 / total) * 100)
                percentages["Type II"].append((t2 / total) * 100)
                percentages["Hopf"].append((th / total) * 100)
                percentages["Turing Filter"].append((tf / total) * 100)
            else:
                for key in percentages:
                    percentages[key].append(0)

        # Loop through columns: col_idx 0 = Percentages, col_idx 1 = Absolute Values
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            current_dataset = percentages if col_idx == 0 else absolutes

            # Plot the structural profile trendline for each pattern type
            for pattern_name, color in PATTERN_COLORS.items():
                y_values = current_dataset[pattern_name]

                # Main sleek trendline
                ax.plot(
                    labels,
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

                # Smooth shaded area under each line
                ax.fill_between(
                    labels, y_values, 0, color=color, alpha=0.1, zorder=2
                )

            # Contextual labels and limits based on column type
            if col_idx == 0:
                ax.set_title(f"Topology {topo} Composition Proportion", fontsize=11, color="#222222", pad=10)
                ax.set_ylabel("Pattern Composition Proportion (%)", fontsize=10, color="#333333", labelpad=8)
                ax.set_ylim(-3, 103)
            else:
                ax.set_title(f"Topology {topo} Absolute Densities", fontsize=11, color="#222222", pad=10)
                ax.set_ylabel("Absolute Pattern Count (Total)", fontsize=10, color="#333333", labelpad=8)
                # Let absolute limits autoscale elegantly with small baseline padding
                ax.autoscale(enable=True, axis='y', tight=False)

            # Panel styling and visual polish
            ax.set_xlabel("Turing Type (Diego et al. 2018)", fontsize=10, color="#333333", labelpad=6)
            ax.tick_params(axis="both", which="major", labelsize=9.5, labelleft=True, colors="#444444")
            ax.grid(True, axis="both", linestyle=":", alpha=0.5, color="#cccccc", zorder=0)
            
            # Remove top and right borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

    # Global Main Header
    plt.suptitle(
        "Distribution of Turing Instability Types Across Topological Network Types",
        fontsize=13,
        x=0.09,                  # <-- Align with the left edge of your subplots (matches your left=0.09)
        y=0.97,
        ha="left",               # <-- Left bind the text anchor
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
    
    # Places legend perfectly aligned horizontally underneath all 4 panels
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=10.5
    )

    # Adjust layout padding to accommodate the 2x2 multi-row structure and bottom legend
    fig.subplots_adjust(
        left=0.08, 
        right=0.95, 
        top=0.88, 
        bottom=0.12,
        wspace=0.15,
        hspace=0.30
    )
    
    save(fig, "thesis_complete_type_profile_trends")


# def fig_pseudo_phase_39542(df):
#     import matplotlib.colors as mcolors
#     import matplotlib.cm as cm

#     sub = df[df["topology_id"] == "#3954"].copy()
#     norm = mcolors.Normalize(
#         vmin=sub["rob_shaberi_type_I"].min(), # previously rob_shaberi_total but maybe just focus on type I
#         vmax=sub["rob_shaberi_type_I"].max(),
#     )

#     pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]
#     # Reduced figure width to 14 to pull the subplots physically closer together
#     fig, axes = plt.subplots(1, 3, figsize=(14, 4))

#     for ax, (xvar, yvar) in zip(axes, pairs):
#         sc = ax.scatter(
#             sub[xvar],
#             sub[yvar],
#             c=sub["rob_shaberi_type_I"],
#             cmap="BuPu",
#             norm=norm,
#             s=300,
#             edgecolors="#444444",
#             linewidths=0.5,
#         )
#         ax.set_xlabel(xvar, fontsize=11)
#         ax.set_ylabel(yvar, fontsize=11)
#         ax.set_xscale("symlog", linthresh=0.1)
#         ax.set_yscale("symlog", linthresh=0.1)
#         ax.xaxis.grid(False)
#         ax.set_title(f"{xvar} vs {yvar}", fontsize=11)

#     # 1. Manually build a dedicated axis box for the colour bar on the far right edge
#     # Syntax: [left_position, bottom_position, width, height] relative to the whole canvas
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.65])
#     cbar = fig.colorbar(sc, cax=cbar_ax)
#     cbar.set_label("Robustness (rob_shaberi_type_I)", fontsize=10)

#     fig.suptitle(
#         "Topology #3954 – Pseudo phase diagram across diffusion rate combinations",
#         fontsize=12,
#         x=0.04,
#         y=0.96,
#         ha="left",
#     )

#     # 2. Tightened wspace from 0.3 to 0.18 to bring the 3 main panels close together
#     # Shrunk right to 0.84 to completely insulate the plots from hitting the colorbar axis at 0.88
#     fig.subplots_adjust(
#         left=0.04, right=0.86, top=0.82, bottom=0.15, wspace=0.2
#     )

#     save(fig, "new2_3954_pseudo_phase_diagram")




def fig_pseudo_phase_combined(df):

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
        left=0.06, right=0.84, top=0.88, bottom=0.12, wspace=0.2, hspace=0.26
    )

    save(fig, "thesis_combined_pseudo_phase_diagram")



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






#######
def fig_topology_robustness_comparison_final(df):
    df = df.copy()

    # 1. Filter strictly for your two target topologies
    target_topologies = ["#1754", "#3954"]
    df = df[df["topology_id"].isin(target_topologies)]
    
    topo_colors = {"#1754": "blueviolet", "#3954": "cornflowerblue"}
    topo_order = ["#1754", "#3954"]

    # 2. Assign standard CSV columns to plotting metrics
    df["plot_total"] = df["rob_shaberi_total"]
    df["plot_type_I"] = df["rob_shaberi_type_I"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- CALCULATE METRICS ---
    med_tot_1754 = df[df["topology_id"] == "#1754"]["plot_total"].median()
    mean_tot_1754 = df[df["topology_id"] == "#1754"]["plot_total"].mean()
    med_tot_3954 = df[df["topology_id"] == "#3954"]["plot_total"].median()
    mean_tot_3954 = df[df["topology_id"] == "#3954"]["plot_total"].mean()
    
    med_t1_1754 = df[df["topology_id"] == "#1754"]["plot_type_I"].median()
    mean_t1_1754 = df[df["topology_id"] == "#1754"]["plot_type_I"].mean()
    med_t1_3954 = df[df["topology_id"] == "#3954"]["plot_type_I"].median()
    mean_t1_3954 = df[df["topology_id"] == "#3954"]["plot_type_I"].mean()

    # Reusable style elements for the legend keys
    line_median = mlines.Line2D([], [], color='#222222', linewidth=1.5)
    line_mean = mlines.Line2D([], [], color='#D32F2F', linestyle='--', linewidth=1.5)

    # --- LEFT PLOT: Global Overall Robustness ---
    sns.boxplot(
        data=df, x="topology_id", y="plot_total", 
        order=topo_order, palette=topo_colors, ax=axes[0], 
        width=0.4, fliersize=0, linewidth=2.0,
        showmeans=True, meanline=True,
        meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"}
    )
    sns.stripplot(
        data=df, x="topology_id", y="plot_total",
        order=topo_order, color="#222222", alpha=0.25, size=4, jitter=0.15, ax=axes[0]
    )
    axes[0].set_title("Total Robustness Score\n(All Types Combined: Type I, II, Hopf and Turing Filter)", fontsize=11, pad=12)
    axes[0].set_ylabel("Robustness Score (%)", fontsize=10)

    # FIXED: Added the cornflowerblue patch handle for #3954
    handles_left = [
        mpatches.Patch(color='blueviolet'), 
        mpatches.Patch(color='cornflowerblue'), 
        line_median, 
        line_mean
    ]
    labels_left = [
        f"#1754 (Median: {med_tot_1754:.4f}%, Mean: {mean_tot_1754:.4f}%)",
        f"#3954 (Median: {med_tot_3954:.4f}%, Mean: {mean_tot_3954:.4f}%)",
        "Median",
        "Mean"
    ]
    axes[0].legend(handles=handles_left, labels=labels_left, loc="upper left", fontsize=9, frameon=True)

    # --- RIGHT PLOT: Genuine Turing Type I Robustness Only ---
    sns.boxplot(
        data=df, x="topology_id", y="plot_type_I",  
        order=topo_order, palette=topo_colors, ax=axes[1], 
        width=0.4, fliersize=0, linewidth=2.0,
        showmeans=True, meanline=True,
        meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"}
    )
    sns.stripplot(
        data=df, x="topology_id", y="plot_type_I",  
        order=topo_order, color="#222222", alpha=0.25, size=4, jitter=0.15, ax=axes[1]
    )
    axes[1].set_title("Pattern-Forming Robustness\n(Genuine Turing Type I Only)", fontsize=11, pad=12)
    axes[1].set_ylabel("Robustness Score (%)", fontsize=10)

    # FIXED: Added the cornflowerblue patch handle for #3954
    handles_right = [
        mpatches.Patch(color='blueviolet'), 
        mpatches.Patch(color='cornflowerblue'), 
        line_median, 
        line_mean
    ]
    labels_right = [
        f"#1754 (Median: {med_t1_1754:.4f}%, Mean: {mean_t1_1754:.4f}%)",
        f"#3954 (Median: {med_t1_3954:.4f}%, Mean: {mean_t1_3954:.4f}%)",
        "Median",
        "Mean"
    ]
    axes[1].legend(handles=handles_right, labels=labels_right, loc="upper left", fontsize=9, frameon=True)

    # 3. Clean up axis boundaries and labels across both panels
    for ax in axes:
        ax.set_xlabel("Topology ID", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=9.5)

    # 4. Global Main Header
    fig.text(0.5, 0.98, "Robustness and Parameter Space Density Comparison Between Topologies", fontsize=12.5, fontweight="semibold", color="#111111", ha="center")

    # 5. Fine tuning plot limits and spacing to guarantee nothing overlaps
    fig.subplots_adjust(left=0.08, right=0.94, top=0.84, bottom=0.12, wspace=0.12)

    save(fig, "thesis_topology_robustness_comparison")



########### RUN THE WHOLE THING ############

df = load_all()
# fig1_heatmap(df)
# fig1_heatmap_typeI(df)
# fig_pseudo_phase_39542(df)

fig1_combined_heatmaps(df)
fig_all_patterns_profile_trends_complete(df)
fig_all_patterns_profile_trends(df)
fig_pseudo_phase_combined(df)
fig_3d_parameter_space_comparison()

# Try Option A: Raw ratio format, zoomed in automatically on the tiny values (Highly Recommended)

# Run with auto-scaling to let your distributions expand fully in the frame
fig_topology_robustness_comparison_final(df)
