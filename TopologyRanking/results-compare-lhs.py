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
    "#1823": "Topology1823/1823_FINAL_lhs_results_summary.csv",
    "#1838": "Topology1838/1838_FINAL_lhs_results_summary.csv",
    "#3954": "Topology3954/3954_FINAL_lhs_results_summary.csv",
}

PARAMS_CSV = "Topology3954/3954_FINAL_lhs_results_parameters.csv"

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


#TYPE_COLORS = {"Type1": 'lightseagreen', "Type2": 'teal', "Type3": 'mediumpurple',} 
TYPE_COLORS = {"Type1": "#2E9F6E", "Type2": "#2B72DB", "Type3": "#E34D93"}
PATTERN_COLORS = {"Type I" : 'steelblue',"Type II" : 'mediumvioletred',"Hopf" : 'darkorange',"Turing Filter" : 'seagreen',}
TOPO_MARKERS = {"#1754": "o", "#1823": "s", "#1838": "^", "#3954": "D",}

topo_order = ["#1754", "#3954"]

def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/{name}.png")

def load_all():
    dfs = []
    for topo_id, path in CSVS.items():
        df = pd.read_csv(path)
        df["topology_id"]  = topo_id
        df["turing_type"]  = df["config_name"].str.extract(r"(Type[123])")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def parse_diff(s):
    try:
        return ast.literal_eval(s.replace("\u2018", "'").replace("\u2019", "'"))
    except:
        return {"dU": None, "dV": None, "dW": None}



########## FIGURE 6: Stacked absolute bar – Type I vs II vs Hopf composition ##########

def fig_all_patterns_profile_trends_complete(df):
    df = df.copy()

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
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.4), sharey=False)

    for row_idx, topo in enumerate(topo_order):
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

            # Maps: (0,0)->A, (0,1)->B, (1,0)->C, (1,1)->D
            panel_letter = chr(65 + (row_idx * 2 + col_idx)) 

            # Plot the structural profile trendline for each pattern type
            for pattern_name, color in PATTERN_COLORS.items():
                y_values = current_dataset[pattern_name]

                # Main sleek trendline
                ax.plot(labels, y_values, color=color, linewidth=3.0, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=pattern_name, zorder=3)

                # Smooth shaded area under each line
                ax.fill_between(labels, y_values, 0, color=color, alpha=0.1, zorder=2)

            # Contextual labels and limits based on column type
            if col_idx == 0:
                ax.set_title(f"({panel_letter}) Topology {topo} Composition Proportion", fontsize=13, color="#222222", loc="left", pad=10)
                ax.set_ylabel("Instability Proportion (%)", fontsize=13, color="#333333", labelpad=8)
                ax.set_ylim(-3, 103)
            else:
                ax.set_title(f"({panel_letter}) Topology {topo} Absolute Count", fontsize=13, color="#222222", loc="left", pad=10)
                ax.set_ylabel("Absolute Pattern Count (Total)", fontsize=13, color="#333333", labelpad=8)
                ax.autoscale(enable=True, axis='y', tight=False)

            # Panel styling and visual polish
            ax.set_xlabel("Type (Diego et al. 2018)", fontsize=13, color="#333333", labelpad=6)
            ax.tick_params(axis="both", which="major", labelsize=13, labelleft=True, colors="#444444")
            ax.grid(True, axis="both", linestyle=":", alpha=0.5, color="#cccccc", zorder=0)
            
            # Remove top and right borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

    plt.suptitle("Distribution of Turing Instability Types Across Diffusion Types", fontsize=16, y=0.97,ha="center",color="#111111")

    legend_handles = [mlines.Line2D([], [], color=c,linewidth=3.0, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=l) for l, c in PATTERN_COLORS.items()]
    
    fig.legend(handles=legend_handles,loc="lower center",bbox_to_anchor=(0.5, -0.02),ncol=4,frameon=False,fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.87, bottom=0.12, wspace=0.27, hspace=0.45)
    
    save(fig, "final_type_profile_trends")










def fig_pseudo_phase_combined(df):

    robustness_col = "rob_shaberi_type_I"
    diffusion_cols = ["dU", "dV", "dW"]
    pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]

    sub_original = df[df["topology_id"].isin(topo_order)].copy()
    sub_original = sub_original[~sub_original["config_name"].str.contains("_Lab", case=False, na=False, regex=False)].copy()
    sub_original["_config_key"] = sub_original["config_name"].str.replace(r"^FINAL_LHS_\d+_", "", regex=True)
    sub_original[robustness_col] = pd.to_numeric(sub_original[robustness_col], errors="coerce")

    config_keys = sub_original["_config_key"].dropna().unique()
    complete_grid = pd.MultiIndex.from_product([topo_order, config_keys], names=["topology_id", "_config_key"]).to_frame(index=False)
    sub_all = complete_grid.merge(sub_original, on=["topology_id", "_config_key"], how="left")

    diffusion_lookup = sub_original.dropna(subset=["_config_key"]).groupby("_config_key")[diffusion_cols].first()

    for column in diffusion_cols:
        sub_all[column] = sub_all[column].fillna(sub_all["_config_key"].map(diffusion_lookup[column]))

    sub_all[robustness_col] = sub_all[robustness_col].fillna(0.0)
    sub_all = sub_all.dropna(subset=diffusion_cols)
    max_robustness = sub_all[robustness_col].max()

    max_robustness = max_robustness if pd.notna(max_robustness) and max_robustness > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_robustness)
    fig, axes = plt.subplots(len(topo_order), 3, figsize=(12.4, 6.2), sharex=True, sharey=True, squeeze=False)

    for row_idx, topo in enumerate(topo_order):
        sub = sub_all[sub_all["topology_id"] == topo].sort_values(robustness_col)
        for col_idx, (xvar, yvar) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            sc = ax.scatter(sub[xvar], sub[yvar], c=sub[robustness_col], cmap="PuRd", norm=norm, s=250, edgecolors="#444444", linewidths=0.5)
            ax.set_xscale("symlog", linthresh=0.1)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.tick_params(axis='x', labelsize=14)
            ax.xaxis.grid(False)
            if row_idx == 0:
                ax.set_title(f"{xvar} vs {yvar}", fontsize=13, pad=8)
            if row_idx == len(topo_order) - 1:
                ax.set_xlabel(xvar, fontsize=13)
            ax.set_ylabel(f"{topo}\n{yvar}" if col_idx == 0 else yvar, fontsize=13)

    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.70])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Robustness of Turing Type I", fontsize=13)

    fig.suptitle(f"Phase Diagram Across Diffusion Rate Combinations for Topologies {' and '.join(topo_order)}", fontsize=16, ha="center")
    fig.subplots_adjust(left=0.06, right=0.84, top=0.88, bottom=0.12, wspace=0.2, hspace=0.2)
    save(fig, "thesis_combined_phase_diagram")








def fig_pseudo_phase_3d(df):
    robustness_col = "rob_shaberi_type_I"
    diffusion_cols = ["dU", "dV", "dW"]

    sub_original = df[df["topology_id"].isin(topo_order)].copy()
    sub_original = sub_original[~sub_original["config_name"].str.contains("_Lab", case=False, na=False, regex=False)].copy()
    sub_original["_config_key"] = sub_original["config_name"].str.replace(r"^FINAL_LHS_\d+_", "", regex=True)
    sub_original[robustness_col] = pd.to_numeric(sub_original[robustness_col], errors="coerce")

    config_keys = sub_original["_config_key"].dropna().unique()
    complete_grid = pd.MultiIndex.from_product([topo_order, config_keys], names=["topology_id", "_config_key"]).to_frame(index=False)
    sub_all = complete_grid.merge(sub_original, on=["topology_id", "_config_key"], how="left")

    diffusion_lookup = sub_original.dropna(subset=["_config_key"]).groupby("_config_key")[diffusion_cols].first()
    for column in diffusion_cols:
        sub_all[column] = sub_all[column].fillna(sub_all["_config_key"].map(diffusion_lookup[column]))

    sub_all[robustness_col] = sub_all[robustness_col].fillna(0.0)
    sub_all = sub_all.dropna(subset=diffusion_cols)
    max_robustness = sub_all[robustness_col].max()
    max_robustness = max_robustness if pd.notna(max_robustness) and max_robustness > 0 else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_robustness)

    def value_to_grid_index(val):
        if pd.isna(val) or val <= 0: return 0.0
        log_val = np.round(np.log10(val), 1)
        if log_val == -1.0: return 1.0
        if log_val == 0.0: return 2.0
        if log_val == 1.0: return 3.0
        return 0.0

    for col in diffusion_cols:
        sub_all[f"{col}_grid"] = sub_all[col].apply(value_to_grid_index)

    tick_positions = [0.0, 1.0, 2.0, 3.0]
    tick_labels = ["0", "$10^{-1}$", "$10^{0}$", "$10^{1}$"]

    fig = plt.figure(figsize=(15.0, 7.5))
    sc = None

    for idx, topo in enumerate(topo_order):
        sub = sub_all[sub_all["topology_id"] == topo].sort_values(robustness_col)
        ax = fig.add_subplot(1, len(topo_order), idx + 1, projection="3d")
        
        # 1. DRAW DROPLINES AND SHADOWS FIRST (so they sit behind the main balls)
        for _, row in sub.iterrows():
            # Only draw lines for active pattern configurations to avoid clutter
            if row[robustness_col] > 0:
                # Vertical line from floating point down to the floor (z = -0.2 boundary)
                ax.plot([row["dU_grid"], row["dU_grid"]], [row["dV_grid"], row["dV_grid"]], [-0.2, row["dW_grid"]], color="#888888", linestyle=":", linewidth=0.8, zorder=1)
                # Flat shadow dot on the floor
                ax.scatter(row["dU_grid"], row["dV_grid"], -0.2, color="#cccccc", s=50, alpha=0.5, zorder=2)

        # 2. PLOT MAIN 3D BALLS
        sc = ax.scatter(sub["dU_grid"], sub["dV_grid"], sub["dW_grid"], c=sub[robustness_col], cmap="PuRd", norm=norm, s=400, edgecolors="white", linewidths=0.6, alpha=0.95, zorder=10)
        
        ax.set_title(f"Topology {topo}", fontsize=14, pad=20, fontweight="bold")
        ax.set_xlabel("dU", fontsize=12, labelpad=12, fontweight="semibold")
        ax.set_ylabel("dV", fontsize=12, labelpad=12, fontweight="semibold")
        ax.set_zlabel("dW", fontsize=12, labelpad=12, fontweight="semibold")
        
        for axis_setup in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis_setup.set_ticks(tick_positions)
            axis_setup.set_ticklabels(tick_labels, fontsize=11)

        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(-0.2, 3.2)
        ax.set_zlim(-0.2, 3.2)
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.view_init(elev=20, azim=130)

    cbar_ax = fig.add_axes([0.91, 0.20, 0.018, 0.55])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Robustness of Turing Type I", fontsize=12, labelpad=12, fontweight="semibold")

    fig.suptitle("3D Phase Diagrams with Floor Projections", fontsize=15, y=0.94, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.88, top=0.86, bottom=0.14, wspace=0.15)
    
    save(fig, "thesis_combined_3d_phase_diagram")





def fig_thesis_combined_robustness_analysis(df):
    topo_colors = {"#1754": "#C588FDFF", "#3954": "#66A5F3FF"}

    # Heatmap 1 data (Total Robustness)
    pivot_total = (df.groupby(["topology_id", "turing_type"])["rob_shaberi_total"].max().unstack("topology_id").reindex(index=["Type3", "Type2", "Type1"]))

    # Heatmap 2 data (Type I Robustness)
    pivot_typeI = (df.groupby(["topology_id", "turing_type"])["rob_shaberi_type_I"].max().unstack("topology_id").reindex(index=["Type3", "Type2", "Type1"]))

    # Boxplot data filtering
    df_box = df.copy()
    target_topologies = ["#1754", "#3954"]
    df_box = df_box[df_box["topology_id"].isin(target_topologies)]

    # Calculate Boxplot Metrics
    med_tot_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_total"].median()
    mean_tot_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_total"].mean()
    med_tot_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_total"].median()
    mean_tot_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_total"].mean()

    med_t1_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_type_I"].median()
    mean_t1_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_type_I"].mean()
    med_t1_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_type_I"].median()
    mean_t1_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_type_I"].mean()

    # Reusable style elements for legends
    line_median = mlines.Line2D([], [], color="#222222", linewidth=1.5)
    line_mean = mlines.Line2D([], [], color="#D32F2F", linestyle="--", linewidth=1.5)

    # 2 rows, 2 columns grid mapping
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9))
    ((ax_hm_tot, ax_box_tot), (ax_hm_t1, ax_box_t1)) = axes

    # Left Column: Heatmap
    sns.heatmap(pivot_total,annot=True,fmt=".4f",cmap="BuPu",linewidths=0.5,linecolor="white",ax=ax_hm_tot,cbar_kws={"label": "Robustness Score (in %)"},)
    ax_hm_tot.set_title("(A) Maximum Robustness Score (Type I, II, Hopf, Filter)", fontsize=14, loc="left", pad=10,)
    ax_hm_tot.set_xlabel("Topology ID", fontsize=13)
    ax_hm_tot.set_ylabel("Type (Diego et al. 2018)", fontsize=13)

    # Right Column: Boxplot
    sns.boxplot(data=df_box, x="topology_id", y="rob_shaberi_total", order=topo_order, palette=topo_colors, ax=ax_box_tot, width=0.4, fliersize=0, linewidth=2.0, showmeans=True, meanline=True, meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"},)
    sns.stripplot(data=df_box, x="topology_id", y="rob_shaberi_total", order=topo_order, color="#0F0F0F", alpha=0.4, size=4, jitter=0.15, ax=ax_box_tot,)
    
    ax_box_tot.set_title("(B) Robustness Distribution (Type I, II, Hopf, Filter)", fontsize=14, loc="left", pad=10,)
    ax_box_tot.set_xlabel("Topology ID", fontsize=13)
    ax_box_tot.set_ylabel("Robustness Score (%)", fontsize=13)

    handles_tot = [mpatches.Patch(color="#8448BBFF"), mpatches.Patch(color="#6C96DAFF"), line_median,line_mean,]
    
    labels_tot = [f"#1754 (Med: {med_tot_1754:.4f}%, Mean: {mean_tot_1754:.4f}%)", f"#3954 (Med: {med_tot_3954:.4f}%, Mean: {mean_tot_3954:.4f}%)", "Median", "Mean",]
    
    ax_box_tot.legend(handles=handles_tot, labels=labels_tot, loc="upper left", fontsize=11, frameon=True,)

    # Left Column: Heatmap
    sns.heatmap(pivot_typeI, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5, linecolor="white", ax=ax_hm_t1, cbar_kws={"label": "Robustness Score (in %)"},)
    ax_hm_t1.set_title("(C) Maximum Robustness Score (Type I only)", fontsize=14, loc="left", pad=10,)
    ax_hm_t1.set_xlabel("Topology ID", fontsize=13)
    ax_hm_t1.set_ylabel("Type (Diego et al. 2018)", fontsize=13)

    # Right Column: Boxplot
    sns.boxplot(data=df_box, x="topology_id", y="rob_shaberi_type_I", order=topo_order, palette=topo_colors, ax=ax_box_t1, width=0.4, fliersize=0, linewidth=2.0, showmeans=True, meanline=True, meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"},)
    sns.stripplot(data=df_box, x="topology_id", y="rob_shaberi_type_I", order=topo_order, color="#0F0F0F", alpha=0.4, size=4, jitter=0.15, ax=ax_box_t1,)
    ax_box_t1.set_title("(D) Robustness Distribution (Type I only)", fontsize=14, loc="left", pad=10,)
    ax_box_t1.set_xlabel("Topology ID", fontsize=13)
    ax_box_t1.set_ylabel("Robustness Score (%)", fontsize=13)

    handles_t1 = [mpatches.Patch(color="#8448BBFF"), mpatches.Patch(color="#6C96DAFF"), line_median, line_mean,]
    labels_t1 = [f"#1754 (Med: {med_t1_1754:.4f}%, Mean: {mean_t1_1754:.4f}%)", f"#3954 (Med: {med_t1_3954:.4f}%, Mean: {mean_t1_3954:.4f}%)", "Median", "Mean",]
    ax_box_t1.legend(handles=handles_t1, labels=labels_t1, loc="upper left", fontsize=11, frameon=True)

    # CLEAN UP & LAYOUT TUNING
    for ax in axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=13)

    # Global Title Options (Choose one for your figure text below)
    title_text = "Total Turing Robustness vs Type I Robustness"
    fig.text(0.5, 0.97, title_text, fontsize=16, color="#111111", ha="center",)

    # Adjust margins tightly to avoid overlap with labels/titles
    fig.subplots_adjust(left=0.1, right=0.95, top=0.91, bottom=0.07, wspace=0.14, hspace=0.3)

    save(fig, "final_robustness_analysis")




# LAB FOCUS
def fig6_lab_configs_comparison(df):
    topo_colors = {"#1754": "#8448BBFF", "#3954": "#6C96DAFF"}
    lab_suffixes = ["_WFreeze_Equal1","_WFreeze_Lab1","_WFreeze_Lab2","_WFreeze_Lab3","_WFreeze_Lab4","_WFreeze_Lab5","_WFreeze_Lab6",]

    def extract_suffix(config_name):
        for suffix in lab_suffixes:
            if str(config_name).endswith(suffix):
                return suffix
        return None

    df_lab = df.copy()
    df_lab["lab_suffix"] = df_lab["config_name"].apply(extract_suffix)
    df_lab = df_lab[df_lab["lab_suffix"].notna()]
    df_lab = df_lab[df_lab["topology_id"].isin(topo_order)]

    type_i_rob = {}
    for topo_id in topo_order:
        subset = df_lab[df_lab["topology_id"] == topo_id]
        values = []
        for suffix in lab_suffixes:
            row = subset[subset["lab_suffix"] == suffix]
            if len(row) > 0:
                values.append(row["rob_shaberi_type_I"].iloc[0])
            else:
                values.append(np.nan)
        type_i_rob[topo_id] = values

    # X-axis labels: short config name on top, diffusion ratio on bottom
    reference_subset = df_lab[df_lab["topology_id"] == "#3954"]
    x_labels = []
    for suffix in lab_suffixes:
        row = reference_subset[reference_subset["lab_suffix"] == suffix]
        if len(row) > 0:
            dU = row["dU"].iloc[0]
            dV = row["dV"].iloc[0]
            dW = row["dW"].iloc[0]

            # Short display: strip "WFreeze_" prefix
            short_name = suffix.replace("_WFreeze_", "")

            diff_str = f"dU:{dU:g}, dV:{dV:g}, dW:{dW:g}"
            x_labels.append(f"{short_name}\n({diff_str})")

            # Format diffusion as ratio (dU:dV:dW)
            # diff_str = f"{dU:g},{dV:g},{dW:g}"
            # x_labels.append(f"{short_name}\n({diff_str})")

        else:
            x_labels.append(suffix.replace("_WFreeze_", ""))

    n_configs = len(lab_suffixes)
    x = np.arange(n_configs)
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(12.4, 5))

    bars_by_topo = {}
    for i, topo_id in enumerate(topo_order):
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            type_i_rob[topo_id],
            bar_width,
            label=f"Topology {topo_id}",
            color=topo_colors[topo_id],
            edgecolor="none",
        )
        bars_by_topo[topo_id] = bars

    # Value labels on top of each bar
    for topo_id, bars in bars_by_topo.items():
        for bar, val in zip(bars, type_i_rob[topo_id]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max(val * 0.03, 0.002),
                    f"{val:.3f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#222222",
                )

    ax.set_xlabel("Diffusion configuration", fontsize=14, labelpad=8)
    ax.set_ylabel("Robustness score (%) for Type I", fontsize=14, labelpad=8)
    ax.set_title("Robustness across Biologically-Realistic Diffusion Configurations for Topologies #3954 and #1754 (Type I only)", fontsize=16, loc="center", pad=12,)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=11)

    legend = ax.legend(
        fontsize=14,
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="#dddddd",
    )
    legend.get_frame().set_facecolor("white")

    all_values = [v for topo_values in type_i_rob.values() for v in topo_values if not np.isnan(v)]
    if all_values:
        ax.set_ylim(0, max(all_values) * 1.20)

    fig.tight_layout()
    save(fig, "final_compare_lab_configs")




########### RUN THE WHOLE THING ############

df = load_all()
fig_all_patterns_profile_trends_complete(df)
fig_pseudo_phase_combined(df)
fig_pseudo_phase_3d(df)
fig_thesis_combined_robustness_analysis(df)
fig6_lab_configs_comparison(df)






#######
# def fig_topology_robustness_comparison_final(df):
#     df = df.copy()

#     # 1. Filter strictly for your two target topologies
#     target_topologies = ["#1754", "#3954"]
#     df = df[df["topology_id"].isin(target_topologies)]
    
#     topo_colors = {"#1754": "blueviolet", "#3954": "cornflowerblue"}
#     topo_order = ["#1754", "#3954"]

#     # 2. Assign standard CSV columns to plotting metrics
#     df["plot_total"] = df["rob_shaberi_total"]
#     df["plot_type_I"] = df["rob_shaberi_type_I"]

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

#     # --- CALCULATE METRICS ---
#     med_tot_1754 = df[df["topology_id"] == "#1754"]["plot_total"].median()
#     mean_tot_1754 = df[df["topology_id"] == "#1754"]["plot_total"].mean()
#     med_tot_3954 = df[df["topology_id"] == "#3954"]["plot_total"].median()
#     mean_tot_3954 = df[df["topology_id"] == "#3954"]["plot_total"].mean()
    
#     med_t1_1754 = df[df["topology_id"] == "#1754"]["plot_type_I"].median()
#     mean_t1_1754 = df[df["topology_id"] == "#1754"]["plot_type_I"].mean()
#     med_t1_3954 = df[df["topology_id"] == "#3954"]["plot_type_I"].median()
#     mean_t1_3954 = df[df["topology_id"] == "#3954"]["plot_type_I"].mean()

#     # Reusable style elements for the legend keys
#     line_median = mlines.Line2D([], [], color='#222222', linewidth=1.5)
#     line_mean = mlines.Line2D([], [], color='#D32F2F', linestyle='--', linewidth=1.5)

#     # --- LEFT PLOT: Global Overall Robustness ---
#     sns.boxplot(
#         data=df, x="topology_id", y="plot_total", 
#         order=topo_order, palette=topo_colors, ax=axes[0], 
#         width=0.4, fliersize=0, linewidth=2.0,
#         showmeans=True, meanline=True,
#         meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"}
#     )
#     sns.stripplot(
#         data=df, x="topology_id", y="plot_total",
#         order=topo_order, color="#222222", alpha=0.45, size=4, jitter=0.15, ax=axes[0]
#     )
#     axes[0].set_title("Total Robustness Score\n(All Types Combined: Type I, II, Hopf and Turing Filter)", fontsize=11, pad=12)
#     axes[0].set_ylabel("Robustness Score (%)", fontsize=10)

#     # FIXED: Added the cornflowerblue patch handle for #3954
#     handles_left = [
#         mpatches.Patch(color='blueviolet'), 
#         mpatches.Patch(color='cornflowerblue'), 
#         line_median, 
#         line_mean
#     ]
#     labels_left = [
#         f"#1754 (Median: {med_tot_1754:.4f}%, Mean: {mean_tot_1754:.4f}%)",
#         f"#3954 (Median: {med_tot_3954:.4f}%, Mean: {mean_tot_3954:.4f}%)",
#         "Median",
#         "Mean"
#     ]
#     axes[0].legend(handles=handles_left, labels=labels_left, loc="upper left", fontsize=9, frameon=True)

#     # --- RIGHT PLOT: Genuine Turing Type I Robustness Only ---
#     sns.boxplot(
#         data=df, x="topology_id", y="plot_type_I",  
#         order=topo_order, palette=topo_colors, ax=axes[1], 
#         width=0.4, fliersize=0, linewidth=2.0,
#         showmeans=True, meanline=True,
#         meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"}
#     )
#     sns.stripplot(
#         data=df, x="topology_id", y="plot_type_I",  
#         order=topo_order, color="#222222", alpha=0.45, size=4, jitter=0.15, ax=axes[1]
#     )
#     axes[1].set_title("Pattern-Forming Robustness\n(Genuine Turing Type I Only)", fontsize=11, pad=12)
#     axes[1].set_ylabel("Robustness Score (%)", fontsize=10)

#     # FIXED: Added the cornflowerblue patch handle for #3954
#     handles_right = [
#         mpatches.Patch(color='blueviolet'), 
#         mpatches.Patch(color='cornflowerblue'), 
#         line_median, 
#         line_mean
#     ]
#     labels_right = [
#         f"#1754 (Median: {med_t1_1754:.4f}%, Mean: {mean_t1_1754:.4f}%)",
#         f"#3954 (Median: {med_t1_3954:.4f}%, Mean: {mean_t1_3954:.4f}%)",
#         "Median",
#         "Mean"
#     ]
#     axes[1].legend(handles=handles_right, labels=labels_right, loc="upper left", fontsize=9, frameon=True)

#     # 3. Clean up axis boundaries and labels across both panels
#     for ax in axes:
#         ax.set_xlabel("Topology ID", fontsize=10)
#         ax.tick_params(axis="both", which="major", labelsize=9.5)

#     # 4. Global Main Header
#     fig.text(0.5, 0.98, "Robustness and Parameter Space Density Comparison Between Topologies", fontsize=12.5, fontweight="semibold", color="#111111", ha="center")

#     # 5. Fine tuning plot limits and spacing to guarantee nothing overlaps
#     fig.subplots_adjust(left=0.08, right=0.94, top=0.84, bottom=0.12, wspace=0.12)

#     save(fig, "thesis_topology_robustness_comparison")




# # think about maybe comparing different parametres so not necessarily beta u, beta v and beta w
# def fig_3d_parameter_space_comparison():
#     df_params = pd.read_csv(PARAMS_CSV)
    
#     # Choose three configs to compare
#     config_a = 4    # type I config 
#     config_b = 32   # type II config
#     config_c = 50   # type III config
    
#     df_a = df_params[df_params['config_id'] == config_a] # type I config 
#     df_b = df_params[df_params['config_id'] == config_b] # type II config
#     df_c = df_params[df_params['config_id'] == config_c] # type III config
    
#     def format_title(df_sub):
#         if df_sub.empty:
#             return "No Data"
        
#         # 1. Extract the raw string (e.g., "NEW_LHS_1754_Type1_Control_Slow")
#         raw_name = str(df_sub["config_name"].iloc[0])
#         parts = raw_name.split("_")
        
#         # Find the part containing "Type" and standardise it to "Type X"
#         type_str = "Type"
#         for part in parts:
#             if "Type" in part:
#                 # Converts 'Type1' -> 'Type 1', 'TypeI' -> 'Type 1', etc.
#                 if "1" in part or "I" in part and "III" not in part:
#                     type_str = "Type 1"
#                 elif "2" in part or "II" in part and "III" not in part:
#                     type_str = "Type 2"
#                 elif "3" in part or "III" in part:
#                     type_str = "Type 3"
#                 else:
#                     type_str = part
#                 break
        
#         # 2. Pull the numerical diffusion rates from their specific columns
#         du = df_sub["dU"].iloc[0]
#         dv = df_sub["dV"].iloc[0]
#         dw = df_sub["dW"].iloc[0]
        
#         # 3. Combine into a clean layout with plain dU, dV, dW text
#         return f"{type_str} (dU={du}, dV={dv}, dW={dw})\n{len(df_sub)} Turing sets"

#     # Create plot
#     fig = plt.figure(figsize=(18, 6))
    
#     # Plot A
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.scatter(df_a['beta_u'], df_a['beta_v'], df_a['beta_w'],
#                c='blue', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
#                label=f'Config {config_a}')
#     ax1.set_xlabel('β$_u$', fontsize=11)
#     ax1.set_ylabel('β$_v$', fontsize=11)
#     ax1.set_zlabel('β$_w$', fontsize=11)
#     ax1.set_title(format_title(df_a), fontsize=11)
#     ax1.view_init(elev=30, azim=45) # Changed to look down from above, prev azim = -60
    
#     # Plot B
#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.scatter(df_b['beta_u'], df_b['beta_v'], df_b['beta_w'],
#                c='fuchsia', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
#                label=f'Config {config_b}')
#     ax2.set_xlabel('β$_u$', fontsize=11)
#     ax2.set_ylabel('β$_v$', fontsize=11)
#     ax2.set_zlabel('β$_w$', fontsize=11)
#     ax2.set_title(format_title(df_b), fontsize=11)
#     ax2.view_init(elev=30, azim=45) # Changed to look down from above, prev azim = -60

#     # Plot C
#     ax3 = fig.add_subplot(133, projection='3d')
#     ax3.scatter(df_c['beta_u'], df_c['beta_v'], df_c['beta_w'],
#                c='green', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
#                label=f'Config {config_c}')
#     ax3.set_xlabel('β$_u$', fontsize=11)
#     ax3.set_ylabel('β$_v$', fontsize=11)
#     ax3.set_zlabel('β$_w$', fontsize=11)
#     ax3.set_title(format_title(df_c), fontsize=11)
#     ax3.view_init(elev=30, azim=45) # Changed to look down from above
    
#     plt.suptitle('3954 Turing Parameter Space Comparison for Turing Instabilities Type I', fontsize=14, y=0.98)
#     save(fig, "new_fig_3d_turing_island_comparison")

