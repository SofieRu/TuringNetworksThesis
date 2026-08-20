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

CSVS = {"#1754": "Topology1754/1754_FINAL_lhs_results_summary.csv", "#1823": "Topology1823/1823_FINAL_lhs_results_summary.csv", "#1838": "Topology1838/1838_FINAL_lhs_results_summary.csv", "#3954": "Topology3954/3954_FINAL_lhs_results_summary.csv",}

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


# type I vs II vs hopf composition, absolute differences 
def fig_all_patterns_profile_trends_complete(df):
    df = df.copy()
    types = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]

    grouped = (df.groupby(["topology_id", "turing_type"])
        .agg(type_I=("shaberi_type_I", "sum"), type_II=("shaberi_type_II", "sum"), hopf=("shaberi_hopf", "sum"), turing_filter=("filter_count", "sum"),)
        .reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.6), sharey=False)
    for row_idx, topo in enumerate(topo_order):
        topo_data = grouped[grouped["topology_id"] == topo]

        percentages = {"Type I": [], "Type II": [], "Hopf": [], "Turing Filter": []}
        absolutes = {"Type I": [], "Type II": [], "Hopf": [], "Turing Filter": []}

        # metrics for each step
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

            # absolute metrics
            absolutes["Type I"].append(t1)
            absolutes["Type II"].append(t2)
            absolutes["Hopf"].append(th)
            absolutes["Turing Filter"].append(tf)

            # relative percentage metrics
            if total > 0:
                percentages["Type I"].append((t1 / total) * 100)
                percentages["Type II"].append((t2 / total) * 100)
                percentages["Hopf"].append((th / total) * 100)
                percentages["Turing Filter"].append((tf / total) * 100)
            else:
                for key in percentages:
                    percentages[key].append(0)

        # col_idx 0 = percentages, col_idx 1 = absolute values
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            current_dataset = percentages if col_idx == 0 else absolutes

            # (0,0)->A, (0,1)->B, (1,0)->C, (1,1)->D
            panel_letter = chr(65 + (row_idx * 2 + col_idx)) 

            for pattern_name, color in PATTERN_COLORS.items():
                y_values = current_dataset[pattern_name]
                ax.plot(labels, y_values, color=color, linewidth=3.0, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=pattern_name, zorder=3)
                ax.fill_between(labels, y_values, 0, color=color, alpha=0.1, zorder=2)

            if col_idx == 0:
                ax.set_title(f"({panel_letter}) Topology {topo} Composition Proportion", fontsize=15, color="#222222", loc="left", pad=10)
                ax.set_ylabel("Instability Proportion (%)", fontsize=14, color="#333333", labelpad=8)
                ax.set_ylim(-3, 103)
            else:
                ax.set_title(f"({panel_letter}) Topology {topo} Absolute Count", fontsize=15, color="#222222", loc="left", pad=10)
                ax.set_ylabel("Instability Count (Total)", fontsize=14, color="#333333", labelpad=8)
                ax.autoscale(enable=True, axis='y', tight=False)

            ax.set_xlabel("Type (Diego et al. 2018)", fontsize=14, color="#333333", labelpad=6)
            ax.tick_params(axis="both", which="major", labelsize=13, labelleft=True, colors="#444444")
            ax.grid(True, axis="both", linestyle=":", alpha=0.5, color="#cccccc", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

    plt.suptitle("Distribution of Turing Instability Types Across Diffusion Types", fontsize=18, y=0.99,ha="center",color="#111111")
    legend_handles = [mlines.Line2D([], [], color=c,linewidth=3.0, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=l) for l, c in PATTERN_COLORS.items()]
    fig.legend(handles=legend_handles,loc="lower center",bbox_to_anchor=(0.5, -0.02),ncol=4,frameon=False,fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.87, bottom=0.12, wspace=0.3, hspace=0.6)
    
    save(fig, "final_type_profile_trends")


def fig_pseudo_phase_combined(df):
    robustness_col = "rob_shaberi_type_I"
    diffusion_cols = ["dU", "dV", "dW"]
    pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]
    label_map = {"dU": r"$D_U$", "dV": r"$D_V$", "dW": r"$D_W$"}
    
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
            sc = ax.scatter(sub[xvar], sub[yvar], c=sub[robustness_col], cmap="PuRd", norm=norm, s=250, edgecolors="#4E4242", linewidths=0.4, clip_on=False)
            
            ax.set_xscale("symlog", linthresh=0.1)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.tick_params(axis='x', labelsize=14)
            ax.xaxis.grid(False)
            
            if row_idx == 0:
                ax.set_title(f"{label_map[xvar]} vs {label_map[yvar]}", fontsize=14, pad=8)
                
            if row_idx == len(topo_order) - 1:
                ax.set_xlabel(label_map[xvar], fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"{topo}\n{label_map[yvar]}", fontsize=14)

    cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.70])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Robustness of Turing Type I", fontsize=14)
    fig.suptitle(f"Phase Diagram Across Diffusion Rate Combinations for {' and '.join(topo_order)}", fontsize=18, ha="center")
    fig.subplots_adjust(left=0.06, right=0.84, top=0.85, bottom=0.12, wspace=0.2, hspace=0.2)
    
    save(fig, "thesis_combined_phase_diagram")



def fig_thesis_combined_robustness_analysis(df):
    topo_colors = {"#1754": "#C588FDFF", "#3954": "#66A5F3FF"}

    # total robustness
    pivot_total = (df.groupby(["topology_id", "turing_type"])["rob_shaberi_total"].max().unstack("topology_id").reindex(index=["Type3", "Type2", "Type1"]))

    # type I robustness
    pivot_typeI = (df.groupby(["topology_id", "turing_type"])["rob_shaberi_type_I"].max().unstack("topology_id").reindex(index=["Type3", "Type2", "Type1"]))

    # data filtering
    df_box = df.copy()
    target_topologies = ["#1754", "#3954"]
    df_box = df_box[df_box["topology_id"].isin(target_topologies)]

    # boxplots average scores
    med_tot_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_total"].median()
    mean_tot_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_total"].mean()
    med_tot_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_total"].median()
    mean_tot_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_total"].mean()

    med_t1_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_type_I"].median()
    mean_t1_1754 = df_box[df_box["topology_id"] == "#1754"]["rob_shaberi_type_I"].mean()
    med_t1_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_type_I"].median()
    mean_t1_3954 = df_box[df_box["topology_id"] == "#3954"]["rob_shaberi_type_I"].mean()

    line_median = mlines.Line2D([], [], color="#222222", linewidth=1.5)
    line_mean = mlines.Line2D([], [], color="#D32F2F", linestyle="--", linewidth=1.5)

    # 2 rows, 2 columns grid mapping
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.1))
    ((ax_hm_tot, ax_box_tot), (ax_hm_t1, ax_box_t1)) = axes

    # heatmap on the left top
    sns.heatmap(pivot_total,annot=True,fmt=".4f",cmap="BuPu",linewidths=0.5,linecolor="white",ax=ax_hm_tot,cbar_kws={"label": "Robustness Score (in %)"},)
    ax_hm_tot.set_title("(A) Maximum Robustness Score (Type I, II, Filter)", fontsize=16, loc="left", pad=10,)
    ax_hm_tot.set_xlabel("Topology ID", fontsize=14)
    ax_hm_tot.set_ylabel("Type (Diego et al. 2018)", fontsize=14)

    # boxplot on the top right
    sns.boxplot(data=df_box, x="topology_id", y="rob_shaberi_total", order=topo_order, palette=topo_colors, ax=ax_box_tot, width=0.4, fliersize=0, linewidth=2.0, showmeans=True, meanline=True, meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"},)
    sns.stripplot(data=df_box, x="topology_id", y="rob_shaberi_total", order=topo_order, color="#0F0F0F", alpha=0.4, size=4, jitter=0.15, ax=ax_box_tot,)
    
    ax_box_tot.set_title("(B) Robustness Distribution (Type I, II, Filter)", fontsize=16, loc="left", pad=10,)
    ax_box_tot.set_xlabel("Topology ID", fontsize=14)
    ax_box_tot.set_ylabel("Robustness Score (%)", fontsize=14)

    handles_tot = [mpatches.Patch(color="#8448BBFF"), mpatches.Patch(color="#6C96DAFF"), line_median,line_mean,]
    labels_tot = [f"#1754 (Med: {med_tot_1754:.4f}%, Mean: {mean_tot_1754:.4f}%)", f"#3954 (Med: {med_tot_3954:.4f}%, Mean: {mean_tot_3954:.4f}%)", "Median", "Mean",]
    ax_box_tot.legend(handles=handles_tot, labels=labels_tot, loc="upper left", fontsize=11, frameon=True,)

    # heatmap on the left
    sns.heatmap(pivot_typeI, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5, linecolor="white", ax=ax_hm_t1, cbar_kws={"label": "Robustness Score (in %)"},)
    ax_hm_t1.set_title("(C) Maximum Robustness Score (Type I only)", fontsize=16, loc="left", pad=10,)
    ax_hm_t1.set_xlabel("Topology ID", fontsize=15)
    ax_hm_t1.set_ylabel("Type (Diego et al. 2018)", fontsize=15)

    # boxplot on the right
    sns.boxplot(data=df_box, x="topology_id", y="rob_shaberi_type_I", order=topo_order, palette=topo_colors, ax=ax_box_t1, width=0.4, fliersize=0, linewidth=2.0, showmeans=True, meanline=True, meanprops={"linestyle": "--", "linewidth": 2.2, "color": "#D32F2F"},)
    sns.stripplot(data=df_box, x="topology_id", y="rob_shaberi_type_I", order=topo_order, color="#0F0F0F", alpha=0.4, size=4, jitter=0.15, ax=ax_box_t1,)
    ax_box_t1.set_title("(D) Robustness Distribution (Type I only)", fontsize=16, loc="left", pad=10,)
    ax_box_t1.set_xlabel("Topology ID", fontsize=15)
    ax_box_t1.set_ylabel("Robustness Score (%)", fontsize=15)

    handles_t1 = [mpatches.Patch(color="#8448BBFF"), mpatches.Patch(color="#6C96DAFF"), line_median, line_mean,]
    labels_t1 = [f"#1754 (Med: {med_t1_1754:.4f}%, Mean: {mean_t1_1754:.4f}%)", f"#3954 (Med: {med_t1_3954:.4f}%, Mean: {mean_t1_3954:.4f}%)", "Median", "Mean",]
    ax_box_t1.legend(handles=handles_t1, labels=labels_t1, loc="upper left", fontsize=11, frameon=True)

    for ax in axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=14)

    title_text = "Total Turing Robustness vs Type I Robustness"
    fig.text(0.5, 0.99, title_text, fontsize=18, color="#111111", ha="center",)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.07, wspace=0.15, hspace=0.36)
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

    # x-axis labels short config name on top, diffusion ratio on bottom
    reference_subset = df_lab[df_lab["topology_id"] == "#3954"]
    x_labels = []
    for suffix in lab_suffixes:
        row = reference_subset[reference_subset["lab_suffix"] == suffix]
        if len(row) > 0:
            dU = row["dU"].iloc[0]
            dV = row["dV"].iloc[0]
            dW = row["dW"].iloc[0]
            short_name = suffix.replace("_WFreeze_", "")
            diff_str = f"{dU:g}, {dV:g}, {dW:g}"
            x_labels.append(f"{short_name}\n({diff_str})")

        else:
            x_labels.append(suffix.replace("_WFreeze_", ""))

    n_configs = len(lab_suffixes)
    x = np.arange(n_configs)
    bar_width = 0.38
    fig, ax = plt.subplots(figsize=(12.4, 5.6))

    bars_by_topo = {}
    for i, topo_id in enumerate(topo_order):
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, type_i_rob[topo_id], bar_width, label=f"Topology {topo_id}", color=topo_colors[topo_id], edgecolor="none",)
        bars_by_topo[topo_id] = bars

    # value labels on top of each bar
    for topo_id, bars in bars_by_topo.items():
        for bar, val in zip(bars, type_i_rob[topo_id]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,val + max(val * 0.03, 0.002),f"{val:.3f}%",ha="center",va="bottom",fontsize=10,color="#222222",)

    ax.set_xlabel("Diffusion configuration", fontsize=16, labelpad=8)
    ax.set_ylabel("Robustness score (%) for Type I", fontsize=16, labelpad=8)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title("Robustness across Biologically-Realistic Diffusion Configurations for #3954 and #1754 (Type I)", fontsize=18, loc="center", pad=12,)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=12)
    legend = ax.legend(fontsize=15,loc="upper right",frameon=True,framealpha=1.0,edgecolor="#dddddd",)
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
fig_thesis_combined_robustness_analysis(df)
fig6_lab_configs_comparison(df)