from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CSVS = {
    "#1754": "Topology1754/1754_lhs_results_corrected_1mio.csv",
    "#1823": "Topology1823/1823_results_summary_corrected.csv",
    "#1838": "Topology1838/1838_results_summary.csv",
    "#3954": "Topology3954/3954_lhs_results_corrected_1mio.csv",
}

# CSVS = {
#     "#1754": "Topology1754/1754_lhs_results_corrected_1mio.csv",
#     "#1823": "Topology1823/1823_results_summary_corrected.csv",
#     "#1838": "Topology1838/1838_results_summary.csv",
#     "#3954": "Topology3954/3954_lhs_results_newshaberi_1mio.csv",
# }

OUT_DIR = Path("ResultPlots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

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
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


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
    save(fig, "compare_fig1_heatmap")






########## FIGURE 2: Dumbbell plot Mean vs Max robustness per topology ##########
 
def fig2_dumbbell(df):
    topos   = ["#1754", "#1823", "#1838", "#3954"]
    means   = [df[df["topology_id"] == t]["rob_shaberi_total"].mean() for t in topos]
    maxes   = [df[df["topology_id"] == t]["rob_shaberi_total"].max()  for t in topos]
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for i, (topo, mean, maxi) in enumerate(zip(topos, means, maxes)):
        # connecting line
        ax.plot([mean, maxi], [i, i], color="#aaaaaa", linewidth=2, zorder=1)
        # mean dot
        ax.scatter(mean, i, color="#194386", s=100, zorder=2, label="Mean" if i == 0 else "")
        # max dot
        ax.scatter(maxi, i, color="#229F2B", s=100, zorder=2, label="Max"  if i == 0 else "")
 
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos, fontsize=11)
    ax.set_xlabel("Robustness (rob_shaberi_total)", fontsize=11)
    ax.set_title(
        "Mean vs max robustness per topology, LHS comparison",
        fontsize=12, loc="left", pad=10,
    )
    ax.yaxis.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
 
    fig.tight_layout()
    save(fig, "compare_fig2_dumbbell")




########## FIGURE 3: BASIC PLOT TO COMPARE ROBUSTNESS ##########

# ITS SUPPOSED TO BE rob_shaberi_type_I but i let it at type total for now bc i am confuseddddd :////
 
def fig3_best_config_bar(df):
    topos = ["#1754", "#1823", "#1838", "#3954"]
 
    best_configs = []
    for t in topos:
        subset  = df[df["topology_id"] == t]
        idx     = subset["rob_shaberi_total"].idxmax()
        best_configs.append({
            "topology_id" : t,
            "rob"         : subset.loc[idx, "rob_shaberi_total"],
            "config_name" : subset.loc[idx, "config_name"],
        })
 
    labels   = [d["topology_id"]  for d in best_configs]
    values   = [d["rob"]          for d in best_configs]
    configs  = [d["config_name"]  for d in best_configs]
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    bars = ax.bar(
        labels,
        values,
        color="#1F4E99",
        edgecolor="white",
        linewidth=0.5,
        width=0.5,
    )
 
    # Annotate each bar with the config name
    for bar, cfg in zip(bars, configs):
        # strip the topology prefix to keep it short
        short = cfg.split("_", 2)[-1]   # e.g. "DCC_Type2_Unequal1"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            short,
            ha="center", va="bottom",
            fontsize=8, color="#444444",
        )
 
    ax.set_ylabel("Max robustness (rob_shaberi_total)", fontsize=11)
    ax.set_title(
        "Highest robustness score per topology – LHS comparison",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
 
    fig.tight_layout()
    save(fig, "compare_fig3_best_config_bar")





########### RUN THE WHOLE THING ############

df = load_all()
fig1_heatmap(df)
fig2_dumbbell(df)
fig3_best_config_bar(df)