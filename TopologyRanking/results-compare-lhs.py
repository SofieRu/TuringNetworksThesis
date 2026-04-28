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
    "Type1": "#4C72B0",
    "Type2": "#DD8452",
    "Type3": "#55A868",
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
        cmap="YlGnBu",
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
    










########### RUN THE WHOLE THING ############

df = load_all()
fig1_heatmap(df)