from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a

CSV = "3954_results_summary_curated_1mio.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family"     : "sans-serif",
    "font.size"       : 10,
    "axes.edgecolor"  : "#444444",
    "axes.linewidth"  : 0.8,
    "grid.color"      : "#dddddd",
    "grid.linewidth"  : 0.7,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})


TYPE_COLORS = {
    "Type1": "#444EA6",
    "Type2": "#AE2BA1",
    "Type3": "#3FA051",
}

def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.svg")


def load_data():
    df = pd.read_csv(CSV)

    # Extract topology (DCC / CDD / CCD …) and Turing type (Type1/2/3)
    df["topology"]    = df["config_name"].str.extract(r"3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")

    return df


# Figure 1: Overview bar chart
def fig1_overview(df):
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = df["turing_type"].map(TYPE_COLORS).fillna("#aaaaaa")

    ax.bar(
        range(len(df)),
        df["rob_shaberi_total"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.75,
    )

    # x-axis labels: strip the "3954_" prefix so they're shorter
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        df["config_name"].str.replace("3954_", "", regex=False),
        rotation=55,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Robustness Score (in %)", fontsize=11)
    ax.set_title(
        "Robustness of Topologies in Topology3954 Dataset\n(1 million simulations per topology)",
        fontsize=12, loc="left", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # Dashed vertical lines between topology groups
    topos = df["topology"].values
    for i in range(1, len(topos)):
        if topos[i] != topos[i - 1]:
            ax.axvline(i - 0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    # Legend
    handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax.legend(handles=handles, title="Turing Type", frameon=False)

    fig.tight_layout()
    save(fig, "3954_fig1_overview_bar_detail")

df = load_data()
fig1_overview(df)