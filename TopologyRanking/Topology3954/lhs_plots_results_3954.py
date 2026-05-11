from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import random
import numpy as np

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a, module load SciPy-bundle/2024.05-gfbf-2024a

# CSV = "3954_lhs_results_curated_1mio.csv"
CSV = "3954_lhs_results_final_1mio.csv"
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
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.png")


def load_data():
    df = pd.read_csv(CSV)

    # Extract topology (DCC / CDD / CCD …) and Turing type (Type1/2/3)
    df["topology"]    = df["config_name"].str.extract(r"3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")

    return df

##################################### FIGURE 1: Overview bar chart #####################################

def fig1_overview(df):
    fig, ax = plt.subplots(figsize=(14, 6))

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
        "Robustness of Topologies for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=12, loc="left", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # fix 1 – remove extra left padding
    ax.set_xlim(-0.5, len(df) - 0.5)

    # fix 2 – horizontal grid lines only
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    # Dashed vertical lines between topology groups
    topos = df["topology"].values
    for i in range(1, len(topos)):
        if topos[i] != topos[i - 1]:
            ax.axvline(i - 0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    # legend
    handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    #ax.legend(handles=handles, title="Turing Type", frameon=False)
    ax.legend(handles=handles, title="Turing Type", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3)

    fig.tight_layout()
    save(fig, "3954_lhs_fig1_overview_bar_detail")



##################################### FIGURE 2: Scatter of Type 1 to Type 3 #####################################


def fig2_dotplot(df):
    random.seed(42)
 
    fig, ax = plt.subplots(figsize=(7, 4))
 
    types = ["Type1", "Type2", "Type3"]
 
    for i, t in enumerate(types):
        subset = df[df["turing_type"] == t]["rob_shaberi_total"]
        # add small random jitter on x so dots don't stack
        jitter = [i + random.uniform(-0.15, 0.15) for _ in subset]
        ax.scatter(
            jitter,
            subset,
            color=TYPE_COLORS[t],
            s=80,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
 
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Type 1", "Type 2", "Type 3"], fontsize=11)
    ax.set_ylabel("Robustness Score (in %, Shaberi Method)", fontsize=11)
    ax.set_title(
        "Robustness Scores by Turing Type for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    ax.set_xlim(-0.5, 2.5)
 
    fig.tight_layout()
    save(fig, "3954_lhs_fig2_dotplot")




##################################### FIGURE 3: Grouped bars, max robustness per topology × type #####################################

def fig3_grouped_topology(df):
 
    fig, ax = plt.subplots(figsize=(9, 5))

    topos = df["topology"].dropna().unique()
    types = ["Type1", "Type2", "Type3"]
    x     = np.arange(len(topos))
    w     = 0.25
 
    for i, t in enumerate(types):
        vals = [
            df[(df["topology"] == topo) & (df["turing_type"] == t)]["rob_shaberi_total"].max()
            for topo in topos
        ]
        ax.bar(
            x + (i - 1) * w,
            vals,
            width=w,
            color=TYPE_COLORS[t],
            label=t,
            edgecolor="white",
            linewidth=0.5,
        )
 
    ax.set_xticks(x)
    ax.set_xticklabels(topos, fontsize=11)
    ax.set_ylabel("Max Robustness Score (in %, Shaberi Method)", fontsize=11)
    ax.set_title(
        "Max robustness for #3954 Topology\n(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    ax.legend(title="Turing Type", frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
 
    fig.tight_layout()
    save(fig, "3954_lhs_fig3_grouped_topology")



##################################### FIGURE 4: Scatter Plot Shaberi vs Diego Robustness across Type 1 to 3 #####################################

def fig4_diego_vs_shaberi(df):
    fig, ax = plt.subplots(figsize=(6, 5))
 
    for t in ["Type1", "Type2", "Type3"]:
        subset = df[df["turing_type"] == t]
        ax.scatter(
            subset["rob_diego"],
            subset["rob_shaberi_total"],
            color=TYPE_COLORS[t],
            label=t,
            s=80,
            edgecolors="white",
            linewidths=0.5,
        )
 
    lim = df[["rob_diego", "rob_shaberi_total"]].max().max() * 1.05 # 1:1 line
    ax.plot([0, lim], [0, lim], color="#444444", linewidth=0.8,
            linestyle="--", label="1:1 line")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
 
    ax.set_xlabel("Robustness Score using Characteristic Polynomial (Diego)", fontsize=11)
    ax.set_ylabel("Robustness Score using Eigenvalues (Shaberi)", fontsize=11)
    ax.set_title(
        "Robustness Scores Diego vs Shaberi for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=12, loc="left", pad=10,
    )
    ax.legend(title="Turing Type", frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
 
    fig.tight_layout()
    save(fig, "3954_lhs_fig4_diego_vs_shaberi")






########### RUN THE WHOLE THING ############

df = load_data()
fig1_overview(df)
fig2_dotplot(df)
fig3_grouped_topology(df)
fig4_diego_vs_shaberi(df)