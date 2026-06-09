from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import random
import numpy as np
import ast
import matplotlib.lines as mlines

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

CSV = "3954_PREFINAL_lhs_results_summary.csv"

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
    "Type1": "#4550AE",
    "Type2": "#D12F85",
    "Type3": "#287836",
} 

def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.png")


def load_data():
    df = pd.read_csv(CSV)

    # Extract topology (DCC / CDD / CCD …) and Turing type (Type1/2/3)
    df["topology"]    = df["config_name"].str.extract(r"NEW_LHS_3954_([A-Z]+)_")
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
        df["config_name"].str.replace("NEW_LHS_3954_", "", regex=False),
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
    save(fig, "new_3954_lhs_fig1_overview_bar_detail")



##################################### FIGURE 2: Scatter of Type 1 to Type 3 #####################################

def fig2_dotplot(df):
    random.seed(42)

    fig, ax = plt.subplots(figsize=(7, 4))

    types = ["Type1", "Type2", "Type3"]

    for i, t in enumerate(types):
        subset = df[df["turing_type"] == t]
        for _, row in subset.iterrows():
            marker = "^" if "Unequal" in row["config_name"] else "o"
            jitter = i + random.uniform(-0.2, 0.2)
            ax.scatter(
                jitter,
                row["rob_shaberi_total"],
                color=TYPE_COLORS[t],
                marker=marker,
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

    handles = [
        mlines.Line2D([], [], color="#313131", marker="o", linestyle="None", markersize=7, markeredgecolor="white", label="Equal"),
        mlines.Line2D([], [], color="#313131", marker="^", linestyle="None", markersize=7, markeredgecolor="white", label="Unequal"),
    ]

    ax.legend(handles=handles, title="Diffusion", frameon=False, loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)

    fig.tight_layout()
    save(fig, "new_3954_lhs_fig2_dotplot")




def fig2_raincloud(df):
    from scipy.stats import gaussian_kde

    random.seed(42)
    types  = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]

    fig, ax = plt.subplots(figsize=(9.5, 6))

    for i, (t, label) in enumerate(zip(types, labels)):
        subset = df[df["turing_type"] == t]["rob_shaberi_total"].dropna().values

        if len(subset) < 2:
            continue

        color = TYPE_COLORS[t]

        # ── Half violin (left side) ───────────────────────────────────────────
        #kde    = gaussian_kde(subset, bw_method=0.4)
        kde    = gaussian_kde(subset, bw_method=0.3)
        #y_range = np.linspace(subset.min(), subset.max(), 200)
        y_range = np.linspace(subset.min() - subset.std()*0.5, subset.max() + subset.std()*0.5, 200)
        kde_vals = kde(y_range)
        kde_vals = kde_vals / kde_vals.max() * 0.35  # normalise width

        ax.fill_betweenx(y_range, i - kde_vals, i, color=color, alpha=1.0, linewidth=0)

        # ── Mean line ─────────────────────────────────────────────────────────

        mean_val     = subset.mean()
        kde_at_mean  = kde(mean_val)[0]
        kde_at_mean  = kde_at_mean / kde_vals.max() * 0.35  # same normalisation as the cloud
        
        ax.hlines(subset.mean(), i - 0.35, i, color="white", linewidth=1.5, zorder=4)
        #ax.hlines(subset.mean(), i - 0.33, i, color="black", linewidth=1.5, zorder=4)

        # ── Jittered dots (right side) ────────────────────────────────────────
        for val in subset:
            jitter = i + random.uniform(0.08, 0.35)
            marker = "^" if any(
                "Unequal" in row["config_name"]
                for _, row in df[(df["turing_type"] == t) &
                                 (df["rob_shaberi_total"] == val)].iterrows()
            ) else "o"
            ax.scatter(jitter, val, color=color, marker=marker, s=95, edgecolors="white", linewidths=0.4, zorder=3)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Robustness Score (rob_shaberi_total)", fontsize=12)
    ax.set_title(
        "Robustness distribution by Turing Type for #3954\n"
        "(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=13, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    #ax.yaxis.grid(False)
    ax.set_xlim(-0.5, len(types) - 0.5)

    # Legend
    handles = [
        mlines.Line2D([], [], color="#313131", marker="o", linestyle="None", markersize=8, markeredgecolor="white", label="Equal"),
        mlines.Line2D([], [], color="#313131", marker="^", linestyle="None", markersize=8, markeredgecolor="white", label="Unequal"),
    ]

    ax.legend(handles=handles, title="Diffusion", frameon=False, loc="center right", bbox_to_anchor=(1.2, 0.5), ncol=1)

    fig.tight_layout()
    save(fig, "new_3954_lhs_fig2_raincloud")


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
    save(fig, "new_3954_lhs_fig3_grouped_topology")



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
    save(fig, "new_3954_lhs_fig4_diego_vs_shaberi")












########### RUN THE WHOLE THING ############

df = load_data()
fig1_overview(df)
fig2_dotplot(df)
fig2_raincloud(df)
fig3_grouped_topology(df)
fig4_diego_vs_shaberi(df)