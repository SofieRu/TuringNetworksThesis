from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CSV     = "3954_rmt_results_prelim_100k.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

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
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.svg / .png")


def load_data():
    df = pd.read_csv(CSV)
    df = df[~df["config_name"].str.contains("OneFast|Control|Limit")]
    df["topology"]    = df["config_name"].str.extract(r"RMT_3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")
    return df


########## FIGURE 1: Robustness vs Sigma overview ##########

def fig1_sigma_vs_robustness(df):
    fig, ax_stable = plt.subplots(figsize=(12, 6))

    # Left y-axis: stable count (dashed black)
    stable = df.groupby("sigma")["stable_without_diffusion"].first()
    ax_stable.plot(stable.index, stable.values, color="black", linewidth=1.5, linestyle="--", label="Stable steady state count")
    ax_stable.set_xlabel("Sigma (σ)", fontsize=11)
    ax_stable.set_ylabel("Number of stable steady states", fontsize=11)
    ax_stable.xaxis.grid(False)

    # Right y-axis: all configs in one colour, label only on first
    ax_rob = ax_stable.twinx()
    for i, cfg in enumerate(df["config_name"].unique()):
        subset = df[df["config_name"] == cfg].sort_values("sigma")
        ax_rob.plot(
            subset["sigma"],
            subset["rob_shaberi_total"],
            color="#A325A9",
            linewidth=1,
            #alpha=0.6,     # keep there if you want: it shows where values cluster so it gets thicker if there is another value exactly like that bc then they are on top of each other but if we dont want thicker lines just remove it
            label="Robustness – all configurations" if i == 0 else "",
        )
    ax_rob.set_ylabel("Robustness (rob_shaberi_total)", fontsize=11)
    ax_rob.spines["right"].set_visible(True)
    ax_rob.spines["top"].set_visible(False)
    ax_rob.yaxis.grid(False)
    ax_rob.xaxis.grid(False)

    ax_stable.set_title(
        "Topology #3954 RMT – Sigma vs Robustness and Stability",
        fontsize=12, loc="left", pad=10,
    )

    lines1, labels1 = ax_stable.get_legend_handles_labels()
    lines2, labels2 = ax_rob.get_legend_handles_labels()
    ax_stable.legend(lines1 + lines2, labels1 + labels2, frameon=False,
                     loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    save(fig, "3954_rmt_fig1_sigma_vs_robustness_all")


########## FIGURE 2: Corrected robustness vs Sigma, coloured by type ##########

def fig2_corrected_robustness(df):
    df = df.copy()
    df["rob_corrected"] = df["shaberi_total"] / df["n_samples"]

    fig, ax_stable = plt.subplots(figsize=(12, 6))

    # Left y-axis: stable count (dashed black)
    stable = df.groupby("sigma")["stable_without_diffusion"].first()
    ax_stable.plot(stable.index, stable.values, color="black", linewidth=1.5, linestyle="--", label="Stable steady state count")
    ax_stable.set_xlabel("Sigma (σ)", fontsize=11)
    ax_stable.set_ylabel("Number of stable steady states", fontsize=11)
    ax_stable.xaxis.grid(False)

    # Right y-axis: corrected robustness, coloured by Turing type
    ax_rob = ax_stable.twinx()
    for t in ["Type1", "Type2", "Type3"]:
        subset = df[df["turing_type"] == t]
        if subset.empty:
            continue
        mean_rob = subset.groupby("sigma")["rob_corrected"].mean()
        ax_rob.plot(mean_rob.index, mean_rob.values, color=TYPE_COLORS[t],
                    linewidth=2, label=t)
    ax_rob.set_ylabel("Corrected robustness (shaberi_total / n_samples)", fontsize=11)
    ax_rob.spines["right"].set_visible(True)
    ax_rob.spines["top"].set_visible(False)
    ax_rob.yaxis.grid(False)
    ax_rob.xaxis.grid(False)

    ax_stable.set_title(
        "Topology #3954 RMT – Corrected robustness (normalised by total samples) vs σ",
        fontsize=12, loc="left", pad=10,
    )

    lines1, labels1 = ax_stable.get_legend_handles_labels()
    lines2, labels2 = ax_rob.get_legend_handles_labels()
    ax_stable.legend(lines1 + lines2, labels1 + labels2, frameon=False,
                     loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

    fig.tight_layout()
    save(fig, "3954_rmt_fig2_corrected_robustness")


########## FIGURE 3: Corrected robustness overview (all configs, one colour) ##########

def fig3_corrected_overview(df):
    df = df.copy()
    # df["rob_corrected"] = df["shaberi_total"] / df["n_samples"]

    fig, ax_stable = plt.subplots(figsize=(12, 6))

    # Left y-axis: stable count (dashed black)
    stable = df.groupby("sigma")["stable_without_diffusion"].first()
    ax_stable.plot(stable.index, stable.values, color="black", linewidth=1.5,
                   linestyle="--", label="Stable steady state count")
    ax_stable.set_xlabel("Sigma (σ)", fontsize=11)
    ax_stable.set_ylabel("Number of stable steady states", fontsize=11)
    ax_stable.xaxis.grid(False)

    # Right y-axis: all configs in one colour, label only on first
    # CHANGED subset[rob_corrected] to subset["shaberi_total"]!! CHANGE BACK LATER IF NEEDED
    ax_rob = ax_stable.twinx()
    for i, cfg in enumerate(df["config_name"].unique()):
        subset = df[df["config_name"] == cfg].sort_values("sigma")
        ax_rob.plot(
            subset["sigma"],
            subset["shaberi_total"],
            color="#A325A9",
            linewidth=1,
            #alpha=0.6,     # keep there if you want: it shows where values cluster so it gets thicker if there is another value exactly like that bc then they are on top of each other but if we dont want thicker lines just remove it
            label="Number of stable steady states for all configurations" if i == 0 else "",
        )
    ax_rob.set_ylabel("Number of Turing instabilities (shaberi_total)", fontsize=11)
    ax_rob.spines["right"].set_visible(True)
    ax_rob.spines["top"].set_visible(False)
    ax_rob.yaxis.grid(False)
    ax_rob.xaxis.grid(False)

    ax_stable.set_title(
        "Topology #3954 RMT Sigma vs Corrected Robustness overview (all configurations)",
        fontsize=12, loc="left", pad=10,
    )

    lines1, labels1 = ax_stable.get_legend_handles_labels()
    lines2, labels2 = ax_rob.get_legend_handles_labels()
    ax_stable.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    save(fig, "3954_rmt_fig3_corrected_overview")






########## FIGURE 4: Dot plot – Robustness by Turing Type at fixed sigma ##########
 
def fig4_dotplot_fixed_sigma(df, sigma_val=1.0):
    import random
    random.seed(42)
 
    subset = df[df["sigma"] == sigma_val].copy()
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    types = ["Type1", "Type2", "Type3"]
    for i, t in enumerate(types):
        t_data = subset[subset["turing_type"] == t]["rob_shaberi_total"]
        jitter = [i + random.uniform(-0.15, 0.15) for _ in t_data]
        ax.scatter(
            jitter,
            t_data,
            color=TYPE_COLORS[t],
            s=80,
            edgecolors="white",
            linewidths=0.5,
        )
 
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Type 1", "Type 2", "Type 3"], fontsize=11)
    ax.set_ylabel("Robustness (rob_shaberi_total)", fontsize=11)
    ax.set_title(
        f"Topology #3954 RMT – Robustness by Turing Type at σ = {sigma_val}",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    ax.set_xlim(-0.5, 2.5)
 
    fig.tight_layout()
    save(fig, f"3954_rmt_fig4_dotplot_sigma{sigma_val}")







########## FIGURE 5: Grouped topology bars – Max robustness per topology × type at fixed sigma ##########
 
def fig5_grouped_topology(df, sigma_val=1.0):

    # Option A: raw Shaberi count (absolute number of Turing instabilities)
    METRIC     = "shaberi_total"
    METRIC_LABEL = "Number of Turing instabilities (shaberi_total)"
 
    # Option B: inflated robustness score (shaberi / stable steady states)
    # METRIC     = "rob_shaberi_total"
    # METRIC_LABEL = "Robustness score (shaberi_total / stable)"
 
    # Option C: corrected robustness (shaberi / total samples, removes inflation)
    # df = df.copy()
    # df["rob_corrected"] = df["shaberi_total"] / df["n_samples"]
    # METRIC     = "rob_corrected"
    # METRIC_LABEL = "Corrected robustness (shaberi_total / n_samples)"
 
    subset = df[df["sigma"] == sigma_val].copy()
 
    topos = subset["topology"].dropna().unique()
    types = ["Type1", "Type2", "Type3"]
    x     = np.arange(len(topos))
    w     = 0.25
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    for i, t in enumerate(types):
        vals = [
            subset[(subset["topology"] == topo) & (subset["turing_type"] == t)][METRIC].max()
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
    ax.set_ylabel(METRIC_LABEL, fontsize=11)
    ax.set_title(
        f"Topology #3954 RMT Max robustness per topology and Turing Type at σ = {sigma_val}",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    ax.legend(title="Turing Type", frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
 
    fig.tight_layout()
    save(fig, f"3954_rmt_fig5_grouped_topology_sigma{sigma_val}")



########### RUN THE WHOLE THING ############

df = load_data()
fig1_sigma_vs_robustness(df)
fig2_corrected_robustness(df)
fig3_corrected_overview(df)
fig4_dotplot_fixed_sigma(df, sigma_val=1.0)
fig5_grouped_topology(df, sigma_val=1.0)