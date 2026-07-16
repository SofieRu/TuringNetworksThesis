from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import gaussian_kde

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a

CSV     = "3954_PREFINAL_rmt_results_summary.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

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

TYPE_COLORS = {"Type1": 'lightseagreen', "Type2": 'teal', "Type3": 'mediumpurple',} 

def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.png")


def load_data():
    df = pd.read_csv(CSV)
    df = df[~df["config_name"].str.contains("Control")]
    df["topology"]    = df["config_name"].str.extract(r"NEW_RMT_3954_([A-Z]+)_")
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
    save(fig, "test_3954_rmt_fig1_sigma_vs_robustness_all")





########## FIGURE 4: Dot plot – Robustness by Turing Type at fixed sigma ##########
 
def fig4_dotplot_fixed_sigma(df, sigma_val=0.4):
    random.seed(42)
 
    subset = df[df["sigma"] == sigma_val].copy()
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    types = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]
    for i, t in enumerate(types):
        t_data = subset[subset["turing_type"] == t]["rob_shaberi_type_I"]  # CHANGED from "rob_shaberi_total" to "rob_shaberi_type_I"!! CHANGE BACK LATER IF NEEDED
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
    ax.set_ylabel("Robustness (rob_shaberi_type_I)", fontsize=11)
    ax.set_title(
        f"Topology #3954 RMT – Robustness by Turing Type at σ = {sigma_val}",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    ax.set_xlim(-0.5, 2.5)
 
    fig.tight_layout()
    save(fig, f"test_3954_rmt_fig4_dotplot_sigma{sigma_val}")





def fig4_dotplot_fixed_sigma(df, sigma_val):
    random.seed(42)
    # Filter for the specific sigma value
    subset_df = df[df["sigma"] == sigma_val].copy()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    types = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]
    
    for i, t in enumerate(types):
        # Using "rob_shaberi_type_I" as requested
        t_data = subset_df[subset_df["turing_type"] == t]["rob_shaberi_total"].dropna().values
        
        if len(t_data) < 2:
            continue
            
        color = TYPE_COLORS[t]
        
        # ── Half violin cloud (left side) ─────────────────────────────────────
        kde = gaussian_kde(t_data, bw_method=0.3)
        y_range = np.linspace(t_data.min() - t_data.std()*0.2, t_data.max() + t_data.std()*0.2, 200)
        kde_vals = kde(y_range)
        
        # Normalise width to fit cleanly beside the dots
        max_kde = kde_vals.max() if kde_vals.max() > 0 else 1
        kde_vals = kde_vals / max_kde * 0.25 
        
        ax.fill_betweenx(y_range, i - kde_vals, i, color=color, alpha=1.0, linewidth=0, zorder=2)
        
        # ── Mean line (left side, over the violin) ────────────────────────────
        mean_val = t_data.mean()
        ax.hlines(mean_val, i - 0.25, i, color="white", linewidth=1.5, zorder=4)
        
        # ── Jittered dots (right side) ────────────────────────────────────────
        # Shifted positive (i + offset) to occupy the right side of the axis entry
        jitter = [i + random.uniform(0.05, 0.25) for _ in t_data]
        
        ax.scatter(
            jitter, 
            t_data, 
            color=color, 
            s=80, 
            edgecolors="white", 
            linewidths=0.5,
            zorder=3
        )
         
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Robustness Score (in %)", fontsize=11)
    ax.set_title(
        f"Topology #3954 RMT, Robustness by Turing Type at σ = {sigma_val}", 
        fontsize=12, 
        loc="left", 
        pad=10,
    )
    ax.xaxis.grid(False)
    ax.set_xlim(-0.5, 2.5)
    fig.tight_layout()
    
    save(fig, f"test_3954_rmt_dotplot_sigma{sigma_val}")






########### RUN THE WHOLE THING ############

df = load_data()
fig1_sigma_vs_robustness(df)
fig4_dotplot_fixed_sigma(df, sigma_val=0.6)






########## FIGURE 2: Corrected robustness vs Sigma, coloured by type ##########

# def fig2_corrected_robustness(df):
#     df = df.copy()
#     df["rob_corrected"] = df["shaberi_total"] / df["n_samples"]

#     fig, ax_stable = plt.subplots(figsize=(12, 6))

#     # Left y-axis: stable count (dashed black)
#     stable = df.groupby("sigma")["stable_without_diffusion"].first()
#     ax_stable.plot(stable.index, stable.values, color="black", linewidth=1.5, linestyle="--", label="Stable steady state count")
#     ax_stable.set_xlabel("Sigma (σ)", fontsize=11)
#     ax_stable.set_ylabel("Number of stable steady states", fontsize=11)
#     ax_stable.xaxis.grid(False)

#     # Right y-axis: corrected robustness, coloured by Turing type
#     ax_rob = ax_stable.twinx()
#     for t in ["Type1", "Type2", "Type3"]:
#         subset = df[df["turing_type"] == t]
#         if subset.empty:
#             continue
#         mean_rob = subset.groupby("sigma")["rob_corrected"].mean()
#         ax_rob.plot(mean_rob.index, mean_rob.values, color=TYPE_COLORS[t],
#                     linewidth=2, label=t)
#     ax_rob.set_ylabel("Corrected robustness (shaberi_total / n_samples)", fontsize=11)
#     ax_rob.spines["right"].set_visible(True)
#     ax_rob.spines["top"].set_visible(False)
#     ax_rob.yaxis.grid(False)
#     ax_rob.xaxis.grid(False)

#     ax_stable.set_title(
#         "Topology #3954 RMT – Corrected robustness (normalised by total samples) vs σ",
#         fontsize=12, loc="left", pad=10,
#     )

#     lines1, labels1 = ax_stable.get_legend_handles_labels()
#     lines2, labels2 = ax_rob.get_legend_handles_labels()
#     ax_stable.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

#     fig.tight_layout()
#     save(fig, "test_3954_rmt_fig2_corrected_robustness")
