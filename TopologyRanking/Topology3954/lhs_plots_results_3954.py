from pathlib import Path
import random
import ast
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

CSV = "3954_FINAL_lhs_results_summary.csv" 
# got rid of control: 
# NEW_LHS_3954_Type1_Control_Slow,0,0.1,0.1,0.1,1000000,969615,951296,0,0,0,0,0,0,0.0,0.0,0.0
# NEW_LHS_3954_Type1_Control_Fast,2,10.0,10.0,10.0,1000000,969615,951296,0,0,0,0,0,0,0.0,0.0,0.0

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

#TYPE_COLORS = {"Type1": 'lightseagreen', "Type2": 'teal', "Type3": 'mediumpurple',} 
TYPE_COLORS = {"Type1": "#2E9F6E", "Type2": "#2B72DB", "Type3": "#E34D93"}


def save(fig, name):
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to plots/{name}.png")


# def load_data():
#     df = pd.read_csv(CSV)
#     df["topology"]    = df["config_name"].str.extract(r"FINAL_LHS_3954_([A-Z]+)_")
#     df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")
#     return df

def load_data():
    df = pd.read_csv(CSV)
    df = df[~df["config_name"].str.contains('Lab', na=False)]
    df["topology"]    = df["config_name"].str.extract(r"FINAL_LHS_3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")
    return df


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
 
    ax.set_xlabel("Robustness Score using Characteristic Polynomial (Diego)", fontsize=12)
    ax.set_ylabel("Robustness Score using Eigenvalues (Shaberi)", fontsize=12)
    ax.set_title(
        "Robustness Scores Diego vs Shaberi for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
        fontsize=12, loc="left", pad=10,
    )
    ax.legend(title="Turing Type", frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
 
    fig.tight_layout()
    save(fig, "new_3954_lhs_fig4_diego_vs_shaberi")


# PLOT FOR THESIS
def fig_combined_overview_and_raincloud(df):
    df = df.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8,6.6)) #previously 14,10, size has to be figsize=(6.3, 5.4) to match dina4 age but liek its sooo small

    # PANEL 1: OVERVIEW BAR CHART (ax1)
    colors = df["turing_type"].map(TYPE_COLORS).fillna("#aaaaaa")

    ax1.bar(range(len(df)), df["rob_shaberi_total"], color=colors, edgecolor="white", linewidth=0.5, width=0.76,)
    ax1.set_xticks(range(len(df)))

    #ax1.set_xticklabels(df["config_name"].str.replace(r"FINAL_LHS_3954_|Type\d*_", "", regex=True),rotation=55,ha="right",fontsize=8,)
    ax1.set_xticklabels(df["config_id"], rotation=20, ha="right", fontsize=10)
    ax1.set_xlabel("ID of Diffusion Configurations", fontsize=12)

    ax1.set_ylabel("Robustness Score (in %)", fontsize=12.5, labelpad=10)
    ax1.set_title("Latin Hypercube Sampling Results, 1 million simulations\nRobustness of different diffusion rate configurations for Topology #3954",fontsize=14,loc="center",pad=10,)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(-0.5, len(df) - 0.5)
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(True)

    # PANEL 2: RAINCLOUD DISTRIBUTION (ax2)
    random.seed(42)
    types = ["Type1", "Type2", "Type3"]
    labels = ["Type 1", "Type 2", "Type 3"]

    for i, (t, label) in enumerate(zip(types, labels)):
        subset = (df[df["turing_type"] == t]["rob_shaberi_total"].dropna().values)

        color = TYPE_COLORS[t]
        # violin distribution cloud
        kde = gaussian_kde(subset, bw_method=0.3)
        y_range = np.linspace(subset.min() - subset.std()*0.3, subset.max() + subset.std()*0.3, 200)
        kde_vals = kde(y_range)
        kde_vals = kde_vals / kde_vals.max() * 0.35
        ax2.fill_betweenx(y_range, i - kde_vals, i, color=color, alpha=1.0, linewidth=0)

        # mean bar inside the cloud
        mean_val = subset.mean() # new
        closest_idx = np.argmin(np.abs(y_range - mean_val))
        kde_at_mean = kde_vals[closest_idx]
        ax2.hlines(mean_val, i - kde_at_mean, i, color="black", linewidth=1.0, zorder=4) # linestyle="--",

        for val in subset:
            jitter = i + random.uniform(0.08, 0.35)
            marker = (
                "^"
                if any(
                    "Unequal" in row["config_name"]
                    for _, row in df[(df["turing_type"] == t) & (df["rob_shaberi_total"] == val)].iterrows())
                else "o"
            )
            ax2.scatter(jitter,val,color=color,marker=marker,s=120,edgecolors="white",linewidths=0.4,zorder=3,)

    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(labels, fontsize=12.5)
    ax2.set_ylabel("Robustness Score (in %)", fontsize=12.5, labelpad=10)
    ax2.set_title("Robustness distribution by Type for Topology #3954",fontsize=14,loc="center",pad=10,)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(True)
    ax2.set_xlim(-0.5, len(types) - 0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    # GLOBAL UNIFIED BOTTOM LEGENDS
    bar_handles =  [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    rain_handles = [mlines.Line2D([],[],color="#313131",marker="o",linestyle="None",markersize=8,markeredgecolor="white",label="Equal Diffusion",),
                    mlines.Line2D([],[],color="#313131",marker="^",linestyle="None",markersize=8,markeredgecolor="white",label="Unequal Diffusion",),]

    fig.legend(handles=bar_handles,frameon=False,loc="lower center",bbox_to_anchor=(0.35, 0.04),ncol=3,fontsize=12)
    fig.legend(handles=rain_handles,frameon=False,loc="lower center",bbox_to_anchor=(0.7, 0.04),ncol=2,fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.96, bottom=0.14, hspace=0.4)
    save(fig, "final_3954_lhs_overview")


########### RUN THE WHOLE THING ############

df = load_data()
# fig4_diego_vs_shaberi(df)
fig_combined_overview_and_raincloud(df) 






##################################### FIGURE 1: Overview bar chart #####################################

# def fig1_overview(df):
#     fig, ax = plt.subplots(figsize=(14, 6))
#     #colors = df["turing_type_I"].map(TYPE_COLORS).fillna("#aaaaaa")
#     ax.bar(
#         range(len(df)),
#         df["rob_shaberi_type_I"], # rob_shaberi_total
#         #color=colors,
#         edgecolor="white",
#         linewidth=0.5,
#         width=0.75,
#     )
#     # x-axis labels: strip prefix and "Type" variations
#     ax.set_xticks(range(len(df)))
#     ax.set_xticklabels(
#         # The | means OR, and \d* means match zero or more digits (like 1, 2, 3)
#         df["config_name"].str.replace(r"FINAL_LHS_3954_|Type\d*_", "", regex=True),
#         rotation=55,
#         ha="right",
#         fontsize=8,
#     )

#     ax.set_ylabel("Robustness Score (in %)", fontsize=11)
#     ax.set_title(
#         "Robustness of Topologies for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
#         fontsize=12, loc="left", pad=10,
#     )
#     ax.spines[["top", "right"]].set_visible(False)
#     # fix 1 – remove extra left padding
#     ax.set_xlim(-0.5, len(df) - 0.5)
#     # fix 2 – horizontal grid lines only
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(True)

#     # legend
#     handles = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
#     #ax.legend(handles=handles, title="Turing Type", frameon=False)
#     ax.legend(handles=handles, title="Turing Type", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3)

#     fig.tight_layout()
#     save(fig, "testttt_3954_lhs_fig1_overview_bar_detail")



##################################### FIGURE 2: Scatter of Type 1 to Type 3 #####################################

# def fig2_dotplot(df):
#     random.seed(42)
#     fig, ax = plt.subplots(figsize=(7, 4))
#     types = ["Type1", "Type2", "Type3"]
#     for i, t in enumerate(types):
#         subset = df[df["turing_type"] == t]
#         for _, row in subset.iterrows():
#             marker = "^" if "Unequal" in row["config_name"] else "o"
#             jitter = i + random.uniform(-0.2, 0.2)
#             ax.scatter(
#                 jitter,
#                 row["rob_shaberi_total"],
#                 color=TYPE_COLORS[t],
#                 marker=marker,
#                 s=80,
#                 edgecolors="white",
#                 linewidths=0.5,
#                 zorder=3,
#             )

#     ax.set_xticks([0, 1, 2])
#     ax.set_xticklabels(["Type 1", "Type 2", "Type 3"], fontsize=11)
#     ax.set_ylabel("Robustness Score (in %, Shaberi Method)", fontsize=11)
#     ax.set_title(
#         "Robustness Scores by Turing Type for #3954\n(Latin Hypercube Sampling, 1 million simulations)",
#         fontsize=12, loc="left", pad=10,
#     )
#     ax.xaxis.grid(False)
#     ax.set_xlim(-0.5, 2.5)

#     handles = [
#         mlines.Line2D([], [], color="#313131", marker="o", linestyle="None", markersize=7, markeredgecolor="white", label="Equal"),
#         mlines.Line2D([], [], color="#313131", marker="^", linestyle="None", markersize=7, markeredgecolor="white", label="Unequal"),
#     ]

#     ax.legend(handles=handles, title="Diffusion", frameon=False, loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)
#     fig.tight_layout()
#     save(fig, "new_3954_lhs_fig2_dotplot")


# def fig2_raincloud(df):
#     random.seed(42)
#     types  = ["Type1", "Type2", "Type3"]
#     labels = ["Type 1", "Type 2", "Type 3"]
#     fig, ax = plt.subplots(figsize=(10, 6))

#     for i, (t, label) in enumerate(zip(types, labels)):
#         subset = df[df["turing_type"] == t]["rob_shaberi_total"].dropna().values

#         if len(subset) < 2:
#             continue

#         color = TYPE_COLORS[t]

#         # ── Half violin (left side) ───────────────────────────────────────────
#         #kde    = gaussian_kde(subset, bw_method=0.4)
#         kde    = gaussian_kde(subset, bw_method=0.3)
#         #y_range = np.linspace(subset.min(), subset.max(), 200)
#         y_range = np.linspace(subset.min() - subset.std()*0.2, subset.max() + subset.std()*0.2, 200)
#         kde_vals = kde(y_range)
#         kde_vals = kde_vals / kde_vals.max() * 0.35  # normalise width

#         ax.fill_betweenx(y_range, i - kde_vals, i, color=color, alpha=1.0, linewidth=0)

#         # ── Mean line ─────────────────────────────────────────────────────────

#         mean_val     = subset.mean()
#         kde_at_mean  = kde(mean_val)[0]
#         kde_at_mean  = kde_at_mean / kde_vals.max() * 0.35  # same normalisation as the cloud
        
#         ax.hlines(subset.mean(), i - 0.35, i, color="white", linewidth=1.5, zorder=4)
#         #ax.hlines(subset.mean(), i - 0.33, i, color="black", linewidth=1.5, zorder=4)

#         # ── Jittered dots (right side) ────────────────────────────────────────
#         for val in subset:
#             jitter = i + random.uniform(0.08, 0.35)
#             marker = "^" if any(
#                 "Unequal" in row["config_name"]
#                 for _, row in df[(df["turing_type"] == t) &
#                                  (df["rob_shaberi_total"] == val)].iterrows()
#             ) else "o"
#             ax.scatter(jitter, val, color=color, marker=marker, s=95, edgecolors="white", linewidths=0.4, zorder=3)

#     ax.set_xticks(range(len(types)))
#     ax.set_xticklabels(labels, fontsize=11)
#     ax.set_ylabel("Robustness Score (rob_shaberi_total)", fontsize=12)
#     ax.set_title(
#         "Robustness distribution by Turing Type for #3954\n"
#         "(Latin Hypercube Sampling, 1 million simulations)",
#         fontsize=12.5, loc="left", pad=10,
#     )
#     ax.xaxis.grid(False)
#     #ax.yaxis.grid(False)
#     ax.set_xlim(-0.5, len(types) - 0.5)

#     # Legend
#     handles = [
#         mlines.Line2D([], [], color="#313131", marker="o", linestyle="None", markersize=8, markeredgecolor="white", label="Equal"),
#         mlines.Line2D([], [], color="#313131", marker="^", linestyle="None", markersize=8, markeredgecolor="white", label="Unequal"),
#     ]

#     ax.legend(handles=handles, title="Diffusion", frameon=False, loc="center right", bbox_to_anchor=(1.2, 0.5), ncol=1)

#     fig.tight_layout()
#     save(fig, "new_3954_lhs_fig2_raincloud")

# fig1_overview(df)
# fig2_dotplot(df)
# fig2_raincloud(df)