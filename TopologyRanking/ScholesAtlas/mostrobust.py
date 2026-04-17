from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

CSV_NON_COMP = "Scholes-2019-Suppl-3N2D-Non-Competitive.csv"
CSV_COMP     = "Scholes-2019-Suppl-3N2D-Competitive.csv"
OUT_DIR      = Path("plots")
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

def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → plots/{name}.svg / .png")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)
    return df

def fig1_top10_non_competitive(df):
    top10 = df.nlargest(10, "Total Robustness").reset_index(drop=True)
    top10.index += 1

    print("\nTop 10 topologies by Total Robustness (Non-Competitive):\n")
    print(top10[["ID", "Intracellular Robustness", "Extracellular Robustness",
                 "Topological Robustness", "Total Robustness"]].to_string())
    print()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(i) for i in top10["ID"]],
        top10["Total Robustness"],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
    )
    ax.set_xlabel("Topology ID", fontsize=11)
    ax.set_ylabel("Total Robustness", fontsize=11)
    ax.set_title(
        "Top 10 topologies by total robustness, Non-Competitive (Scholes 2019)",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    fig.tight_layout()
    save(fig, "scholes_top10_non_competitive")

def fig2_top10_competitive(df):
    top10 = df.nlargest(10, "Total Robustness").reset_index(drop=True)
    top10.index += 1

    print("\nTop 10 topologies by Total Robustness (Competitive):\n")
    print(top10[["ID", "Intracellular Robustness", "Extracellular Robustness",
                 "Topological Robustness", "Total Robustness"]].to_string())
    print()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(i) for i in top10["ID"]],
        top10["Total Robustness"],
        color="#DD8452",
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
    )
    ax.set_xlabel("Topology ID", fontsize=11)
    ax.set_ylabel("Total Robustness", fontsize=11)
    ax.set_title(
        "Top 10 topologies by total robustness, Competitive (Scholes 2019)",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
    fig.tight_layout()
    save(fig, "scholes_top10_competitive")

########### RUN THE WHOLE THING ############

df_non_comp = load_data(CSV_NON_COMP)
df_comp     = load_data(CSV_COMP)

fig1_top10_non_competitive(df_non_comp)
fig2_top10_competitive(df_comp)