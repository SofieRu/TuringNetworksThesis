import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CSV     = "3954_results_summary_curated_1mio.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "Type1": "#4C72B0",
    "Type2": "#DD8452",
    "Type3": "#55A868",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → plots/{name}.svg / .png")


def load_data():
    df = pd.read_csv(CSV)

    # Extract topology (DCC / CDD / CCD …) and Turing type (Type1/2/3)
    df["topology"]    = df["config_name"].str.extract(r"3954_([A-Z]+)_")
    df["turing_type"] = df["config_name"].str.extract(r"(Type[123])")

    return df


# ── Figure 1: Overview bar chart ──────────────────────────────────────────────
def fig1_overview(df):
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = df["turing_type"].map(TYPE_COLORS).fillna("#aaaaaa")

    ax.bar(
        range(len(df)),
        df["rob_shaberi_total"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.8,
    )

    # x-axis labels: strip the "3954_" prefix so they're shorter
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        df["config_name"].str.replace("3954_", "", regex=False),
        rotation=55,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Robustness  (rob_shaberi_total)", fontsize=11)
    ax.set_title(
        "Robustness varies strongly across configurations Type 2/3 outperform Type 1",
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
    save(fig, "fig1_overview_bar")


# ── Run ───────────────────────────────────────────────────────────────────────
df = load_data()
fig1_overview(df)