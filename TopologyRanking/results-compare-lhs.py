from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

CSVS = {
    "#1754": "Topology1754/1754_PREFINAL_lhs_results_summary.csv",
    "#1823": "Topology1823/1823_PREFINAL_lhs_results_summary.csv",
    "#1838": "Topology1838/1838_PREFINAL_lhs_results_summary.csv",
    "#3954": "Topology3954/3954_FILTER_lhs_results_summary.csv",
}

PARAMS_CSV = "Topology3954/3954_FILTER_lhs_results_parameters.csv"

# for both 3954 and 1754 i got rid of _CCD_Type1_Var1/2 bc the values were really high and kinda did not match the rest but later if it does match then we can put it back in and see if it changes the results

OUT_DIR = Path("ResultPlots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

TOPO_MARKERS = {
    "#1754": "o",
    "#1823": "s",
    "#1838": "^",
    "#3954": "D",
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
        #parsed = df["diffusion"].apply(parse_diff)
        #df["dU"] = parsed.apply(lambda d: d.get("dU"))
        #df["dV"] = parsed.apply(lambda d: d.get("dV"))
        #df["dW"] = parsed.apply(lambda d: d.get("dW"))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def parse_diff(s):
    try:
        return ast.literal_eval(s.replace("\u2018", "'").replace("\u2019", "'"))
    except:
        return {"dU": None, "dV": None, "dW": None}






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
    save(fig, "new_compare_fig1_heatmap")










########## FIGURE 6: Stacked absolute bar – Type I vs II vs Hopf composition ##########

def fig6_pattern_composition(df):
    df = df.copy()

    # UNCOMMENT to exclude Unequal diffusion configurations
    #df = df[~df["config_name"].str.contains("Unequal")]

    topos = ["#1754", "#1823", "#1838", "#3954"]
    types = ["Type1", "Type2", "Type3"]
 
    PATTERN_COLORS = {
        "Type I"  : "#2770A0",
        "Type II" : "#9243A8",
        "Hopf"    : "#C7C93C",
        "Turing Filter" : "#58C675"
    }
 
    import matplotlib.patches as mpatches
 
    # aggregate: max robustness and pattern counts per topology × turing type
    grouped = df.groupby(["topology_id", "turing_type"]).agg(
        rob    = ("rob_shaberi_total", "max"),
        type_I = ("shaberi_type_I",   "sum"),
        type_II= ("shaberi_type_II",  "sum"),
        hopf   = ("shaberi_hopf",     "sum"),
        turing_filter = ("filter_count", "sum"),
    )
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    n_topos = len(topos)
    width   = 0.16
    gap     = 0.7
 
    for i, t in enumerate(types):
        for j, topo in enumerate(topos):
            try:
                row = grouped.loc[(topo, t)]
            except KeyError:
                continue
 
            rob   = row["rob"]
            #total = row["type_I"] + row["type_II"] + row["hopf"]
            total = row["type_I"] + row["type_II"] + row["hopf"] + row["turing_filter"]
            if total == 0 or rob == 0:
                continue
 
            # split robustness proportionally by pattern type
            f1 = row["type_I"]  / total * rob
            f2 = row["type_II"] / total * rob
            fh = row["hopf"]    / total * rob
            ft = row["turing_filter"] / total * rob
 
            x = i * gap + (j - n_topos / 2 + 0.5) * width
 
            ax.bar(x, f1,      width=width, color=PATTERN_COLORS["Type I"],  edgecolor="white", linewidth=0.3)
            ax.bar(x, f2,      width=width, color=PATTERN_COLORS["Type II"], edgecolor="white", linewidth=0.3, bottom=f1)
            ax.bar(x, fh,      width=width, color=PATTERN_COLORS["Hopf"],    edgecolor="white", linewidth=0.3, bottom=f1+f2)
            ax.bar(x, ft,      width=width, color=PATTERN_COLORS["Turing Filter"], edgecolor="white", linewidth=0.3, bottom=f1+f2+fh)

            ax.text(x, -0.019, topo, ha="right", va="top", fontsize=7, rotation=45)
 
    ax.set_xticks([i * gap for i in range(len(types))])
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("Robustness (rob_shaberi_total)", fontsize=11)
    ax.set_title(
        "Type I is mainly in Turing Type 1, Type II and Hopf drive Type 2/3 robustness",
        fontsize=12, loc="left", pad=10,
    )
    ax.xaxis.grid(False)
 
    handles = [mpatches.Patch(color=c, label=l) for l, c in PATTERN_COLORS.items()]
    #ax.legend(handles=handles, title="Pattern type", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.legend(handles=handles, title="Pattern type", frameon=False, loc="center right", bbox_to_anchor=(1.15, 0.5), ncol=1)
    
    fig.tight_layout()
    save(fig, "new_compare_fig6_pattern_composition")




def fig_pseudo_phase_3954(df):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    sub = df[df["topology_id"] == "#3954"].copy()

    cmap = cm.YlGnBu
    norm = mcolors.Normalize(
        vmin=sub["rob_shaberi_total"].min(),
        vmax=sub["rob_shaberi_total"].max(),
    )

    pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (xvar, yvar) in zip(axes, pairs):
        sc = ax.scatter(
            sub[xvar],
            sub[yvar],
            c=sub["rob_shaberi_total"],
            cmap=cmap,
            norm=norm,
            s=300,
            edgecolors="#444444",
            linewidths=0.5,
        )
        ax.set_xlabel(xvar, fontsize=11)
        ax.set_ylabel(yvar, fontsize=11)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.xaxis.grid(False)
        ax.set_title(f"{xvar} vs {yvar}", fontsize=11)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Robustness (rob_shaberi_total)", fontsize=10)

    fig.suptitle(
        "Topology #3954 – Pseudo phase diagram across diffusion rate combinations",
        fontsize=12, x=0.01, ha="left",
    )

    fig.tight_layout()
    save(fig, "new_3954_pseudo_phase_diagram")





############

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# Add this near the top with other file paths
PARAMS_CSV = "Topology3954/3954_FILTER_lhs_results_parameters.csv"

# Add this function with your other plotting functions

# def fig_3d_parameter_space():
#     # Load detailed parameter data
#     df_params = pd.read_csv(PARAMS_CSV)
    
#     # Choose which config(s) to plot
#     # Option 1: Plot just one config (the most robust)
#     config_to_plot = 13  # Your most robust config
#     df_plot = df_params[df_params['config_id'] == config_to_plot]
    
#     # Extract the 3 production rates
#     beta_u = df_plot['beta_u'].values
#     beta_v = df_plot['beta_v'].values
#     beta_w = df_plot['beta_w'].values
    
#     print(f"\nConfig {config_to_plot}: {len(df_plot)} points")
#     print(f"  beta_u range: [{beta_u.min():.2f}, {beta_u.max():.2f}]")
#     print(f"  beta_v range: [{beta_v.min():.2f}, {beta_v.max():.2f}]")
#     print(f"  beta_w range: [{beta_w.min():.2f}, {beta_w.max():.2f}]")
    
#     # Create 3D plot
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Scatter plot of all Turing parameter sets
#     scatter = ax.scatter(beta_u, beta_v, beta_w,
#                         c=df_plot['max_growth_rate'],  # Color by Turing strength
#                         cmap='YlOrRd',
#                         s=50,
#                         alpha=0.6,
#                         edgecolors='k',
#                         linewidth=0.5)
    
#     # Optional: Draw convex hull if you have enough points (>10)
#     if len(df_plot) > 10:
#         points = np.column_stack([beta_u, beta_v, beta_w])
#         try:
#             hull = ConvexHull(points)
#             # Plot hull surface
#             for simplex in hull.simplices:
#                 triangle = points[simplex]
#                 ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
#                                color='orange', alpha=0.2, linewidth=0)
#         except:
#             print("  (Not enough points for convex hull)")
    
#     # Labels and styling
#     ax.set_xlabel('β$_u$ (u production rate)', fontsize=12, labelpad=10)
#     ax.set_ylabel('β$_v$ (v production rate)', fontsize=12, labelpad=10)
#     ax.set_zlabel('β$_w$ (w production rate)', fontsize=12, labelpad=10)
    
#     config_name = df_plot['config_name'].iloc[0]
#     ax.set_title(f'Turing Parameter Space: {config_name}\n'
#                  f'(dU={df_plot["dU"].iloc[0]}, dV={df_plot["dV"].iloc[0]}, dW={df_plot["dW"].iloc[0]})',
#                  fontsize=13, pad=20)
    
#     # Colorbar
#     cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
#     cbar.set_label('Max Growth Rate (Turing strength)', fontsize=11)
    
#     # Better viewing angle
#     ax.view_init(elev=20, azim=45)
    
#     save(fig, "fig_3d_turing_island_single_config")


# think about maybe comparing different parametres so not necessarily beta u, beta v and beta w
def fig_3d_parameter_space_comparison():
    df_params = pd.read_csv(PARAMS_CSV)
    
    # Choose two configs to compare (e.g., low vs high robustness)
    config_a = 4    # type I config 
    config_b = 32   # type II config
    config_c = 50   # type III config
    
    df_a = df_params[df_params['config_id'] == config_a] # type I config 
    df_b = df_params[df_params['config_id'] == config_b] # type II config
    df_c = df_params[df_params['config_id'] == config_c] # type III config
    
    # Create plot
    fig = plt.figure(figsize=(18, 6))
    
    # Plot A
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(df_a['beta_u'], df_a['beta_v'], df_a['beta_w'],
               c='blue', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_a}')
    ax1.set_xlabel('β$_u$', fontsize=11)
    ax1.set_ylabel('β$_v$', fontsize=11)
    ax1.set_zlabel('β$_w$', fontsize=11)
    ax1.set_title(f'{df_a["config_name"].iloc[0]}\n{len(df_a)} Turing sets', fontsize=12)
    ax1.view_init(elev=20, azim=45)
    
    # Plot B
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(df_b['beta_u'], df_b['beta_v'], df_b['beta_w'],
               c='fuchsia', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_b}')
    ax2.set_xlabel('β$_u$', fontsize=11)
    ax2.set_ylabel('β$_v$', fontsize=11)
    ax2.set_zlabel('β$_w$', fontsize=11)
    ax2.set_title(f'{df_b["config_name"].iloc[0]}\n{len(df_b)} Turing sets', fontsize=12)
    ax2.view_init(elev=20, azim=45)

    # Plot C
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(df_c['beta_u'], df_c['beta_v'], df_c['beta_w'],
               c='green', s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
               label=f'Config {config_c}')
    ax3.set_xlabel('β$_u$', fontsize=11)
    ax3.set_ylabel('β$_v$', fontsize=11)
    ax3.set_zlabel('β$_w$', fontsize=11)
    ax3.set_title(f'{df_c["config_name"].iloc[0]}\n{len(df_c)} Turing sets', fontsize=12)
    ax3.view_init(elev=20, azim=45)
    
    plt.suptitle('Turing Parameter Space: Diffusion Configuration Comparison', 
                 fontsize=14, y=0.98)
    
    save(fig, "new_fig_3d_turing_island_comparison")










########### RUN THE WHOLE THING ############

df = load_all()
fig1_heatmap(df)
fig6_pattern_composition(df)
fig_pseudo_phase_3954(df)

# NEW: Add 3D parameter space plots
# fig_3d_parameter_space()
fig_3d_parameter_space_comparison()

# fig_phase_interpolated_3954(df)
# fig_ratio_sensitivity(df)
# fig_3d_scatter(df)






###############################################

# PLOTS THAT LOOK COOL BUT MIGHT NOT USE

# def fig_phase_interpolated_3954(df):
#     from scipy.interpolate import griddata
#     import matplotlib.cm as cm
#     import numpy as np

#     sub = df[df["topology_id"] == "#3954"].dropna(subset=["dU", "dV", "dW"]).copy()

#     pairs = [("dU", "dV"), ("dU", "dW"), ("dV", "dW")]
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for ax, (xvar, yvar) in zip(axes, pairs):
#         x   = sub[xvar].values
#         y   = sub[yvar].values
#         rob = sub["rob_shaberi_total"].values

#         # grid for interpolation — use log spacing to match symlog axis
#         # but handle 0 separately
#         x_vals = np.unique(x)
#         y_vals = np.unique(y)
#         xi = np.linspace(x.min(), x.max(), 200)
#         yi = np.linspace(y.min(), y.max(), 200)
#         xi_grid, yi_grid = np.meshgrid(xi, yi)

#         # interpolate onto grid
#         zi = griddata(
#             points=(x, y),
#             values=rob,
#             xi=(xi_grid, yi_grid),
#             method="cubic",     # smooth interpolation
#             fill_value=0,
#         )
#         zi = np.clip(zi, 0, None)  # no negative values

#         # filled contour plot
#         cf = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap="YlGnBu")
#         # overlay actual data points
#         ax.scatter(x, y, c=rob, cmap="YlGnBu",
#                    edgecolors="#444444", linewidths=0.8, s=80, zorder=3,
#                    vmin=rob.min(), vmax=rob.max())

#         ax.set_xlabel(xvar, fontsize=11)
#         ax.set_ylabel(yvar, fontsize=11)
#         ax.set_title(f"{xvar} vs {yvar}", fontsize=11)
#         ax.xaxis.grid(False)

#     cbar = fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.02)
#     cbar.set_label("Robustness (rob_shaberi_total)", fontsize=10)

#     fig.suptitle(
#         "Topology #3954 – Interpolated phase diagram across diffusion rate combinations\n"
#         "(note: surface interpolated between discrete measurements)",
#         fontsize=11, x=0.01, ha="left",
#     )

#     fig.tight_layout()
#     save(fig, "3954_phase_interpolated")




# def fig_ratio_sensitivity(df):
#     import matplotlib.lines as mlines

#     df = df[df["topology_id"].isin(["#3954", "#1754"])].copy()

#     # only Type3 configs (dW = 0, two immobile)
#     df = df[df["turing_type"] == "Type3"]

#     # classify diffusion ratio from config name
#     def ratio_label(name):
#         if "Unequal1" in name:
#             return "Unequal1\n(10:1)"
#         elif "Unequal2" in name:
#             return "Unequal2\n(1:10)"
#         elif "Equal" in name:
#             return "Equal\n(1:1)"
#         else:
#             return None

#     df["ratio"] = df["config_name"].apply(ratio_label)
#     df = df.dropna(subset=["ratio"])

#     ratio_order = ["Equal\n(1:1)", "Unequal1\n(10:1)", "Unequal2\n(1:10)"]
#     topos = ["#3954", "#1754"]

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

#     for ax, topo in zip(axes, topos):
#         sub = df[df["topology_id"] == topo]

#         # one line per topology variant (e.g. DCC, CDD, CCD)
#         sub["topo_variant"] = sub["config_name"].str.extract(r"(?:LHS|RMT)_\d+_([A-Z]+)_")
#         variants = sub["topo_variant"].dropna().unique()

#         for var in variants:
#             var_data = sub[sub["topo_variant"] == var]
#             vals = [
#                 var_data[var_data["ratio"] == r]["rob_shaberi_total"].mean()
#                 for r in ratio_order
#             ]
#             ax.plot(
#                 ratio_order, vals,
#                 marker="o", linewidth=1.5,
#                 markersize=7, markeredgecolor="white", markeredgewidth=0.5,
#                 label=var,
#             )

#         ax.set_title(topo, fontsize=11)
#         ax.set_xlabel("Diffusion ratio", fontsize=11)
#         ax.xaxis.grid(False)

#     axes[0].set_ylabel("Robustness (rob_shaberi_total)", fontsize=11)

#     fig.suptitle(
#         "Type 3 – Robustness sensitivity to diffusion ratio in #3954 and #1754",
#         fontsize=12, x=0.01, ha="left",
#     )

#     # shared legend
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, title="Topology variant", frameon=False,
#                loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles))

#     fig.tight_layout()
#     save(fig, "compare_ratio_sensitivity_type3")



# def fig_3d_scatter(df):
#     df = df.dropna(subset=["dU", "dV", "dW"])
 
#     fig = plt.figure(figsize=(10, 7))
#     ax  = fig.add_subplot(111, projection="3d")
 
#     cmap = cm.YlGnBu
#     norm = mcolors.Normalize(
#         vmin=df["rob_shaberi_total"].min(),
#         vmax=df["rob_shaberi_total"].max(),
#     )
 
#     for topo, marker in TOPO_MARKERS.items():
#         sub = df[df["topology_id"] == topo]
#         sc  = ax.scatter(
#             sub["dU"], sub["dV"], sub["dW"],
#             c=sub["rob_shaberi_total"],
#             cmap=cmap, norm=norm,
#             marker=marker,
#             s=100,
#             edgecolors="white",
#             linewidths=0.4,
#             label=topo,
#         )
 
#     ax.set_xlabel("dU", fontsize=10, labelpad=8)
#     ax.set_ylabel("dV", fontsize=10, labelpad=8)
#     ax.set_zlabel("dW", fontsize=10, labelpad=8)
#     ax.set_title(
#         "3D diffusion rate space coloured by robustness – all topologies",
#         fontsize=11, pad=12,
#     )
 
#     # colourbar
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.1,
#                  label="Robustness (rob_shaberi_total)")
 
#     # marker legend
#     ax.legend(title="Topology", frameon=False, loc="upper left")
 
#     plt.tight_layout()
#     for ext in ("png",):
#         fig.savefig(OUT_DIR / f"explore_3d_scatter.{ext}", bbox_inches="tight", dpi=300)
#     print("Saved → ResultPlots/explore_3d_scatter.png")
#     plt.show()   # keeps it interactive so you can rotate in VS Code
