#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator


plt.rcParams.update({ 
    'font.size': 15, 
    'axes.titlesize': 16, 
    'axes.labelsize': 15, 
    'xtick.labelsize': 15, 
    'ytick.labelsize': 15, 
    'legend.fontsize': 15, 
    'axes.spines.top': False, 
    'axes.spines.right': False, 
    'figure.dpi': 300
})

# ---> FIX 1: FIX THE DUPLICATE TYPO TO INCLUDE THE SECOND TOPOLOGY <---
GRID = [('1754', 'lab_1'), ('3954', 'lab_3')]

COLORS = {('1754', 'lab_1'): 'blueviolet', ('3954', 'lab_3'): 'cornflowerblue'}

CV_FILES = {
    ('1754', 'lab_1'): '1754_cv_sweep_wetlab_config40_N{N}.pkl',
    ('3954', 'lab_3'): '3954_cv_sweep_wetlab_config40_N{N}.pkl',
}

N_SIZES = [10, 20]
N_STYLE = {10: ('-', 'o'), 20: ('--', 's')}
LETTERS = ['A', 'B', 'C', 'D']

def load_pkl(path):
    if not os.path.exists(path):
        print(f"Warning: File not found -> {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract(cv):
    results = cv.get('results', [])
    if not results:
        return None
        
    CV = np.array([r['CV'] for r in results])
    
    # Safely search for valid structural keys inside your pkl dictionary
    dist_key = next((k for k in ('all_full', 'all_band', 'all_eigenvalues', 'allv') if k in results[0]), None)
                     
    if dist_key is None:
        print("Warning: Could not find eigenvalue array key in pkl results structure.")
        allv = [np.array([np.nan]) for _ in results]
    else:
        allv = [np.asarray(r[dist_key], float) for r in results]
        
    stat = lambda f: np.array([f(v) if (v is not None and v.size > 0) else np.nan for v in allv])
    rob_key = 'robustness_marginal' if 'robustness_marginal' in results[0] else 'robustness'
    
    return {
        'CV': CV, 'all': allv,
        'mean': stat(np.mean), 'min': stat(np.min), 'max': stat(np.max),
        'robustness': np.array([r.get(rob_key, np.nan) for r in results]),
        'config_id': cv.get('config_id', '?'), 'hopping': cv.get('hopping', {})
    }

def panel_title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', fontsize=15, pad=8)

# Load data into tracking dictionary
data10 = {}
for k in GRID:
    d = load_pkl(CV_FILES[k].format(N=10))
    if d is not None:
        extracted_data = extract(d)
        if extracted_data is not None:
            data10[k] = extracted_data

def desc(key):
    cid = data10[key]['config_id'] if key in data10 else '?'
    return f"Topology #{key[0]} (ID {cid})"




#### BOXPLOT 
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6), sharex=True)
line_handle = None

for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
    # If a file is completely missing, display an empty placeholder box so layout stays intact
    if key not in data10:
        ax.text(0.5, 0.5, f"Data Missing:\n{CV_FILES[key].format(N=10)}", ha='center', va='center', color='gray')
        panel_title(ax, LETTERS[i], f"{key[0]} (No Data)")
        continue
        
    d = data10[key]
    box = [v if (v is not None and v.size > 0) else np.array([np.nan]) for v in d['all']]
    
    bp = ax.boxplot(
        box, positions=range(len(d['CV'])), widths=0.78, patch_artist=True, 
        showfliers=True, medianprops=dict(color='black', linewidth=1.4), 
        flierprops=dict(marker='o', markersize=3, alpha=0.3)
    )
    
    for p in bp['boxes']:
        p.set_facecolor(COLORS[key])
        p.set_alpha(0.85)
        
    hl = ax.axhline(0, color='red', ls='--', lw=2, zorder=10, label=r'Turing threshold (Re($\lambda$) = 0)')
    if line_handle is None:
        line_handle = hl 
        
    ax.set_xticks(range(len(d['CV'])))
    ax.set_xticklabels([f'{c:.2f}' for c in d['CV']])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.set_ylim(top=2.4) if key == ('3954', 'lab_3') else ax.set_ylim(top=1.7)
    ax.grid(True, alpha=0.3, axis='y')
    panel_title(ax, LETTERS[i], desc(key))

for ax in axes:
    ax.set_ylabel(r'max Re($\lambda$)', fontsize=16)
    ax.set_xlabel('CV (coefficient of variation)', fontsize=16)

fig.suptitle('Distribution of Growth Rates under Parameter Heterogeneity (N=10) for Wet-Lab Configurations', fontsize=17, y=0.9)

if line_handle:
    fig.legend(handles=[line_handle], loc='lower center', bbox_to_anchor=(0.5, -0.01), frameon=False)

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.subplots_adjust(wspace=0.18)
fig.savefig('thesis_wetlab_boxplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: thesis_wetlab_boxplot.png")



# FIG 4 GENERATION
robust_data = {}
for k in GRID:
    for N in N_SIZES:
        d = load_pkl(CV_FILES[k].format(N=N))
        if d is not None:
            robust_data[(k[0], k[1], N)] = extract(d)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, (ax, topo) in enumerate(zip(axes, ['1754', '3954'])):
    
    for (g_topo, kind) in GRID:
        if g_topo != topo:
            continue  # Skip configurations that don't belong to this subplot panel
            
        col = COLORS.get((topo, kind), 'black') # Safe fallback color to prevent a KeyError crash
        
        for N in N_SIZES:
            d = robust_data.get((topo, kind, N))
            if d is None:
                continue
            ls, mk = N_STYLE[N]
            ax.plot(d['CV'], d['robustness'], marker=mk, color=col, linestyle=ls,
                    lw=2, markersize=6, markeredgecolor='white', markeredgewidth=0.9,
                    label=f'{kind}, N={N}', zorder=3)
                    
    ax.set_xlim(-0.01, 0.42); ax.set_ylim(-3, 103)
    ax.set_xlabel('CV (coefficient of variation)')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    panel_title(ax, LETTERS[i], f'Topology {topo}')

axes[0].set_ylabel('Robustness (% of trials that stay Turing)')
fig.suptitle('Robustness to parameter noise across ring sizes (N = 10, 20)', fontsize=16, y=0.93)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('thesis_wetlab_robustness.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: thesis_wetlab_robustness.png")
