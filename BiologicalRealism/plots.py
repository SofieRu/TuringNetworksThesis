#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines

# have to run this first: module load matplotlib/3.9.2-gfbf-2024a
# have to run this first: module load SciPy-bundle/2024.05-gfbf-2024a
# have to run this first: pip install seaborn --user

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 13,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 300,
})

GRID = [('1754', 'robust'), ('1754', 'fragile'), ('3954', 'robust'), ('3954', 'fragile')]

COLORS = {('1754', 'robust'): 'blueviolet',    ('1754', 'fragile'): 'mediumorchid', ('3954', 'robust'): 'cornflowerblue', ('3954', 'fragile'): 'lightskyblue'}

CV_FILES = {
    ('1754', 'robust'):  '1754_cv_sweep_high_config49_N{N}.pkl',
    ('1754', 'fragile'): '1754_cv_sweep_low_config14_N{N}.pkl',
    ('3954', 'robust'):  '3954_cv_sweep_high_config49_N{N}.pkl',
    ('3954', 'fragile'): '3954_cv_sweep_low_config12_N{N}.pkl',
}

SENS_FILES = {
    ('1754', 'robust'):  ('1754_sensitivity_results_config49_N10.pkl', 49),
    ('1754', 'fragile'): ('1754_sensitivity_results_config18_N10.pkl', 18),
    ('3954', 'robust'):  ('3954_sensitivity_results_config49_N10.pkl', 49),
    ('3954', 'fragile'): ('3954_sensitivity_results_config21_N10.pkl', 21),
}

N_SIZES = [10, 20, 30]
N_STYLE = {10: ('-', 'o'), 20: ('--', 's'), 30: (':', '^')}
LETTERS = ['A', 'B', 'C', 'D']

PARAM_LABELS = {
    'alpha_u': 'u basal prod.', 'beta_u': 'u reg. prod.', 'K_uu': 'u self-activation',
    'K_vu': 'v-u inhibition', 'delta_u': 'u degradation',
    'alpha_v': 'v basal prod.', 'beta_v': 'v reg. prod.', 'K_uv': 'u-v activation',
    'K_wv': 'w-v inhibition', 'delta_v': 'v degradation',
    'alpha_w': 'w basal prod.', 'beta_w': 'w reg. prod.', 'K_ww': 'w self-activation',
    'K_uw': 'u-w inhibition', 'K_vw': 'v-w inhibition', 'delta_w': 'w degradation',
}

def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract(cv):
    results = cv['results']
    CV = np.array([r['CV'] for r in results])
    dist_key = next((k for k in ('all_full', 'all_band', 'all_eigenvalues')
                     if k in results[0]), None)
    allv = [np.asarray(r[dist_key], float) for r in results]
    stat = lambda f: np.array([f(v) if v.size else np.nan for v in allv])
    rob_key = ('robustness_marginal' if 'robustness_marginal' in results[0]
               else 'robustness')
    return {'CV': CV, 'all': allv,
            'mean': stat(np.mean), 'min': stat(np.min), 'max': stat(np.max),
            'robustness': np.array([r.get(rob_key, np.nan) for r in results]),
            'config_id': cv.get('config_id', '?'), 'hopping': cv.get('hopping', {})}

def panel_title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', fontsize=12.8, pad=8)

data10 = {}
for k in GRID:
    d = load_pkl(CV_FILES[k].format(N=10))
    if d is not None:
        data10[k] = extract(d)

def desc(key):
    cid = data10[key]['config_id'] if key in data10 else '?'
    return f"{key[0]} {key[1]} (ID {cid})"





# # FIG 1: boxplot
# fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), sharex=True)
# line_handle = None  # To store the handle for the legend

# for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
#     if key not in data10:
#         ax.set_visible(False); continue
#     d = data10[key]
#     box = [v if v.size else np.array([np.nan]) for v in d['all']]
#     bp = ax.boxplot(box, positions=range(len(d['CV'])), widths=0.78, patch_artist=True, showfliers=True, medianprops=dict(color='black', linewidth=1.4), flierprops=dict(marker='o', markersize=3, alpha=0.3))
    
#     for p in bp['boxes']:
#         p.set_facecolor(COLORS[key]); p.set_alpha(0.85)
        
#     # Added label text directly to the threshold line
#     hl = ax.axhline(0, color='red', ls='--', lw=2, zorder=10, label=r'Turing threshold (Re($\lambda$) = 0)')
#     if line_handle is None:
#         line_handle = hl  # Keep a single reference for the global legend
        
#     ax.set_xticks(range(len(d['CV'])))
#     ax.set_xticklabels([f'{c:.2f}' for c in d['CV']])
#     ax.grid(True, alpha=0.3, axis='y')
#     panel_title(ax, LETTERS[i], desc(key))

# for ax in axes[:, 0]:
#     ax.set_ylabel(r'Ring growth rate max Re($\lambda$)', fontsize=14)
# for ax in axes[1, :]:
#     ax.set_xlabel('CV (coefficient of variation)', fontsize=14)

# fig.suptitle('Distribution of ring growth rates under parameter heterogeneity (N = 10)', fontsize=16, y=0.92)

# # Draw a formal, clean figure legend at the bottom center using correct parameters
# if line_handle:
#     fig.legend(handles=[line_handle], loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, frameon=False)

# # Raised the bottom rect bound from 0.02 to 0.07 to give the new legend padding
# fig.tight_layout(rect=[0, 0.05, 1, 0.94])
# fig.subplots_adjust(wspace=0.15)
# fig.savefig('cv_boxplot_2x2.png', dpi=300, bbox_inches='tight')
# plt.close(fig); print("Saved: cv_boxplot_2x2.png")


# FIG 1: boxplot
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)
line_handle = None

for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
    if key not in data10:
        ax.set_visible(False); continue
    d = data10[key]
    box = [v if v.size else np.array([np.nan]) for v in d['all']]
    bp = ax.boxplot(box, positions=range(len(d['CV'])), widths=0.78, patch_artist=True, showfliers=True, medianprops=dict(color='black', linewidth=1.4), flierprops=dict(marker='o', markersize=3, alpha=0.3))
    
    for p in bp['boxes']:
        p.set_facecolor(COLORS[key]); p.set_alpha(0.85)
        
    hl = ax.axhline(0, color='red', ls='--', lw=2, zorder=10, label=r'Turing threshold (Re($\lambda$) = 0)')
    if line_handle is None:
        line_handle = hl 
        
    ax.set_xticks(range(len(d['CV'])))
    ax.set_xticklabels([f'{c:.2f}' for c in d['CV']])
    
    # CHANGED: Added MaxNLocator to increase the density of ticks on the y-axis
    # Increase nbins (e.g., to 10 or 12) if you want even more ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    ax.grid(True, alpha=0.3, axis='y')
    panel_title(ax, LETTERS[i], desc(key))

for ax in axes[:, 0]:
    ax.set_ylabel(r'Ring growth rate max Re($\lambda$)', fontsize=14)
for ax in axes[1, :]:
    ax.set_xlabel('CV (coefficient of variation)', fontsize=14)

fig.suptitle('Distribution of ring growth rates under parameter heterogeneity (N = 10)', fontsize=16, y=0.92)

if line_handle:
    fig.legend(handles=[line_handle], loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, frameon=False)

fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.subplots_adjust(wspace=0.18)
fig.savefig('cv_boxplot_2x2.png', dpi=300, bbox_inches='tight')
plt.close(fig); print("Saved: cv_boxplot_2x2.png")




# FIG 2: min-max
fig, axes = plt.subplots(2, 2, figsize=(12.8, 9), sharex=True)
for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
    if key not in data10:
        ax.set_visible(False); continue
    d = data10[key]
    ax.fill_between(d['CV'], d['min'], d['max'], alpha=0.2, color=COLORS[key], label='full range (min-max)', zorder=1)
    ax.plot(d['CV'], d['mean'], 'o-', color=COLORS[key], lw=2.2, ms=6, markeredgecolor='black', markeredgewidth=0.6, label=r'mean max Re($\lambda$)', zorder=3)
    ax.axhline(0, color='red', ls='--', lw=1.6, zorder=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    panel_title(ax, LETTERS[i], desc(key))
for ax in axes[:, 0]:
    ax.set_ylabel(r'Ring growth rate  max Re($\lambda$)', fontsize=14)
for ax in axes[1, :]:
    ax.set_xlabel('CV (coefficient of variation)', fontsize=14)
fig.suptitle('Mean and full range of ring growth rate vs parameter heterogeneity (N = 10)', fontsize=16, y=0.96)
fig.tight_layout(rect=[0, 0.1, 1, 0.98])
fig.subplots_adjust(wspace=0.15)
fig.savefig('cv_minmax_2x2.png', dpi=300, bbox_inches='tight')
plt.close(fig); print("Saved: cv_minmax_2x2.png")



# FIG 3: sensitivity 2x2
# def sensitivity_on_ax(ax, sens, letter, title_text):
#     names = list(sens['param_names'])
#     s = np.array(sens['sensitivities'], float)
#     nan_mask = np.isnan(s)
#     clean = np.where(~nan_mask)[0]; nans = np.where(nan_mask)[0]
#     order = np.concatenate([clean[np.argsort(s[clean])[::-1]], nans])
#     labels = [PARAM_LABELS.get(names[i], names[i]) for i in order]
#     vals = s[order]; is_nan = np.isnan(vals)
    
#     bars = ax.bar(range(len(labels)), np.where(is_nan, 1e-4, vals), color='steelblue')
    
#     n_clean = int((~is_nan).sum())
    
#     # 2-TONE COLOR SCHEME: Top 3 are pink (stiff), everything else is blue (sloppy)
#     for j in range(len(labels)):
#         if j < min(4, n_clean):
#             bars[j].set_color('mediumvioletred')
#         else:
#             bars[j].set_color('steelblue')
            
#     ax.set_yscale('log'); ax.set_ylim(bottom=1e-4)
#     ax.set_xticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
#     ax.grid(True, alpha=0.3, axis='y', which='major')
#     panel_title(ax, letter, title_text)

# sens_data = {k: load_pkl(p) for k, (p, _id) in SENS_FILES.items()}
# fig, axes = plt.subplots(2, 2, figsize=(12.8, 9), sharex=True)
# for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
#     sd = sens_data.get(key)
#     if sd is None:
#         ax.set_visible(False); continue
#     sensitivity_on_ax(ax, sd, LETTERS[i], f"{key[0]} {key[1]} (ID {SENS_FILES[key][1]})")
# for ax in axes[:, 0]:
#     ax.set_ylabel('Change in growth rate (log)', fontsize=14)

# # COMBINED LEGEND FOR LOG VERSION
# fig.legend(handles=[
#     Patch(facecolor='mediumvioletred', alpha=0.9, label='stiff (critical)'),
#     Patch(facecolor='steelblue', alpha=0.9, label='sloppy (tolerant)'),
# ], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

# fig.suptitle('Parameter sensitivity of the Turing growth rate (N = 10, smth with per 10% parameter change)', fontsize=16, y=0.99)
# fig.tight_layout(rect=[0, 0.02, 1, 1])
# fig.savefig('sensitivity_log.png', dpi=300, bbox_inches='tight')

# FIG 3: sensitivity 2x2
def sensitivity_on_ax(ax, sens, letter, title_text):
    names = list(sens['param_names'])
    s = np.array(sens['sensitivities'], float)
    nan_mask = np.isnan(s)
    clean = np.where(~nan_mask)[0]; nans = np.where(nan_mask)[0]
    order = np.concatenate([clean[np.argsort(s[clean])[::-1]], nans])
    labels = [PARAM_LABELS.get(names[i], names[i]) for i in order]
    vals = s[order]; is_nan = np.isnan(vals)
    
    bars = ax.bar(range(len(labels)), np.where(is_nan, 1e-4, vals), color='steelblue')
    
    n_clean = int((~is_nan).sum())
    
    # 2-TONE COLOR SCHEME: Top 3 are pink (stiff), everything else is blue (sloppy)
    for j in range(len(labels)):
        if j < min(4, n_clean):
            bars[j].set_color('mediumvioletred')
        else:
            bars[j].set_color('steelblue')
            
    ax.set_yscale('log'); ax.set_ylim(bottom=1e-4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=13)
    
    # CHANGED: Force labels to stay visible on every single panel
    ax.tick_params(labelbottom=True)
    
    ax.grid(True, alpha=0.3, axis='y', which='major')
    panel_title(ax, letter, title_text)

sens_data = {k: load_pkl(p) for k, (p, _id) in SENS_FILES.items()}

# CHANGED: Removed sharex=True so the top subplots keep their own independent x-axes
fig, axes = plt.subplots(2, 2, figsize=(12.8, 11))

for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
    sd = sens_data.get(key)
    if sd is None:
        ax.set_visible(False); continue
    sensitivity_on_ax(ax, sd, LETTERS[i], f"{key[0]} {key[1]} (ID {SENS_FILES[key][1]})")
for ax in axes[:, 0]:
    ax.set_ylabel('Change in growth rate (log)', fontsize=15)

# COMBINED LEGEND FOR LOG VERSION
fig.legend(handles=[
    Patch(facecolor='mediumvioletred', alpha=0.9, label='Top 4 Stiff Parameters'),
    Patch(facecolor='steelblue', alpha=0.9, label='Sloppy Parameters'),
], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01)) # Shifted anchor slightly up for padding

fig.suptitle('Local Sensitivity Analysis of Turing Growth Rates (N = 10, 10% Perturbation)', fontsize=16, y=0.92)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])

# CHANGED: Added vertical spacing (hspace) so the top row's rotated labels don't crash into bottom row titles
fig.subplots_adjust(hspace=0.5, wspace=0.12)

fig.savefig('sensitivity_log.png', dpi=300, bbox_inches='tight')
plt.close(fig); print("Saved: sensitivity_log.png")









# FIG 4: robustness vs CV, N sweep
robust_data = {}
for k in GRID:
    for N in N_SIZES:
        d = load_pkl(CV_FILES[k].format(N=N))
        if d is not None:
            robust_data[(k[0], k[1], N)] = extract(d)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, (ax, topo) in enumerate(zip(axes, ['1754', '3954'])):
    ax.axhline(50, color='gray', ls=':', lw=1.1, alpha=0.7, zorder=1)
    for kind in ['robust', 'fragile']:
        col = COLORS[(topo, kind)]
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
    ax.legend(loc='upper right', fontsize=12, ncol=2)
    panel_title(ax, LETTERS[i], f'Topology {topo}')
axes[0].set_ylabel('Robustness (% of trials that stay Turing)')
fig.suptitle('Robustness to parameter heterogeneity across ring sizes (N = 10, 20, 30)', fontsize=16, y=0.93)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('robustness_N10_30.png', dpi=300, bbox_inches='tight')
plt.close(fig); print("Saved: robustness_N10_30.png")










###### BACKUP ######

# # FIG 3: sensitivity 2x2
# def sensitivity_no_log(ax, sens, letter, title_text):
#     names = list(sens['param_names'])
#     s = np.array(sens['sensitivities'], float)
#     nan_mask = np.isnan(s)
#     clean = np.where(~nan_mask)[0]; nans = np.where(nan_mask)[0]
#     order = np.concatenate([clean[np.argsort(s[clean])[::-1]], nans])
#     labels = [PARAM_LABELS.get(names[i], names[i]) for i in order]
#     vals = s[order]; is_nan = np.isnan(vals)
    
#     bars = ax.bar(range(len(labels)), np.where(is_nan, 0.0, vals), color='steelblue', alpha=0.9)
                  
#     n_clean = int((~is_nan).sum())
    
#     # 2-TONE COLOR SCHEME: Top 3 are pink (stiff), everything else is blue (sloppy)
#     for j in range(len(labels)):
#         if j < min(3, n_clean):
#             bars[j].set_color('mediumvioletred')
#         else:
#             bars[j].set_color('steelblue')
        
#     ax.set_ylim(bottom=0) 
#     ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    
#     ax.set_xticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
#     ax.grid(True, alpha=0.3, axis='y', which='major')
#     panel_title(ax, letter, title_text)

# sens_data = {k: load_pkl(p) for k, (p, _id) in SENS_FILES.items()}
# fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
# for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
#     sd = sens_data.get(key)
#     if sd is None:
#         ax.set_visible(False); continue
#     sensitivity_no_log(ax, sd, LETTERS[i], f"{key[0]} {key[1]} (ID {SENS_FILES[key][1]})")
# for ax in axes[:, 0]:
#     ax.set_ylabel('|Δ growth rate| per 10% parameter change') 
    
# # COMBINED LEGEND FOR LINEAR VERSION
# fig.legend(handles=[
#     Patch(facecolor='mediumvioletred', alpha=0.9, label='stiff (critical)'),
#     Patch(facecolor='steelblue', alpha=0.9, label='sloppy (tolerant)'),
# ], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))

# fig.suptitle('Parameter sensitivity of the Turing growth rate (N = 10)', fontsize=16, y=0.99)
# fig.tight_layout(rect=[0, 0, 1, 1])
# fig.savefig('sensitivity_nolog.png', dpi=300, bbox_inches='tight')
