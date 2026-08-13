#!/usr/bin/env python3
# Schematic of the four dispersion-relation types (Shaberi 2025 + Diego 2018 filter), one row of four panels.
# All have a stable homogeneous state at k=0. Colours match the classifier.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False
})

k = np.linspace(0, 6, 500)

reI = 0.85 * np.exp(-((k - 1.8) / 1.1) ** 2) - 0.42 # Type I (restabilises)
reII = 0.34 * np.tanh(1.4 * (k - 1.0)) + 0.26 * np.exp(-((k - 2.2)/1.2) ** 2) - 0.05 # Type II (stays positive)
reH = 0.85 * np.exp(-((k - 1.8) / 1.1) ** 2) - 0.40 # Hopf Re
imH = 0.42 * np.exp(-((k - 1.8) / 1.1) ** 2.) # Hopf Im (clean bump)
reF = 0.45 * np.tanh((k - 1.9) / 0.9) # Filter (monotonic)

COLORS = {
    'Type I': 'steelblue',
    'Type II': 'mediumvioletred',
    'Hopf': 'darkorange',
    'Filter': 'seagreen'
}

TYPES = [
    ('Type I', reI, None),
    ('Type II', reII, None),
    ('Hopf', reH, imH),
    ('Filter', reF, None)
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)

for ax, (name, re, im) in zip(axes, TYPES):
    c = COLORS[name]
    
    ax.fill_between(k, 0, re, where=(re > 0), color=c, alpha=0.15, zorder=1)
    ax.plot(k, re, color=c, lw=2.8, zorder=3)
    
    if im is not None:
        ax.plot(k, im, color='0.45', lw=1.8, ls='--', zorder=2)
        
    ax.axhline(0, color='red', ls='--', lw=2, zorder=2)
    ax.set_title(name, fontsize=16, loc='center', fontweight='bold', color=c)
    ax.set_ylim(-0.5, 0.6)
    ax.set_xlim(0, 6)
    ax.set_xlabel('wavenumber $k$', fontsize=15)
    
axes[0].set_ylabel(r'max Re($\lambda$)', fontsize=15)

legend_handles = [
    mlines.Line2D([], [], color='red', ls='--', lw=2, label=r'Turing threshold (Re($\lambda$) = 0)'),
    mlines.Line2D([], [], color='0.45', ls='--', lw=1.8, label=r'Im($\lambda$) (Hopf)'),
]

fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig('turing_types_schematic.png', dpi=300, bbox_inches='tight')

print("Saved: turing_types_schematic.png")
