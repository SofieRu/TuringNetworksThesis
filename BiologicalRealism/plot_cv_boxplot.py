#!/usr/bin/env python3
"""
Boxplot of the per-trial growth rates (max Re lambda) at each CV, using the
arrays saved by heterogeneous_ring_3954_corrected.py.

Two panels:
  (1) BAND  = max Re over finite modes m>0  -> the pattern-forming mode
  (2) m=0   = uniform-mode growth           -> the background stability
The two Turing conditions are: band > 0 (dashed line) AND m=0 < 0.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL = '3954_cv_sweep_high_config49_N10.pkl'   # <- adjust if you renamed it

with open(PKL, 'rb') as f:
    data = pickle.load(f)
results = data['results']

CVs       = [r['CV'] for r in results]
band_data = [r['all_band'] for r in results]   # list of arrays, one per CV
m0_data   = [r['all_m0']   for r in results]
labels    = [f"{cv:.2f}\n(n={len(b)})" for cv, b in zip(CVs, band_data)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

# ---- Panel 1: Turing band (the pattern mode) ----
ax1.boxplot(band_data, labels=labels, showfliers=True)
ax1.axhline(0, color='crimson', ls='--', lw=1, label='Turing threshold (band > 0)')
ax1.set_title('Finite-wavelength band:  max Re $\\lambda$ over $m>0$')
ax1.set_xlabel('CV (parameter noise)')
ax1.set_ylabel('max Re $\\lambda$  (band)')
ax1.legend(loc='upper left', fontsize=8)

# ---- Panel 2: uniform mode (background stability) ----
ax2.boxplot(m0_data, labels=labels, showfliers=True)
ax2.axhline(0, color='crimson', ls='--', lw=1, label='stability threshold (m=0 < 0)')
ax2.set_title('Uniform mode $m=0$:  background stability')
ax2.set_xlabel('CV (parameter noise)')
ax2.set_ylabel('Re $\\lambda$  ($m=0$)')
ax2.legend(loc='lower left', fontsize=8)

fig.suptitle(f"Config {data['config_id']} ({data.get('config_name','')}) — "
             f"growth-rate distributions vs parameter noise", fontsize=11)
fig.tight_layout()
fig.savefig('3954_cv_boxplot.png', dpi=200)
print("saved -> 3954_cv_boxplot.png")
# plt.show()   # uncomment for interactive
