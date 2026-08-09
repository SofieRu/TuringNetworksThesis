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
    'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 13,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 300,
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
    dist_key = next((k for k in ('all_full', 'all_band', 'all_eigenvalues', 'allv')
                     if k in results[0]), None)
                     
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
    ax.set_title(f'({letter}) {text}', loc='left', fontsize=12.8, pad=8)

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
    return f"Topology #{key[0]} {key[1].capitalize()} (ID {cid})"

# --- FIG GENERATION ---
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6), sharex=True)
line_handle = None

for i, (ax, key) in enumerate(zip(axes.flat, GRID)):
    # If a file is completely missing, display an empty placeholder box so layout stays intact
    if key not in data10:
        ax.text(0.5, 0.5, f"Data Missing:\n{CV_FILES[key].format(N=10)}", ha='center', va='center', color='gray')
        panel_title(ax, LETTERS[i], f"{key[0]} {key[1]} (No Data)")
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
    ax.grid(True, alpha=0.3, axis='y')
    panel_title(ax, LETTERS[i], desc(key))

for ax in axes:
    ax.set_ylabel(r'Ring growth rate max Re($\lambda$)', fontsize=14)
    ax.set_xlabel('CV (coefficient of variation)', fontsize=14)

fig.suptitle('Distribution of ring growth rates under parameter heterogeneity (N = 10)', fontsize=16, y=0.96)

if line_handle:
    fig.legend(handles=[line_handle], loc='lower center', bbox_to_anchor=(0.5, -0.01), frameon=False)

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.subplots_adjust(wspace=0.18)
fig.savefig('thesis_wetlab.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved: thesis_wetlab.png")



# --- FIG 4 GENERATION ---
robust_data = {}
for k in GRID:
    for N in N_SIZES:
        d = load_pkl(CV_FILES[k].format(N=N))
        if d is not None:
            robust_data[(k[0], k[1], N)] = extract(d)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, (ax, topo) in enumerate(zip(axes, ['1754', '3954'])):
    
    # ---> FIX: ONLY LOOP THROUGH THE ACTUAL CONFIGURATIONS MATCHING THIS TOPO <---
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








PKLS = {
    '#1754 (ID 40)': '1754_cv_sweep_wetlab_config40_N10.pkl',
    #'1754 config 12': '1754_cv_sweep_low_config12_N10.pkl',
    '#3954 (ID 40)': '3954_cv_sweep_wetlab_config40_N10.pkl',
    #'3954 config 47': '3954_cv_sweep_low_config17_N10.pkl',
}

N_RING       = 10
RINGS_PANEL  = 50
CVS          = [0.1, 0.2, 0.3, 0.4]
COLORS       = {0.1: 'steelblue', 0.2: 'deeppink', 0.3: 'darkorange', 0.4: 'forestgreen'}
SEED         = 42

nH = 2
Ha  = lambda X, K: X**nH / (K**nH + X**nH)
Hi  = lambda X, K: K**nH / (K**nH + X**nH)
dHa = lambda x, K:  nH*K**nH*x**(nH-1) / (K**nH + x**nH)**2
dHi = lambda x, K: -nH*K**nH*x**(nH-1) / (K**nH + x**nH)**2

def ode_1754(s, p):
    u, v, w = s
    au, bu, Kvu, du = p[0:4]; av, bv, Kuv, Kwv, dv = p[4:9]
    aw, bw, Kww, Kuw, Kvw, dw = p[9:15]
    return np.array([au + bu*Hi(v, Kvu) - du*u, av + bv*Ha(u, Kuv)*Hi(w, Kwv) - dv*v, aw + bw*Ha(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw*w])

def jac_1754(s, p):
    u, v, w = s
    au, bu, Kvu, du = p[0:4]; av, bv, Kuv, Kwv, dv = p[4:9]
    aw, bw, Kww, Kuw, Kvw, dw = p[9:15]
    J = np.zeros((3, 3))
    J[0, 0] = -du                       
    J[0, 1] = bu*dHi(v, Kvu)
    J[1, 0] = bv*dHa(u, Kuv)*Hi(w, Kwv); J[1, 1] = -dv
    J[1, 2] = bv*Ha(u, Kuv)*dHi(w, Kwv)
    J[2, 0] = bw*Ha(w, Kww)*dHi(u, Kuw)*Hi(v, Kvw)
    J[2, 1] = bw*Ha(w, Kww)*Hi(u, Kuw)*dHi(v, Kvw)
    J[2, 2] = bw*dHa(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw
    return J

def ode_3954(s, p):
    u, v, w = s
    au, bu, Kuu, Kvu, du = p[0:5]; av, bv, Kuv, Kwv, dv = p[5:10]
    aw, bw, Kww, Kuw, Kvw, dw = p[10:16]
    return np.array([au + bu*Ha(u, Kuu)*Hi(v, Kvu) - du*u, av + bv*Ha(u, Kuv)*Hi(w, Kwv) - dv*v, aw + bw*Ha(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw*w])

def jac_3954(s, p):
    u, v, w = s
    au, bu, Kuu, Kvu, du = p[0:5]; av, bv, Kuv, Kwv, dv = p[5:10]
    aw, bw, Kww, Kuw, Kvw, dw = p[10:16]
    J = np.zeros((3, 3))
    J[0, 0] = bu*dHa(u, Kuu)*Hi(v, Kvu) - du            # u self-activation
    J[0, 1] = bu*Ha(u, Kuu)*dHi(v, Kvu)
    J[1, 0] = bv*dHa(u, Kuv)*Hi(w, Kwv); J[1, 1] = -dv
    J[1, 2] = bv*Ha(u, Kuv)*dHi(w, Kwv)
    J[2, 0] = bw*Ha(w, Kww)*dHi(u, Kuw)*Hi(v, Kvw)
    J[2, 1] = bw*Ha(w, Kww)*Hi(u, Kuw)*dHi(v, Kvw)
    J[2, 2] = bw*dHa(w, Kww)*Hi(u, Kuw)*Hi(v, Kvw) - dw
    return J

def find_ss(ode, jac, p, guess, n_newton=80):
    x = np.array(guess, float)
    for _ in range(n_newton):
        try:
            dx = np.linalg.solve(jac(x, p), -ode(x, p))
        except np.linalg.LinAlgError:
            return None
        x = x + dx
        if np.max(np.abs(dx)) < 1e-12:
            break
    if np.max(np.abs(ode(x, p))) < 1e-8 and np.all(x > 0):
        return x
    return None

def local_dispersion(J3, D, kgrid):
    return np.array([np.max(np.real(np.linalg.eigvals(J3 - (k*k)*D))) for k in kgrid])

def ring_curve(ode, jac, base, D, kgrid, CV, ss0, rng, N):
    """One noisy ring's dispersion = MAX over its cells of the local dispersion
    (the envelope / driver region). This tracks the full-ring max Re(lambda) -- the
    box-plot value -- so the figure is consistent with the robustness plots. (Use
    np.mean instead of np.max for the 'typical cell', which is more pessimistic.)"""
    sg = np.sqrt(np.log(1 + CV**2)); mu = -sg**2/2
    cells = []
    for _ in range(N):
        pi = base * rng.lognormal(mu, sg, len(base))
        si = find_ss(ode, jac, pi, ss0)
        if si is None:
            return None
        cells.append(local_dispersion(jac(si, pi), D, kgrid))
    return np.max(cells, axis=0)

M_VALUES = np.arange(0, N_RING // 2 + 1) # mode indices m = 0..N/2
fig, axes = plt.subplots(len(PKLS), len(CVS), figsize=(12.8, 6.8), sharex=True, sharey='row')

for row, (label, pkl) in enumerate(PKLS.items()):
    d = pickle.load(open(pkl, 'rb'))
    base = np.array(d['baseline_params'], float)
    hp = d['hopping']; hv = np.array([hp['h_u'], hp['h_v'], hp['h_w']]); D = np.diag(hv)
    ode, jac = (ode_1754, jac_1754) if len(base) == 15 else (ode_3954, jac_3954)

    ss0 = None
    for g in ([0.5, 0.15, 0.1], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]):
        ss0 = find_ss(ode, jac, base, g)
        if ss0 is not None:
            break
    kd = 2*np.sin(np.pi*M_VALUES/N_RING) # the ONLY wavenumbers a ring of N cells can realise (m = 0..N/2)
    disp0 = local_dispersion(jac(ss0, base), D, kd) # evaluated ONLY at those modes

    for col, CV in enumerate(CVS):
        ax = axes[row, col]
        rng = np.random.default_rng(SEED); drawn = tries = 0
        while drawn < RINGS_PANEL and tries < RINGS_PANEL*80:
            tries += 1
            c = ring_curve(ode, jac, base, D, kd, CV, ss0, rng, N_RING)   # discrete k only
            if c is not None:
                ax.plot(kd, c, 'o-', color=COLORS[CV], linewidth=1.2, markersize=5, alpha=0.4, zorder=3)
                drawn += 1
        ax.plot(kd, disp0, 'o-', color='black', linewidth=2.0, markersize=8, zorder=5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        if row == 0:
            ax.set_title(f'CV = {CV:.2f}', fontsize=14)
        if row == len(PKLS) - 1:
            # thin out near-duplicate high-k modes so labels don't collide
            keep = [i for i in range(len(kd)) if i != len(kd) - 2]
            ax.set_xticks(kd[keep])
            ax.set_xticklabels([f'$k_{{{M_VALUES[i]}}}$={kd[i]:.2f}' for i in keep], rotation=40, ha='right', fontsize=12)
            ax.set_xlabel('Wavenumber $k_m$', fontsize=14)

    axes[row, 0].set_ylabel(f'{label}\n' + r'Max Re($\lambda$)', fontsize=14)
    axes[row, 0].yaxis.set_major_locator(MultipleLocator(0.15 if '3954' in label else 0.25))
    for col in range(axes.shape[1]):
        axes[row, col].tick_params(axis='y', labelsize=12)

fig.align_ylabels(axes[:, 0])
fig.subplots_adjust(left=0.09, right=0.96, top=0.83, bottom=0.18, wspace=0.07, hspace=0.06)
fig.suptitle(f'Dispersion Relation under Parameter Noise (N={N_RING} cells, ' f'{RINGS_PANEL} trials)\n' f'Robust Topology #1754 vs Robust Topology #3954', fontsize=16, y=0.97)

legend_handles = [
    mlines.Line2D([], [], color='black', linewidth=2, marker='o', linestyle='-', label='Baseline (CV=0.0)'),
    mlines.Line2D([], [], color='red', linewidth=2, linestyle='--', label='Turing Threshold'),
]

fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=14)
fig.savefig('thesis_wetlab_dispersion.png', dpi=300, bbox_inches='tight')