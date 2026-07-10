import numpy as np
import matplotlib.pyplot as plt

N = 10
cells = np.arange(N)

# Compute the m=1 and m=5 spatial modes
mode_1 = np.cos(2 * np.pi * 1 * cells / N)  # one cycle across the ring
mode_5 = np.cos(2 * np.pi * 5 * cells / N)  # five cycles across the ring

# Corresponding k_m values
k_1 = 2 * np.sin(1 * np.pi / N)
k_5 = 2 * np.sin(5 * np.pi / N)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

# --- Left: m=1 ---
ax = axes[0]
ax.plot(cells, mode_1, 'o-', color='darkorchid', linewidth=2.2, markersize=7)
ax.axhline(0, color='gray', linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_xlabel('Cell index', fontsize=12)
ax.set_ylabel('Eigenvector amplitude', fontsize=12)
ax.set_title(f'm=1, $k_1 = {k_1:.3f}$', fontsize=12)
ax.grid(alpha=0.3)
ax.set_xticks(np.arange(0, N + 1, 2))

# --- Right: m=5 ---
ax = axes[1]
ax.plot(cells, mode_5, 'o-', color='darkorchid', linewidth=2.2, markersize=7)
ax.axhline(0, color='gray', linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_xlabel('Cell index', fontsize=12)
ax.set_title(f'm=5, $k_5 = {k_5:.3f}$',fontsize=12)
ax.grid(alpha=0.3)
ax.set_xticks(np.arange(0, N + 1, 2))

# fig.suptitle(
#     f'Discrete spatial modes on a ring of N={N} cells\n'
#     f'Each mode $m$ is a sine wave with $m$ full cycles',
#     fontsize=12, fontweight='semibold' 
# )

# 1. Add the main title
fig.suptitle(
    f'Discrete spatial modes on a ring of N={N} cells', 
    fontsize=13, 
    fontweight='semibold'
)

plt.tight_layout()
plt.savefig('sine_wave_modes_n10.png', dpi=200, bbox_inches='tight')
print("Saved: sine_wave_modes_n10.png")
plt.close()