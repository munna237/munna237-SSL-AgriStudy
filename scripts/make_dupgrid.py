"""Fig. 3: agreement of the two near-duplicate detectors (shared 0-22 Hamming axis).
Input: supplementary_data/detector_scores.npz, written by notebook 14 (results_v2/14_detector_scores.npz),
keys '<target>_cos' (nearest train/val cosine, ImageNet DINO) and '<target>_ham' (nearest pHash Hamming distance)."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/../results/supplementary_data/detector_scores.npz'
z = np.load(src)
T = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('PlantVillage', 'PlantVillage'),
     ('PlantDoc', 'PlantDoc'), ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3), sharey=True)
for ax, (k, name) in zip(axes.ravel(), T):
    cb, hb = z[f'{k}_cos'], z[f'{k}_ham'].astype(float)
    fc, fh = cb >= 0.95, hb <= 8
    ax.scatter(cb, hb, s=2.5, color='#2b6cb0', alpha=0.35, lw=0, rasterized=True)
    ax.axvline(0.95, color='#c0392b', lw=0.9, ls=(0, (4, 2)), zorder=3); ax.axhline(8, color='#c0392b', lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax.set_ylim(-1, 23); ax.set_yticks([0, 4, 8, 12, 16, 20])
    lo = max(0.3, np.floor(np.percentile(cb, 0.1) * 20) / 20); ax.set_xlim(lo, 1.005)
    ax.set_title(f'{name} (n = {len(cb):,})', pad=3)
    ax.text(0.03, 0.04, f'cos ≥ 0.95: {fc.sum():,}\npHash ≤ 8: {fh.sum():,}\nboth: {(fc & fh).sum():,}',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb', lw=0.4))
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
for ax in axes[-1]: ax.set_xlabel('Nearest cosine similarity (ImageNet DINO)')
for ax in axes[:, 0]: ax.set_ylabel('Nearest pHash Hamming distance')
fig.tight_layout(w_pad=0.8, h_pad=1.0)
for e in ('pdf', 'png'): fig.savefig(f'{HERE}/../paper/figures/fig_dup_detectors.{e}', dpi=300)
print('written paper/figures/fig_dup_detectors.pdf')
