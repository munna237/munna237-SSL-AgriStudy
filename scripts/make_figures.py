"""Regenerate Fig. 2 (heatmap), Fig. 3 (scaling), Fig. 5 (label efficiency) and Supplementary Fig. S1 (deployment
Pareto) from the notebook CSVs, in a Nature-style layout (Arial/Helvetica metrics, 5-7 pt text, bold lowercase panel
letters, 0.5 pt axes, outward ticks, no gridlines). Usage: python make_figures.py <tables_dir> <out_dir>"""
import sys, re, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize

TAB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical order (Fig. S1)
# One colour per pretraining source, shared by Figs. 3 and 5 (all-pairs validated within each figure;
# markers and dash patterns are the second cue). Released models in black/grey.
COL = {'rice': '#2a78d6', 'flora': '#eda100', 'pv': '#1baf7a', 'inet_sub': '#e34948',
       'cont_flora': '#e87ba4', 'cont_rice': '#008300', 'inet_dino': '#1a1a1a', 'dinov2': '#8c8c8c'}
INK, MUTED = '#1a1a1a', '#4d4d4d'
# Arial/Helvetica-metric TrueType (embeds as Type 42; CFF OpenType fonts do not embed cleanly)
FAM = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
MM = 1 / 25.4                                     # Nature: 89 mm single column, 183 mm double column
plt.rcParams.update({
    'axes.unicode_minus': True, 'font.family': FAM, 'font.size': 6, 'axes.titlesize': 6.5, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.linewidth': 0.5, 'axes.edgecolor': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.labelpad': 2.5, 'axes.titlepad': 3, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'lines.solid_capstyle': 'round', 'legend.frameon': False, 'legend.handlelength': 2.4,
    'legend.handletextpad': 0.5, 'legend.columnspacing': 1.5, 'savefig.dpi': 600})

def style(ax, ny=4):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(ny, integer=True))

def panel(ax, letter, name, dx=-0.30):
    ax.text(dx, 1.13, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')
    ax.set_title(name, fontsize=6.5, loc='center')

def parse(v):
    m = re.match(r'\s*([-\d.]+)\s*(?:±\s*([\d.]+))?', str(v)); return (float(m.group(1)), float(m.group(2) or 0)) if m else (np.nan, 0)

def save(fig, name):
    for e in ('pdf', 'png'): fig.savefig(f'{OUT}/{name}.{e}', bbox_inches='tight', pad_inches=0.015)
    plt.close(fig)

A = pd.read_csv(f'{TAB}/rev_dedup_all_backbones.csv')
A = A[A.protocol == 'lp']
acc = {(r.backbone, r.target): r['full test set'] for _, r in A.iterrows()}

# ---------------- Fig. 2: LP difference from ImageNet DINO (heatmap) ----------------
ROWS = [('FloraBD-1k', 'florabd_1k'), ('FloraBD-5k', 'florabd_5k'), ('FloraBD-10k', 'florabd_10k'),
        ('FloraBD-full', 'florabd_full_scratch'), ('RiceLeaf-1k', 'rice_1k'), ('RiceLeaf-5k', 'rice_5k'),
        ('RiceLeaf-full', 'rice_full'), ('PlantVillage-5k', 'pv_5k'), ('PlantVillage-10k', 'pv_10k'),
        ('PlantVillage-full', 'pv_full'), ('ImageNet→FloraBD', 'florabd_cont_in'), ('ImageNet→RiceLeaf', 'rice_cont_in')]
COLS = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeaf'), ('PlantVillage', 'PlantVillage'),
        ('Flowers-102', 'Flowers102'), ('CUB', 'CUB')]
M = np.array([[acc[(b, t)] - acc[('imagenet_dino', t)] for _, t in COLS] for _, b in ROWS])
cmap = LinearSegmentedColormap.from_list('deficit', ['#7f1d1d', '#c0392b', '#e34948', '#f2a49c', '#fbe9e6', '#ffffff'])
norm = Normalize(vmin=-75, vmax=0)
fig, ax = plt.subplots(figsize=(89 * MM, 78 * MM))
im = ax.imshow(np.clip(M, -75, 0), cmap=cmap, norm=norm, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        lab = f'{v:+.1f}' if abs(v) < 9.95 else f'{v:+.0f}'
        lab = '0.0' if lab in ('+0.0', '-0.0') else lab.replace('-', '\u2212')
        ax.text(j, i, lab, ha='center', va='center', fontsize=5.5,
                color='white' if v < -35 else INK)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c for c, _ in COLS])
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r.replace('→', r'$\rightarrow$') for r, _ in ROWS])
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
ax.set_xlabel('Evaluation target (linear probe)', labelpad=4)
ax.tick_params(length=0, pad=2.5)
for sp in ax.spines.values(): sp.set_visible(False)
for y in (3.5, 6.5, 9.5):
    ax.axhline(y, color='white', lw=1.6)
ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True); ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
ax.grid(which='minor', color='white', lw=0.5); ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, aspect=28)
cb.set_label('Difference from ImageNet DINO (percentage points)', fontsize=6)
cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2, labelsize=5.5)
cb.set_ticks([-75, -50, -25, 0])
save(fig, 'fig_domain_vs_imagenet')

# ---------------- Fig. 3: LP accuracy vs pretraining images ----------------
T2 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('PlantVillage', 'PlantVillage'),
      ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
ARMS = [('RiceLeaf (scratch)', ['rice_1k', 'rice_5k', 'rice_full'], [996, 4671, 6835], COL['rice'], 'o', '-'),
        ('FloraBD (scratch)', ['florabd_1k', 'florabd_5k', 'florabd_10k', 'florabd_full_scratch'], [990, 4995, 9671, 13598], COL['flora'], 'D', (0, (1, 1.2))),
        ('PlantVillage (scratch)', ['pv_5k', 'pv_10k', 'pv_full'], [4920, 9444, 30829], COL['pv'], '^', (0, (4, 1.3, 1, 1.3))),
        ('ImageNet subset, same N (scratch)', ['inet_n996', 'inet_n4671', 'inet_n6835', 'inet_n30829'], [996, 4671, 6835, 30829], COL['inet_sub'], 's', (0, (3, 1.3)))]
fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 50 * MM))
YLIM = {'FloraBD': (-23.5, 2), 'RiceLeaf': (-23.5, 2), 'PlantVillage': (-23.5, 2),   # agricultural targets share one axis
        'Flowers102': (-78, 4), 'CUB': (-78, 4)}                                     # natural-image targets share another
for k, (ax, (t, ttl)) in enumerate(zip(axes, T2)):
    ref = acc[('imagenet_dino', t)]
    g = lambda b: acc[(b, t)] - ref                                                   # gap to ImageNet DINO (pp)
    ax.axhline(0, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=1, label='ImageNet DINO (1.28M images) = 0')
    for lab, names, n, col, mk, ls in ARMS:
        pts = [(m, g(b)) for b, m in zip(names, n) if (b, t) in acc]
        if pts:
            ax.plot(*zip(*pts), ls=ls, marker=mk, ms=3.2, lw=1.0, color=col, label=lab, mec='white', mew=0.4, zorder=3)
    for b0, b1, m in [('rice_1k', 'rice_1k_stepmatch', 996), ('rice_5k', 'rice_5k_stepmatch', 4671)]:
        if (b0, t) in acc and (b1, t) in acc:
            ax.annotate('', xy=(m, g(b1)), xytext=(m, g(b0)),
                        arrowprops=dict(arrowstyle='-|>', color=COL['rice'], lw=0.6, shrinkA=2.2, shrinkB=2.8,
                                        mutation_scale=4.5), zorder=2)
    sm = [(m, g(b)) for b, m in zip(['rice_1k_stepmatch', 'rice_5k_stepmatch'], [996, 4671]) if (b, t) in acc]
    if sm:
        ax.plot(*zip(*sm), ls='none', marker='o', ms=3.8, mfc='white', mec=COL['rice'], mew=0.9, zorder=4,
                label='RiceLeaf, step-matched (≈10.6K steps)')
    ax.set_xscale('log'); ax.set_xlim(700, 45000)
    ax.set_xticks([1e3, 1e4]); ax.set_xticklabels(['10$^3$', '10$^4$'])
    style(ax); panel(ax, 'abcde'[k], ttl, dx=-0.28)
    ax.set_ylim(*YLIM[t])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5 if YLIM[t][0] > -30 else 20))
    if k in (1, 2, 4): ax.tick_params(labelleft=False)
axes[0].set_ylabel('Difference from\nImageNet DINO (pp)')
axes[3].set_ylabel('Difference (pp)')
h, l = axes[1].get_legend_handles_labels()
order = [l.index(x) for x in ['RiceLeaf (scratch)', 'FloraBD (scratch)', 'PlantVillage (scratch)',
                               'ImageNet subset, same N (scratch)', 'RiceLeaf, step-matched (≈10.6K steps)', 'ImageNet DINO (1.28M images) = 0']]
fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), markerscale=1.5)
fig.tight_layout(rect=(0, 0.08, 1, 0.86), w_pad=1.6)
fig.text(0.5, axes[0].get_position().y0 - 0.15, 'Pretraining images (log scale)', ha='center', va='top', fontsize=6.5)
save(fig, 'rev_scaling_with_generic')

# ---------------- Fig. 5: label efficiency ----------------
L = pd.read_csv(f'{TAB}/tab_label_efficiency.csv')
SER = [('ImageNet DINO', COL['inet_dino'], 'o', '-', 1.1),
       ('DINOv2 ViT-S/14', COL['dinov2'], 's', (0, (3, 1.3)), 1.0),
       ('RiceLeaf-full (scratch)', COL['rice'], 'o', '-', 1.0),
       ('FloraBD-full (scratch)', COL['flora'], 'D', (0, (1, 1.2)), 1.0),
       ('ImageNet→RiceLeaf (cont.)', COL['cont_rice'], 'P', (0, (4, 1.3, 1, 1.3)), 1.0),
       ('ImageNet→FloraBD (cont.)', COL['cont_flora'], 'v', (0, (2.2, 1.1)), 1.0)]
T4 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 52 * MM))
for k, (ax, (t, ttl)) in enumerate(zip(axes, T4)):
    d = L[L.target == t].copy()
    d['n'] = d['labels'].str.extract(r'~(\d+)/class').astype(float)
    d = d.sort_values('n')
    for s, col, mk, ls, lw in SER:
        if s not in d: raise KeyError(f'{s} missing from tab_label_efficiency.csv')
        mu = np.array([parse(v)[0] for v in d[s]]); sd = np.array([parse(v)[1] for v in d[s]])
        ax.fill_between(d.n, mu - sd, mu + sd, color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(d.n, mu, ls=ls, marker=mk, ms=3.0, lw=lw, color=col, label=s.replace('→', r'$\rightarrow$'),
                mec='white', mew=0.4, zorder=3)
    ax.set_xscale('log'); style(ax); panel(ax, 'abcd'[k], ttl, dx=-0.24)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel('Accuracy (%)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0.08, 1, 0.86), w_pad=1.6)
fig.text(0.5, axes[0].get_position().y0 - 0.15, 'Labeled images per class (log scale)', ha='center', va='top', fontsize=6.5)
save(fig, 'fig_label_efficiency')

# ---------------- Supplementary Fig. S1: deployment Pareto ----------------
K = pd.read_csv(f'{TAB}/tab_kd.csv'); D = pd.read_csv(f'{TAB}/tab_deploy.csv'); TT = pd.read_csv(f'{TAB}/tab_kd_teachers.csv')
lat = dict(zip(D.model, D.cpu_ms_1thread_median))
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.set_title('RiceLeafDiseaseBD students', pad=4)
for i, st in enumerate(['MobileNetV3-Small', 'EfficientNet-B0']):
    k = K[K.student == st]; ce = k['CE mean'].iloc[0]
    best = k.assign(m=k['KD mean ± std'].map(lambda v: parse(v)[0])).sort_values('m').iloc[-1]
    ax.plot(lat[st], ce, ls='none', marker='o', ms=7, mfc='white', mec=C[0], mew=1.5, label='Plain fine-tuned' if i == 0 else None)
    ax.plot(lat[st], best.m, ls='none', marker='o', ms=7, color=C[0], mec='white', mew=0.6, label='Best distilled' if i == 0 else None)
    ax.annotate(st, (lat[st], best.m), xytext=(6, 4), textcoords='offset points', fontsize=8, color=INK)
tv = TT.loc[TT.teacher == 'ImageNet DINO', 'test_acc'].iloc[0]
ax.plot(lat['ViT-S/16 (teacher)'], tv, ls='none', marker='s', ms=7, color=C[1], mec='white', mew=0.6, label='ViT-S/16 teacher (ImageNet DINO)')
ax.annotate('ViT-S/16 teacher', (lat['ViT-S/16 (teacher)'], tv), xytext=(-8, 6), textcoords='offset points', ha='right', fontsize=8)
ax.set_xscale('log'); ax.set_xlim(7, 160)
ax.set_xticks([10, 20, 50, 100]); ax.set_xticklabels(['10', '20', '50', '100'])
ax.set_xlabel('CPU latency, 1 thread, batch 1 (ms, log scale)'); ax.set_ylabel('Test accuracy (%)')
ax.set_ylim(89.2, 92.2); style(ax); ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/fig_pareto.{e}', dpi=300)
plt.close(fig)
print('figures written to', OUT)
