"""Regenerate Fig. 3 (scaling), Fig. 5 (label efficiency) and Supplementary Fig. S1 (deployment Pareto)
from the notebook CSVs. Usage: python make_figures.py <tables_dir> <out_dir>"""
import sys, re, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager

TAB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical order (Fig. S1)
# Entity colours shared by Figs. 3 and 5 (all-pairs validated within each figure; markers + dashes as second cue)
COL = {'rice': '#2a78d6', 'flora': '#eda100', 'pv': '#1baf7a', 'inet_sub': '#e34948',
       'cont_flora': '#e87ba4', 'cont_rice': '#008300', 'inet_dino': '#1f1f1f', 'dinov2': '#8a8a8a'}
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e6e6e6'
# TrueType sans (Helvetica/Arial metrics): embeds cleanly as Type 42, unlike CFF-based OpenType fonts
FAM = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 2.5, 'ytick.major.size': 0,
                     'xtick.major.width': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
                     'font.family': FAM, 'pdf.fonttype': 42, 'lines.solid_capstyle': 'round'})

def style(ax):
    ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.tick_params(axis='y', pad=1)

def title(ax, k, t):
    ax.set_title(f'({k}) {t}', loc='left', fontweight='bold', pad=5, fontsize=7.8)

# ---------------- Fig. 3: LP accuracy vs pretraining images ----------------
A = pd.read_csv(f'{TAB}/rev_dedup_all_backbones.csv')
A = A[A.protocol == 'lp']
acc = {(r.backbone, r.target): r['full test set'] for _, r in A.iterrows()}
T2 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('PlantVillage', 'PlantVillage'),
      ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
ARMS = [('RiceLeaf (scratch)', ['rice_1k', 'rice_5k', 'rice_full'], [996, 4671, 6835], COL['rice'], 'o', '-'),
        ('FloraBD (scratch)', ['florabd_1k', 'florabd_5k', 'florabd_10k', 'florabd_full_scratch'], [990, 4995, 9671, 13598], COL['flora'], 'D', (0, (1, 1.3))),
        ('PlantVillage (scratch)', ['pv_5k', 'pv_10k', 'pv_full'], [4920, 9444, 30829], COL['pv'], '^', (0, (5, 1.5, 1, 1.5))),
        ('ImageNet subset (scratch, same N)', ['inet_n996', 'inet_n4671', 'inet_n6835', 'inet_n30829'], [996, 4671, 6835, 30829], COL['inet_sub'], 's', (0, (3.5, 1.5)))]
fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.35))
for k, (ax, (t, ttl)) in enumerate(zip(axes, T2)):
    if ('imagenet_dino', t) in acc:
        ref = acc[('imagenet_dino', t)]
        ax.axhline(ref, color=INK, lw=0.9, ls=(0, (5, 2.5)), zorder=1, label='ImageNet DINO (1.28M images)')
    for lab, names, n, col, mk, ls in ARMS:
        pts = [(m, acc[(b, t)]) for b, m in zip(names, n) if (b, t) in acc]
        if pts:
            ax.plot(*zip(*pts), ls=ls, marker=mk, ms=4.2, lw=1.5, color=col, label=lab, mec='white', mew=0.7, zorder=3)
    for b0, b1, m in [('rice_1k', 'rice_1k_stepmatch', 996), ('rice_5k', 'rice_5k_stepmatch', 4671)]:
        if (b0, t) in acc and (b1, t) in acc:
            ax.annotate('', xy=(m, acc[(b1, t)]), xytext=(m, acc[(b0, t)]),
                        arrowprops=dict(arrowstyle='-|>', color=COL['rice'], lw=0.8, alpha=0.7,
                                        shrinkA=3, shrinkB=3.5, mutation_scale=6), zorder=2)
    sm = [(m, acc[(b, t)]) for b, m in zip(['rice_1k_stepmatch', 'rice_5k_stepmatch'], [996, 4671]) if (b, t) in acc]
    if sm:
        ax.plot(*zip(*sm), ls='none', marker='o', ms=5.2, mfc='white', mec=COL['rice'], mew=1.3, zorder=4,
                label='RiceLeaf, step-matched (≈10.6K steps)')
    ax.set_xscale('log'); ax.set_xlim(700, 45000)
    ax.set_xticks([1e3, 1e4]); ax.set_xticklabels(['1K', '10K'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    title(ax, 'abcde'[k], ttl); style(ax)
axes[0].set_ylabel('Linear-probe test accuracy (%)')
h, l = axes[1].get_legend_handles_labels()
order = [l.index(x) for x in ['RiceLeaf (scratch)', 'FloraBD (scratch)', 'PlantVillage (scratch)',
                               'ImageNet subset (scratch, same N)', 'RiceLeaf, step-matched (≈10.6K steps)', 'ImageNet DINO (1.28M images)']]
fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.84), w_pad=1.3)
fig.text(0.5, axes[0].get_position().y0 - 0.115, 'Pretraining images (log scale)', ha='center', va='top', fontsize=8)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/rev_scaling_with_generic.{e}', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- Fig. 5: label efficiency ----------------
L = pd.read_csv(f'{TAB}/tab_label_efficiency.csv')
SER = [('ImageNet DINO', COL['inet_dino'], 'o', '-', 1.6),
       ('DINOv2 ViT-S/14', COL['dinov2'], 's', (0, (3.5, 1.5)), 1.4),
       ('RiceLeaf-full (scratch)', COL['rice'], 'o', '-', 1.5),
       ('FloraBD-full (scratch)', COL['flora'], 'D', (0, (1, 1.3)), 1.5),
       ('ImageNet→RiceLeaf (cont.)', COL['cont_rice'], 'P', (0, (5, 1.5, 1, 1.5)), 1.4),
       ('ImageNet→FloraBD (cont.)', COL['cont_flora'], 'v', (0, (2.5, 1.2)), 1.4)]
T4 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
def parse(v):
    m = re.match(r'\s*([\d.]+)\s*(?:±\s*([\d.]+))?', str(v)); return (float(m.group(1)), float(m.group(2) or 0)) if m else (np.nan, 0)
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.4))
for k, (ax, (t, ttl)) in enumerate(zip(axes, T4)):
    d = L[L.target == t].copy()
    d['n'] = d['labels'].str.extract(r'~(\d+)/class').astype(float)
    d = d.sort_values('n')
    for s, col, mk, ls, lw in SER:
        if s not in d: raise KeyError(f'{s} missing from tab_label_efficiency.csv')
        mu = np.array([parse(v)[0] for v in d[s]]); sd = np.array([parse(v)[1] for v in d[s]])
        ax.fill_between(d.n, mu - sd, mu + sd, color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot(d.n, mu, ls=ls, marker=mk, ms=3.8, lw=lw, color=col, label=s.replace('→', r'$\rightarrow$'), mec='white', mew=0.6, zorder=3)
    ax.set_xscale('log'); title(ax, 'abcd'[k], ttl); style(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel('Linear-probe test accuracy (%)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6,
           handlelength=2.6, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.84), w_pad=1.3)
fig.text(0.5, axes[0].get_position().y0 - 0.115, 'Labelled images per class (log scale)', ha='center', va='top', fontsize=8)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/fig_label_efficiency.{e}', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

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
