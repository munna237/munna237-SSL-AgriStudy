"""Regenerate Fig. 2 (scaling), Fig. 4 (label efficiency) and Supplementary Fig. S1 (deployment Pareto)
from the notebook CSVs. Usage: python make_figures.py <tables_dir> <out_dir>"""
import sys, re, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

TAB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical order
INK, MUTED, GRID = '#222222', '#666666', '#dddddd'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})

def style(ax):
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

# ---------------- Fig. 2: LP accuracy vs pretraining images ----------------
A = pd.read_csv(f'{TAB}/rev_dedup_all_backbones.csv')
A = A[A.protocol == 'lp']
acc = {(r.backbone, r.target): r['full test set'] for _, r in A.iterrows()}
T2 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('PlantVillage', 'PlantVillage'),
      ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
ARMS = [('RiceLeaf (scratch)', ['rice_1k', 'rice_5k', 'rice_full'], [996, 4671, 6835], C[0], 'o', '-'),
        ('ImageNet subset (scratch)', ['inet_n996', 'inet_n4671', 'inet_n6835', 'inet_n30829'], [996, 4671, 6835, 30829], C[1], 's', '--'),
        ('PlantVillage (scratch)', ['pv_5k', 'pv_10k', 'pv_full'], [4920, 9444, 30829], C[2], '^', '-.'),
        ('FloraBD (scratch)', ['florabd_1k', 'florabd_5k', 'florabd_10k', 'florabd_full_scratch'], [990, 4995, 9671, 13598], C[3], 'D', ':')]
fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.55))
for ax, (t, title) in zip(axes, T2):
    for lab, names, n, col, mk, ls in ARMS:
        pts = [(m, acc[(b, t)]) for b, m in zip(names, n) if (b, t) in acc]
        if pts:
            ax.plot(*zip(*pts), ls=ls, marker=mk, ms=4.5, lw=1.6, color=col, label=lab, mec='white', mew=0.6)
    sm = [(m, acc[(b, t)]) for b, m in zip(['rice_1k_stepmatch', 'rice_5k_stepmatch'], [996, 4671]) if (b, t) in acc]
    if sm:
        ax.plot(*zip(*sm), ls='none', marker='o', ms=6, mfc='white', mec=C[0], mew=1.4,
                label='RiceLeaf, step-matched (≈10.6K steps)')
    if ('imagenet_dino', t) in acc:
        ax.axhline(acc[('imagenet_dino', t)], color=INK, lw=1.0, ls=(0, (4, 2)), label='ImageNet DINO (1.28M images)')
    ax.set_xscale('log'); ax.set_xlim(700, 45000); ax.set_title(title, pad=4)
    ax.set_xticks([1e3, 1e4]); ax.set_xticklabels(['1K', '10K'])
    style(ax)
axes[0].set_ylabel('Linear-probe test accuracy (%)')
fig.text(0.5, 0.135, 'Pretraining images (log scale)', ha='center', fontsize=9)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=2.4)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/rev_scaling_with_generic.{e}', dpi=300)
plt.close(fig)

# ---------------- Fig. 4: label efficiency ----------------
L = pd.read_csv(f'{TAB}/tab_label_efficiency.csv')
SER = ['ImageNet DINO', 'DINOv2 ViT-S/14', 'FloraBD-full (scratch)', 'RiceLeaf-full (scratch)',
       'ImageNet→FloraBD (cont.)', 'ImageNet→RiceLeaf (cont.)']
MK = ['o', 's', '^', 'D', 'v', 'P']; LS = ['-', '--', '-', '--', ':', ':']
T4 = [('FloraBD', 'FloraBD'), ('RiceLeaf', 'RiceLeafDiseaseBD'), ('Flowers102', 'Flowers-102'), ('CUB', 'CUB-200-2011')]
def parse(v):
    m = re.match(r'\s*([\d.]+)\s*(?:±\s*([\d.]+))?', str(v)); return (float(m.group(1)), float(m.group(2) or 0)) if m else (np.nan, 0)
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.65))
for ax, (t, title) in zip(axes, T4):
    d = L[L.target == t].copy()
    d['n'] = d['labels'].str.extract(r'~(\d+)/class').astype(float)
    d = d.sort_values('n')
    for s, col, mk, ls in zip(SER, C, MK, LS):
        if s not in d: continue
        mu = np.array([parse(v)[0] for v in d[s]]); sd = np.array([parse(v)[1] for v in d[s]])
        ax.plot(d.n, mu, ls=ls, marker=mk, ms=4, lw=1.5, color=col, label=s, mec='white', mew=0.5)
        ax.fill_between(d.n, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    ax.set_xscale('log'); ax.set_title(title, pad=4); style(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel('Linear-probe test accuracy (%)')
fig.text(0.5, 0.155, 'Labelled images per class (log scale)', ha='center', fontsize=9)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2)
fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=0.8)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/fig_label_efficiency.{e}', dpi=300)
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
