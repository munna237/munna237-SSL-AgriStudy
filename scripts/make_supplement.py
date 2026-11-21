"""Build supplement.tex from the notebook CSVs. Usage: python make_supplement.py <tables_dir>"""
import sys, re
import pandas as pd

TAB = sys.argv[1]
import os as _os0
SUPP = _os0.environ.get('SSL_SUPP_DATA', _os0.path.join(_os0.path.dirname(_os0.path.abspath(__file__)), '..', 'results', 'supplementary_data'))
def esc(s): return str(s).replace('_', '\\_').replace('%', '\\%').replace('→', '$\\rightarrow$').replace('≥', '$\\geq$').replace('≤', '$\\leq$').replace('∪', '$\\cup$').replace('−', '$-$')

NAME = {'imagenet_dino': 'ImageNet DINO', 'dinov2_vits14': 'DINOv2 ViT-S/14', 'dinov2_vitb14': 'DINOv2 ViT-B/14',
        'mae_vitb16': 'MAE ViT-B/16', 'clip_vitb16': 'CLIP ViT-B/16', 'deit_s16_sup': 'DeiT-S/16 (supervised)',
        'resnet50_sup': 'ResNet-50 (supervised)', 'florabd_1k': 'FloraBD-1k', 'florabd_5k': 'FloraBD-5k',
        'florabd_10k': 'FloraBD-10k', 'florabd_full_scratch': 'FloraBD-full', 'rice_1k': 'RiceLeaf-1k', 'rice_5k': 'RiceLeaf-5k',
        'rice_full': 'RiceLeaf-full', 'rice_full_s1': 'RiceLeaf-full (seed 1)', 'rice_full_s2': 'RiceLeaf-full (seed 2)',
        'rice_full_ep400': 'RiceLeaf-full, 400 ep', 'rice_full_inrecipe': 'RiceLeaf-full, released recipe',
        'rice_1k_stepmatch': 'RiceLeaf-1k, step-matched', 'rice_5k_stepmatch': 'RiceLeaf-5k, step-matched',
        'pv_5k': 'PlantVillage-5k', 'pv_10k': 'PlantVillage-10k', 'pv_full': 'PlantVillage-full',
        'inet_n996': 'ImageNet-996', 'inet_n4671': 'ImageNet-4,671', 'inet_n6835': 'ImageNet-6,835', 'inet_n30829': 'ImageNet-30,829',
        'florabd_cont_in': 'ImageNet$\\rightarrow$FloraBD', 'rice_cont_in': 'ImageNet$\\rightarrow$RiceLeaf (seed 42)',
        'rice_cont_in_s1': 'ImageNet$\\rightarrow$RiceLeaf (seed 1)', 'rice_cont_in_s2': 'ImageNet$\\rightarrow$RiceLeaf (seed 2)'}
ORDER = list(NAME)

# ---- Table S1/S2: distillation and deployment ----
K = pd.read_csv(f'{TAB}/tab_kd.csv'); D = pd.read_csv(f'{TAB}/tab_deploy.csv'); TT = pd.read_csv(f'{TAB}/tab_kd_teachers.csv')
TEACH = ['ImageNet DINO', 'ImageNet→RiceLeaf (cont.)', 'ImageNet→FloraBD (cont.)', 'RiceLeaf-full (scratch)']
import json as _json, os as _os
RS = _json.load(open(_os.path.join(SUPP, 'ft_kd_dedup_rescore.json')))
_SK = {'MobileNetV3-Small': 'mobilenet_v3_small', 'EfficientNet-B0': 'efficientnet_b0'}
CE_SD = {k: RS['none']['kd_ce'][v]['sd'] for k, v in _SK.items()}
_FT = [('imagenet_dino', 'ImageNet DINO'), ('dinov2_vits14', 'DINOv2 ViT-S/14'), ('dinov2_vitb14', 'DINOv2 ViT-B/14'),
       ('mae_vitb16', 'MAE ViT-B/16'), ('florabd_full_scratch', 'FloraBD-full (scratch)'), ('rice_full', 'RiceLeaf-full (scratch)'),
       ('florabd_cont_in', r'ImageNet$\rightarrow$FloraBD (cont.)'), ('rice_cont_in', r'ImageNet$\rightarrow$RiceLeaf (cont.)')]
_DET = ['none', 'cos0.95', 'cos0.95+phash8']
def _ftcell(d, b):
    v = RS[d]['ft'][b]
    base = f"{v['mean']:.2f} $\\pm$ {v['sd']:.2f}"
    return base if b == 'imagenet_dino' else base + f" ({v['diff']:+.2f}; {v['p_holm7']:.3f})"
ftdd_rows = [r'\multicolumn{4}{l}{\textit{Full fine-tuning}} \\'] + [f"{n} & " + ' & '.join(_ftcell(d, b) for d in _DET) + r' \\' for b, n in _FT]
_TN = {'imagenet_dino': 'ImageNet DINO', 'rice_cont_in': r'IN$\rightarrow$RiceLeaf', 'florabd_cont_in': r'IN$\rightarrow$FloraBD', 'rice_full': 'RiceLeaf-full'}
_SN = {'mobilenet_v3_small': 'MobileNetV3-S', 'efficientnet_b0': 'EfficientNet-B0'}
ftdd_rows.append('\\hline\n' + r'\multicolumn{4}{l}{\textit{Distillation (student $\leftarrow$ teacher)}} \\')
for st in ('mobilenet_v3_small', 'efficientnet_b0'):
    ftdd_rows.append(f"{_SN[st]}, plain & " + ' & '.join(f"{RS[d]['kd_ce'][st]['mean']:.2f} $\\pm$ {RS[d]['kd_ce'][st]['sd']:.2f}" for d in _DET) + r' \\')
    for t in ('imagenet_dino', 'rice_cont_in', 'florabd_cont_in', 'rice_full'):
        cells = []
        for d in _DET:
            v = [r for r in RS[d]['kd'] if r['student'] == st and r['teacher'] == t][0]
            cells.append(f"{v['mean']:.2f} $\\pm$ {v['sd']:.2f} ({v['gain']:+.2f}; {v['p_holm']:.2f})")
        ftdd_rows.append(f"{_SN[st]} $\\leftarrow$ {_TN[t]} & " + ' & '.join(cells) + r' \\')
kd_rows = []
for st in ['MobileNetV3-Small', 'EfficientNet-B0']:
    k = K[K.student == st].set_index('teacher')
    ce = k['CE mean'].iloc[0]
    cells = [f"{ce:.2f} $\\pm$ {CE_SD[st]:.2f}"] + [f"{k.loc[t, 'KD mean ± std'].replace('±', '$\\pm$')} ({k.loc[t, 'gain_pp']:+.2f}; {k.loc[t, 'p_holm']:.2f})" for t in TEACH]
    kd_rows.append(f'{st} & ' + ' & '.join(cells) + ' \\\\')
teach_row = 'Teacher (fine-tuned ViT-S/16) & & ' + ' & '.join(f"{TT.set_index('teacher').loc[t, 'test_acc']:.2f}" for t in TEACH) + ' \\\\'
_LE = pd.read_csv(f'{TAB}/tab_label_efficiency.csv')
_LEC = ['ImageNet DINO', 'DINOv2 ViT-S/14', 'FloraBD-full (scratch)', 'RiceLeaf-full (scratch)', 'ImageNet→FloraBD (cont.)', 'ImageNet→RiceLeaf (cont.)']
_TGT = {'FloraBD': 'FloraBD', 'RiceLeaf': 'RiceLeafDiseaseBD', 'Flowers102': 'Flowers-102', 'CUB': 'CUB-200-2011'}
_LEN = {'ImageNet→FloraBD (cont.)': 'florabd_cont_in', 'ImageNet→RiceLeaf (cont.)': 'rice_cont_in', 'FloraBD-full (scratch)': 'florabd_full_scratch',
        'RiceLeaf-full (scratch)': 'rice_full', 'DINOv2 ViT-S/14': 'dinov2_vits14'}
_PUB = _json.load(open(_os.path.join(SUPP, 'label_efficiency_tests_published.json')))
_NS = {(r['target'], round(r['fraction'], 2), r['backbone']) for r in _PUB if r['p_holm'] >= 0.05}
def _lecell(t, frac, col, v):
    v = str(v).replace('±', '$\\pm$')
    return v + (' n.s.' if (t, frac, _LEN.get(col)) in _NS else '')
le_rows = []
for _t, _tn in _TGT.items():
    _d = _LE[_LE.target == _t]
    for _k, (_, _r) in enumerate(_d.iterrows()):
        _lab = str(_r['labels']).replace('%', '\\%').replace('~', '$\\approx$').replace('/class', '')
        le_rows.append((_tn if _k == 0 else '') + ' & ' + _lab + ' & ' + ' & '.join(_lecell(_t, round(float(str(_r['labels']).split('%')[0]) / 100, 2), c, _r[c]) for c in _LEC) + ' \\\\')
    le_rows.append('\\hline')
le_rows = le_rows[:-1]
# ---- Table S11: label efficiency on de-duplicated test sets ----
LED = _json.load(open(_os.path.join(SUPP, 'label_efficiency_dedup.json')))
_LEB = ['dinov2_vits14', 'florabd_full_scratch', 'rice_full', 'florabd_cont_in', 'rice_cont_in']
_LET = {(r['target'], r['fraction'], r['backbone'], d): r for d in ('none', 'cos0.95') for r in LED['tests'][d]}
def _lec(t, f, b, d):
    r = _LET.get((t, f, b, d))
    if r is None: return '--'
    return f"${r['mean_diff']:+.2f}$" + ('$^{*}$' if r['p_holm'] < 0.05 else '')
led_rows = []
for _t in ('RiceLeaf', 'FloraBD'):
    for _k, _f in enumerate((0.01, 0.05, 0.10, 0.25, 0.50)):
        led_rows.append((f"{_TGT[_t]} ($n$ = {LED['n_test'][_t]['none']:,} / {LED['n_test'][_t]['cos0.95']:,})" if _k == 0 else '') + f' & {int(_f*100)}\\% & '
                        + ' & '.join(_lec(_t, _f, b, 'none') + ' & ' + _lec(_t, _f, b, 'cos0.95') for b in _LEB) + ' \\\\')
    led_rows.append('\\hline')
led_rows = led_rows[:-1]
# ---- Section: near-duplicate evidence (notebook 17) ----
_SD = SUPP
_BIN = pd.read_csv(f'{_SD}/same_label_by_cosine_bin.csv')
_BL = ['< 0.90', '0.90–0.95', '0.95–0.98', '≥ 0.98']
_DSN = {'FloraBD': 'FloraBD', 'RiceLeaf': 'RiceLeafDiseaseBD', 'PlantVillage': 'PlantVillage', 'PlantDoc': 'PlantDoc', 'Flowers102': 'Flowers-102', 'CUB': 'CUB-200-2011'}
def _bc(r): return f"{r['same label (%)']:.1f} ({int(r['test images']):,})" if r['test images'] else '--'
bin_rows = [_DSN[d] + ' & ' + ' & '.join(_bc(_BIN[(_BIN.dataset == d) & (_BIN['cosine bin'] == b)].iloc[0]) for b in _BL) + ' \\\\' for d in _DSN]
_W = pd.read_csv(f'{_SD}/within_train_duplicates.csv')
within_rows = [f"{_DSN[r.iloc[0]]} & {int(r.iloc[1]):,} & {int(r.iloc[2]):,} & {r.iloc[3]:.1f} & {r.iloc[4]:.1f} & {int(r.iloc[5]):,} ({r.iloc[6]:.1f}) \\\\" for _, r in _W.iterrows()]
top_figs = []
for _d, _n in _DSN.items():
    top_figs.append(f"\\begin{{figure}}[p]\n\\centering\n\\includegraphics[width=\\textwidth]{{figures/supp/top_pairs_{_d}.jpg}}\n"
                    f"\\caption{{\\textbf{{{_n}: most similar and near-threshold test--train pairs.}} Each pair shows the test image (left) and its nearest training or validation image (right), with cosine similarity, Hamming distance and label agreement (green: same label; red: different). Top: the eight most similar test images. Middle and bottom: eight random test images from nearest cosine 0.93--0.95 and 0.95--0.97 (fewer where the band has fewer images).}}\n"
                    f"\\label{{fig:s-top-{_d}}}\n\\end{{figure}}")
dep_rows = [f"{r.model} & {r.params_M:.2f} & {r.size_MB:.1f} & {r.GMACs:.3f} & {r.cpu_ms_1thread_median:.1f} & {r.cpu_ms_allthreads_median:.1f} \\\\" for r in D.itertuples()]

# ---- Table S3: detector agreement ----
A = pd.read_csv(f'{TAB}/rev_dup_detectors.csv').set_index('dataset')
agr_cols = [('n_test', 'Test'), ('exact (MD5)', 'MD5'), ('cos≥0.9', 'cos $\\geq$ 0.90'), ('cos≥0.95', 'cos $\\geq$ 0.95'), ('cos≥0.98', 'cos $\\geq$ 0.98'), ('pHash≤8', 'pHash $\\leq$ 8'),
            ('both (cos≥0.95 & pHash≤8)', 'Both'), ('Jaccard', 'Jaccard'), ("Cohen's κ", '$\\kappa$'),
            ('same label, cos-flagged pairs (%)', 'Same label, cos (\\%)'), ('same label, pHash-flagged pairs (%)', 'Same label, pHash (\\%)')]
DS = {'FloraBD': 'FloraBD', 'RiceLeaf': 'RiceLeafDiseaseBD', 'PlantVillage': 'PlantVillage', 'PlantDoc': 'PlantDoc',
      'Flowers102': 'Flowers-102', 'CUB': 'CUB-200-2011'}
def fmt(v):
    try:
        f = float(v); return f'{int(f):,}' if f == int(f) and abs(f) > 1 else f'{f:g}'
    except Exception: return esc(v)
agr_rows = [f"{DS[d]} & " + ' & '.join(fmt(A.loc[d, c]) for c, _ in agr_cols) + ' \\\\' for d in DS]

# ---- Table S4: k-NN on de-duplicated test sets ----
KD = pd.read_csv(f'{TAB}/rev_knn_dedup.csv').set_index('backbone')
TG = ['FloraBD', 'RiceLeaf', 'PlantVillage', 'PlantDoc', 'Flowers102', 'CUB']
knn_rows = [f'{NAME[b]} & ' + ' & '.join(f'{KD.loc[b, t]:.2f}' for t in TG) + ' \\\\' for b in ORDER if b in KD.index]

# ---- Tables S5/S6: sensitivity ----
DETS = ['full test set', 'MD5 only', 'cos ≥ 0.90', 'cos ≥ 0.95 (paper)', 'cos ≥ 0.98', 'pHash ≤ 4', 'pHash ≤ 8', 'pHash ≤ 12', 'cos ≥ 0.95 ∪ pHash ≤ 8']
def comp_label(c):
    t, rest = c.split(': ', 1); a, b = rest.split(' − ')
    return f'{DS.get(t, t)}: {NAME.get(a, esc(a))} $-$ {NAME.get(b, esc(b))}'
def sens_table(f):
    S = pd.read_csv(f'{TAB}/{f}', index_col=0)
    rows = []
    for c, r in S.iterrows():
        cells = [esc(r[d]).replace('[', '{\\scriptsize[').replace(']', ']}') if d in S.columns and isinstance(r[d], str) else '--' for d in DETS]
        rows.append(f'{comp_label(c)} & ' + ' & '.join(cells) + f" & {sign_code(r.iloc[-1])} \\\\")
    return rows
# The notebook stored the CI-sign code in alphabetical column order; re-order it to the table's column order
# (Full, MD5, cos 0.90, cos 0.95, cos 0.98, pHash 4, pHash 8, pHash 12, Union).
ALPHA_TO_TABLE = [2, 3, 8, 4, 0, 1, 7, 5, 6]
def sign_code(v):
    v = str(v)
    if not v.startswith('NO'): return 'all'
    t = ['?'] * 9
    for i, ch in enumerate(v[4:]): t[ALPHA_TO_TABLE[i]] = ch
    sym = {'0': '0', '\u2212': '$-$', '-': '$-$', '+': '$+$'}
    return '\\,'.join(sym[ch] for ch in t)
sens_lp, sens_knn = sens_table('rev_dup_sensitivity_lp.csv'), sens_table('rev_dup_sensitivity_knn.csv')
SIZES = pd.read_csv(f'{TAB}/rev_dup_test_sizes.csv', index_col=0)
size_rows = [f'{esc(i)} & ' + ' & '.join(f'{int(SIZES.loc[i, t]):,}' for t in TG) + ' \\\\' for i in SIZES.index]

det_head = ' & '.join(['Full', 'MD5', 'cos 0.90', 'cos 0.95', 'cos 0.98', 'pHash 4', 'pHash 8', 'pHash 12', 'Union'])
def figure(stem, ds):
    import os
    parts = [k for k in ('cos', 'pHash') if os.path.exists(f'figures/supp/dup_pairs_{stem}_{k}.jpg')]
    desc = {'cos': 'flagged only by the feature detector (cos $\\geq 0.95$, Hamming $> 8$), highest cosine first',
            'pHash': 'flagged only by pHash (Hamming $\\leq 8$, cos $< 0.95$)'}
    out = ''
    for n, k in enumerate(parts):
        cont = '\\ContinuedFloat\n' if n else ''
        head = (f"\\textbf{{{ds}: test--train pairs on which the two detectors disagree.}} Each numbered pair shows a test image (left) "
                f"and its nearest training or validation image (right), with the cosine similarity of their ImageNet DINO features and "
                f"the Hamming distance of their perceptual hashes. Pairs {desc[k]}.") if n == 0 else \
               f"\\textbf{{{ds} (continued).}} Pairs {desc[k]}."
        lab = f'\\label{{fig:s-{stem}}}\n' if n == 0 else ''
        out += (f"\\begin{{figure}}[p]\n{cont}\\centering\n\\includegraphics[width=0.9\\textwidth]{{figures/supp/dup_pairs_{stem}_{k}.jpg}}\n"
                f"\\caption{{{head}}}\n{lab}\\end{{figure}}\n")
    return out

tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,array,pdflscape,cite,caption}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}
\title{Supplementary Material for\\ ``Does Local Data Beat ImageNet at Laboratory Scale? A Budget-Matched Study of Self-Supervised Pretraining and Near-Duplicate Inflation in Agricultural Vision''}
\author{Mohammad Mehedi Hasan Munna, Acramul Haque Kabir, Reefka Fabliha Tulona, Tanvir Bin Zahid,\\Nahid Hossain Antu, Md.\ Mansur ul Mulk and Arvid Chowdhury}
\date{}
\begin{document}
\maketitle
All values come from the released notebooks and their result files; the main paper's protocols, splits and statistics apply unless stated. Machine-readable versions of every table are provided as CSV files in the folder \emph{supplementary\_data}.

\section{Distillation and deployment (exploratory)}
\label{sec:s1}
This section is exploratory: none of its gains is significant, and distillation methods for rice disease are the subject of a separate paper \cite{agrikd}, with which no results are shared. It tests only whether the pretraining source changes what a small deployable student reaches.

\textbf{Protocol.} One ViT-S/16 teacher per pretraining source is fine-tuned on RiceLeafDiseaseBD (one seed). MobileNetV3-Small \cite{howard2019searching} and EfficientNet-B0 \cite{tan2019efficientnet} students with ImageNet weights are trained for 50 epochs (AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch 64) with
\begin{equation}
\mathcal{L} = \alpha\, T^{2}\, \mathrm{KL}\!\left(p^{T}_{\mathrm{teacher}} \,\|\, p^{T}_{\mathrm{student}}\right) + (1-\alpha)\,\mathcal{L}_{\mathrm{CE}},
\end{equation}
where $p^{T}$ is the softmax at temperature $T = 4$ and $\alpha = 0.5$ \cite{hinton2015distilling}, against a plain fine-tuned student, three seeds each (Welch's $t$-test, Holm over the eight student--teacher pairs). Latency is PyTorch fp32 at batch 1 on an Intel Xeon CPU at 2.20~GHz (median of 200 runs).

\textbf{Results.} Distillation from the ImageNet DINO teacher (91.17\%) gave $+1.40$ points for MobileNetV3-Small and $+1.02$ for EfficientNet-B0 over plain fine-tuning, not significant ($p_{\mathrm{Holm}} = 0.21$ and $0.16$); the RiceLeaf-full teacher (82.81\%) gave none (Table~\ref{tab:s-kd}). The distilled MobileNetV3-Small came within 0.2 points of the ViT-S/16 teacher at one tenth of its single-thread server-CPU latency (Table~\ref{tab:s-deploy}); phones and embedded boards were not measured.

\begin{table}[h]
\centering\small
\caption{\textbf{RiceLeafDiseaseBD students: test accuracy (\%), mean $\pm$ s.d. over three seeds}, with the gain over plain fine-tuning (pp) and the Holm-adjusted $p$-value in parentheses. Last row: teacher test accuracy.}
\label{tab:s-kd}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccc}
\hline
Student & Plain & KD $\leftarrow$ ImageNet DINO & KD $\leftarrow$ IN$\rightarrow$RiceLeaf & KD $\leftarrow$ IN$\rightarrow$FloraBD & KD $\leftarrow$ RiceLeaf-full \\
\hline
""" + '\n'.join(kd_rows) + '\n\\hline\n' + teach_row + r"""
\hline
\end{tabular}}
\end{table}

\begin{table}[h]
\centering\small
\caption{\textbf{Deployment cost}: fp32 PyTorch, batch 1, $224\times224$ input, Intel Xeon CPU at 2.20~GHz, median of 200 runs.}
\label{tab:s-deploy}
\begin{tabular}{lrrrrr}
\hline
Model & Params (M) & Size (MB) & GMACs & 1 thread (ms) & 12 threads (ms) \\
\hline
""" + '\n'.join(dep_rows) + r"""
\hline
\end{tabular}
\end{table}


\clearpage
\section{Near-duplicate detector agreement}
\label{sec:s2}
Table~\ref{tab:s-agree} reports, for each dataset, how many test images each detector flags and how well the two detectors agree, together with how often a flagged test image shares its label with its nearest training or validation image. A detector that finds true near-duplicates should flag pairs that almost always share a label; the cosine detector did so in 94--100\% of cases on every dataset except PlantDoc, whereas pHash-flagged pairs did so on FloraBD (100\%) and RiceLeafDiseaseBD (95.8\%) but in only 47.8\% of cases on PlantVillage and 14.5\% on CUB, because there pHash flags different objects on similar uniform backgrounds (Figs.~\ref{fig:s-PlantVillage}--\ref{fig:s-CUB}). Pairs flagged only by the cosine detector were the same scene photographed with a small shift in framing or scale, which changes 14--18 of the 64 hash bits. The six RiceLeafDiseaseBD images flagged only by pHash were same-scene pairs just below the cosine threshold. On PlantDoc, flagged pairs included resized copies of the same stock photograph, and only 10 of the 21 cosine-flagged pairs shared a label (47.6\%, Table~\ref{tab:s-agree}); the other 11 carried different labels in training and test, i.e., the same image appears under two classes. Exact-copy (MD5) counts are included in every de-duplicated set.

\begin{table}[h]
\centering\small
\caption{\textbf{Near-duplicate audit and agreement of the two detectors.} Counts of test images that are exact (MD5) copies of a training or validation image, or are flagged at cosine $\geq$ 0.90, 0.95 and 0.98 (ImageNet DINO features) and at pHash Hamming $\leq 8$; Jaccard index and Cohen's $\kappa$ of the two flags; percentage of flagged test images whose nearest training/validation image has the same label, out of the test images flagged by that detector (for the small sets: PlantDoc 10 of 21 cosine-flagged and 6 of 15 pHash-flagged, Flowers-102 12 of 15 and CUB 11 of 76 pHash-flagged). $\kappa$ also counts the test images that neither detector flags, so it can exceed the Jaccard index; it is highest on PlantDoc because there the two flagged sets largely coincide (14 of the 22 flagged images are flagged by both).}
\label{tab:s-agree}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrrrrr}
\hline
Dataset & """ + ' & '.join(h for _, h in agr_cols) + r""" \\
\hline
""" + '\n'.join(agr_rows) + r"""
\hline
\end{tabular}}
\end{table}

""" + ''.join(figure(s, DS[s]) for s in ['FloraBD', 'RiceLeaf', 'PlantVillage', 'PlantDoc', 'Flowers102', 'CUB']) + r"""
\clearpage
\section{k-NN on de-duplicated test sets}
\label{sec:s3}
\begin{table}[h]
\centering\small
\caption{\textbf{k-NN test accuracy (\%, $k = 20$, $T = 0.07$) on de-duplicated test sets} (cosine $\geq 0.95$ $\cup$ MD5). Test images kept: """ + ', '.join(f"{DS[t]} {int(SIZES.loc['cos ≥ 0.95 (paper)', t]):,}" for t in TG) + r""". The continued-pretraining variants are reported in Table~8 of the main paper.}
\label{tab:s-knn}
\begin{tabular}{lrrrrrr}
\hline
Backbone & FloraBD & RiceLeaf & PlantVillage & PlantDoc & Flowers-102 & CUB \\
\hline
""" + '\n'.join(knn_rows) + r"""
\hline
\end{tabular}
\end{table}

\clearpage
\section{Sensitivity of the comparisons to the duplicate detector}
\label{sec:s4}
Tables~\ref{tab:s-sens-lp} and~\ref{tab:s-sens-knn} give, for every comparison of the main paper and every detector and threshold, the accuracy difference (pp) with its 95\% paired-bootstrap CI after removing the flagged test images; every set also removes exact (MD5) copies. The last column states whether the sign of the interval (above, below or including zero) is the same under every detector. Table~\ref{tab:s-sizes} gives the number of test images kept. At cosine $\geq 0.90$ only 128 RiceLeafDiseaseBD test images remain, so that column is uninformative.

\begin{table}[h]
\centering\small
\caption{\textbf{Test images kept} by each detector and threshold.}
\label{tab:s-sizes}
\begin{tabular}{lrrrrrr}
\hline
Set & FloraBD & RiceLeaf & PlantVillage & PlantDoc & Flowers-102 & CUB \\
\hline
""" + '\n'.join(size_rows) + r"""
\hline
\end{tabular}
\end{table}

\begin{landscape}
\begin{table}
\centering
\caption{\textbf{Linear probe}: difference (pp, 95\% CI) under each detector and threshold. \emph{Same sign}: `all' if the 95\% CI falls on the same side of zero (or includes zero) under every column; otherwise one symbol per column, in the column order of the table (Full to Union): $-$ interval below zero, 0 interval includes zero, $+$ interval above zero, computed from unrounded bounds.}
\label{tab:s-sens-lp}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l*{9}{c}l}
\hline
Comparison & """ + det_head + r""" & Same sign \\
\hline
""" + '\n'.join(sens_lp) + r"""
\hline
\end{tabular}}
\end{table}
\begin{table}
\centering
\caption{\textbf{k-NN}: difference (pp, 95\% CI) under each detector and threshold. \emph{Same sign}: `all' if the 95\% CI falls on the same side of zero (or includes zero) under every column; otherwise one symbol per column, in the column order of the table (Full to Union): $-$ interval below zero, 0 interval includes zero, $+$ interval above zero, computed from unrounded bounds.}
\label{tab:s-sens-knn}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l*{9}{c}l}
\hline
Comparison & """ + det_head + r""" & Same sign \\
\hline
""" + '\n'.join(sens_knn) + r"""
\hline
\end{tabular}}
\end{table}
\end{landscape}

\section{Fine-tuning and distillation on the de-duplicated rice test set}
\label{sec:s6dd}
Table~\ref{tab:s-ftdd} re-scores the RiceLeafDiseaseBD fine-tuning and distillation runs from their saved per-image predictions on the de-duplicated test set (cosine $\geq 0.95$ $\cup$ MD5, $n = 539$) and on its union with pHash $\leq 8$ ($n = 533$), with the protocols used on the full test set: Welch's $t$-test over three seeds, Holm over the seven fine-tuning comparisons with ImageNet DINO and, separately, over the eight student--teacher pairs. On the full test set the same code reproduces the published values (e.g., $90.90 \pm 0.49$\% for ImageNet DINO; $p_{\mathrm{Holm}} = 0.21$ and $0.16$ for the two ImageNet DINO students). Label-efficiency probes are re-scored in Section S7 (Table~\ref{tab:s-led}).

\begin{table}[h]
\centering\small
\caption{\textbf{RiceLeafDiseaseBD fine-tuning and distillation re-scored on de-duplicated test sets}: test accuracy (\%), mean $\pm$ s.d. over three seeds; in parentheses, the difference from ImageNet DINO (fine-tuning) or from the plain student (distillation), and the Holm-adjusted $p$-value.}
\label{tab:s-ftdd}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\hline
 & Full test ($n = 1{,}472$) & cos $\geq 0.95$ ($n = 539$) & cos $\cup$ pHash ($n = 533$) \\
\hline
""" + "\n".join(ftdd_rows) + r"""
\hline
\end{tabular}}
\end{table}

\section{Full fine-tuning on Flowers-102 and CUB}
\label{sec:s5}
Table~\ref{tab:s-ft} gives the full fine-tuning results on the two natural-image targets, moved here from Table~12 of the main paper; RiceLeafDiseaseBD remains the headline fine-tuning result there. The recipe and statistics are those of the main paper (30 epochs, AdamW, learning rate $5\times10^{-5}$, three seeds; Welch's $t$-test, Holm over the same 21 tests as Table~12 of the main paper). With this single untuned recipe, fine-tuned ImageNet DINO scored below its own linear probe on Flowers-102 (93.99 vs 95.72\%) and CUB (76.77 vs 81.86\%), a known effect of full fine-tuning on small fine-grained datasets \cite{kumar2022finetuning}, and DINOv2 ViT-S/14 had one unstable CUB seed (53.56\% against 80.05\% and 74.92\%). These results are therefore indicative only. Under this recipe the continued checkpoints trailed ImageNet DINO on CUB by 5.0--6.1 points ($p_{\mathrm{Holm}} < 0.05$); on Flowers-102 their deficits (0.9--2.4 points) were not significant.

\begin{table}[h]
\centering\small
\caption{\textbf{Full fine-tuning test accuracy (\%) on Flowers-102 and CUB-200-2011, mean $\pm$ s.d. over three seeds.} $^{*}$Significantly different from ImageNet DINO (Welch, Holm over the 21 tests of Table~12 of the main paper).}
\label{tab:s-ft}
\begin{tabular}{lcc}
\hline
Backbone & Flowers-102 & CUB-200-2011 \\
\hline
ImageNet DINO & 93.99 $\pm$ 1.02 & 76.77 $\pm$ 0.59 \\
DINOv2 ViT-S/14 & 97.96 $\pm$ 0.27 & 69.51 $\pm$ 14.05 \\
DINOv2 ViT-B/14 & 99.33 $\pm$ 0.17 & 88.21 $\pm$ 0.05$^{*}$ \\
MAE ViT-B/16 & 90.81 $\pm$ 0.39 & 76.57 $\pm$ 0.66 \\
FloraBD-full (scratch) & 62.08 $\pm$ 0.17$^{*}$ & 24.82 $\pm$ 0.48$^{*}$ \\
RiceLeaf-full (scratch) & 48.92 $\pm$ 0.86$^{*}$ & 18.58 $\pm$ 0.30$^{*}$ \\
ImageNet$\rightarrow$FloraBD (cont.) & 93.04 $\pm$ 0.71 & 70.69 $\pm$ 1.12$^{*}$ \\
ImageNet$\rightarrow$RiceLeaf (cont.) & 91.63 $\pm$ 1.03 & 71.74 $\pm$ 0.95$^{*}$ \\
\hline
\multicolumn{3}{l}{\textit{Supervised ImageNet baselines, learning rate tuned per architecture (not tested)}} \\
ResNet-50 & 93.73 $\pm$ 0.21 & 79.17 $\pm$ 0.44 \\
EfficientNet-B0 & 94.96 $\pm$ 0.40 & 75.56 $\pm$ 0.50 \\
ViT-S/16 (ImageNet-21k) & 98.11 $\pm$ 0.40 & 83.06 $\pm$ 0.52 \\
Swin-T & 95.87 $\pm$ 0.25 & 82.21 $\pm$ 0.61 \\
\hline
\end{tabular}
\end{table}

\clearpage
\section{Compute, linear-probe differences and label efficiency}
\label{sec:s7}
\textbf{Compute.} Table~4 of the main paper gives two measures of pretraining compute. Because wall-clock time was not logged consistently across resumed sessions, the analytic estimate of training floating-point operations is
\begin{equation}
F = 2\,S\,B\,\big[\,3\,(2M_{224} + n_\ell M_{96}) + 2M_{224}\big],
\end{equation}
which counts the student's forward and backward passes ($3\times$ forward) over 2 global and $n_\ell$ local crops ($n_\ell = 8$; 10 for the released recipe) and the teacher's forward pass over the 2 global crops, with $M_{224} = 4.60$ and $M_{96} = 0.81$ GMACs for ViT-S/16 (plus 0.02 GMACs of projection head per crop) and 2 FLOPs per multiply--accumulate; it ignores data loading and the optimizer. In addition, the exact training code and settings were timed on one NVIDIA L4 GPU for each configuration (two short epochs, warm epoch used), with GPU board power sampled once per second; multiplying the measured seconds and joules per step by each arm's step count gives GPU-hours and GPU energy. One optimizer step took 0.835~s with 8 local crops and 0.922~s with 10, at a mean board power of 62--63~W, and continued pretraining cost the same per step as training from scratch (0.838~s). RiceLeaf-full therefore took 2.5 GPU-hours, the 400-epoch run 9.8, and all laboratory-scale pretraining 87.8 GPU-hours and 5.5~kWh of GPU energy, whereas the released ImageNet DINO corresponds to about 4,100 L4-hours at the same per-image cost; the largest laboratory-scale run, PlantVillage-full or ImageNet-30,829, used 0.28\% of its compute. For RiceLeaf runs completed in a single session, the logged wall-clock time was 4--10\% above these estimates; the exception is the step-matched RiceLeaf-1k run, whose 707 short epochs each restart the data loader (5.25~h logged against 2.46~h estimated). GPU energy excludes the host and is therefore a lower bound.

\textbf{Linear-probe differences.} Fig.~\ref{fig:s-heatmap} shows the differences of Table~5 of the main paper as a heatmap.

\begin{figure}[h]
\centering
\includegraphics[width=0.62\textwidth]{figures/fig_domain_vs_imagenet.pdf}
\caption{\textbf{Linear-probe accuracy difference from ImageNet DINO (percentage points) for each domain checkpoint and target.} Rows are grouped by pretraining source; the color scale covers deficits only, so the one positive value (PlantVillage-full on PlantVillage, $+0.2$, not significant) is white.}
\label{fig:s-heatmap}
\end{figure}

\textbf{Label efficiency.} Table~\ref{tab:s-le} gives the values plotted in Fig.~4 of the main paper. Table~\ref{tab:s-led} re-scores the same probes on the de-duplicated test sets. The probes were re-fitted from the cached features to obtain per-image predictions: every one of the 2,424 refits was within three test images (at most 0.20 points) of the stored accuracy, 15 of the 24 backbone--target pairs reproduced exactly, and the 90 published tests were reproduced up to one borderline case (ImageNet$\rightarrow$RiceLeaf at 25\%, $p_{\mathrm{Holm}} = 0.055$ published, 0.042 re-fitted). On Flowers-102 and CUB, which lose 5\% and 0.2\% of their test images, no difference from ImageNet DINO changed by more than 1.7 points (values in \texttt{label\_efficiency\_dedup.json}, which also gives the union with pHash $\leq 8$).

\begin{table}[h]
\centering\small
\caption{\textbf{Label efficiency: linear-probe test accuracy (\%), mean $\pm$ s.d. over 20 shared label subsets.} All differences from ImageNet DINO are significant (paired $t$-test, Holm over 90 tests) except those marked n.s. The 100\% rows use the label-efficiency protocol (fixed $C = 0.316$, training split only, no refit on train plus validation) and are not comparable with Table~5 of the main paper (e.g., 82.81 vs 85.12\% for ImageNet DINO on RiceLeafDiseaseBD).}
\label{tab:s-le}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccc}
\hline
Target & Labels (per class) & ImageNet DINO & DINOv2 ViT-S/14 & FloraBD-full & RiceLeaf-full & IN$\rightarrow$FloraBD & IN$\rightarrow$RiceLeaf \\
\hline
""" + "\n".join(le_rows) + r"""
\hline
\end{tabular}}
\end{table}

\begin{table}[h]
\centering\small
\caption{\textbf{Label efficiency on de-duplicated test sets: difference from ImageNet DINO (points), mean over 20 shared label subsets.} Full: full test set, re-fitted probes (Section S7; the only change from Table~\ref{tab:s-le} in significance is IN$\rightarrow$RiceLeaf at 25\%, $p_{\mathrm{Holm}} = 0.055$ published); Dedup: cosine $\geq 0.95$ $\cup$ MD5 removed. $^{*}$Significant (paired $t$-test over subsets, Holm over the 90 tests of each test set). The $t$-test treats the test images as fixed. Paired bootstrap intervals over test images (2,000 resamples of subset-averaged per-image correctness) for every cell are in \texttt{label\_efficiency\_dedup\_bootstrap.json}. They exclude zero for far fewer advantages: for IN$\rightarrow$RiceLeaf on the de-duplicated rice set only at 50\% ($+3.01$ $[+0.5, +5.4]$; at 10\% and 25\% $[-0.5, +3.0]$ and $[-0.5, +3.5]$), and for IN$\rightarrow$FloraBD on the full FloraBD set only at 1\% ($+1.46$ $[+1.0, +1.9]$).}
\label{tab:s-led}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccccccc}
\hline
& & \multicolumn{2}{c}{DINOv2 ViT-S/14} & \multicolumn{2}{c}{FloraBD-full} & \multicolumn{2}{c}{RiceLeaf-full} & \multicolumn{2}{c}{IN$\rightarrow$FloraBD} & \multicolumn{2}{c}{IN$\rightarrow$RiceLeaf} \\
Target & Labels & Full & Dedup & Full & Dedup & Full & Dedup & Full & Dedup & Full & Dedup \\
\hline
""" + "\n".join(led_rows) + r"""
\hline
\end{tabular}}
\end{table}

\section{Holm families}
Table~\ref{tab:s-fam} defines every Holm family used in the paper. The file \texttt{holm\_families.csv} lists each member comparison (model, reference, target, protocol, test set, raw and adjusted $p$), built from the stored results by \texttt{make\_families.py}. Recomputing Holm over the 75 comparisons of Table~5 reproduces every significance mark in that table.

\begin{table}[h]
\centering\small
\caption{\textbf{Holm families.} All comparisons are against ImageNet DINO unless stated otherwise. Targets: FB FloraBD, RL RiceLeafDiseaseBD, PV PlantVillage, PD PlantDoc, FL Flowers-102, CUB.}
\label{tab:s-fam}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{r>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{7.6cm}>{\raggedright\arraybackslash}p{2.1cm}l}
\hline
Size & Analysis (where reported) & Members & Targets & Test \\
\hline
75 & Linear probe, Table 5 & 15 backbones: DINOv2 S/14, B/14, MAE B/16; FloraBD 1k/5k/10k/full; RiceLeaf 1k/5k/full; PlantVillage 5k/10k/full; IN$\rightarrow$FloraBD, IN$\rightarrow$RiceLeaf (seed 42) & FB RL PV FL CUB & McNemar \\
135 & All other linear-probe comparisons & the 15 above plus 12: ImageNet-996/4,671/6,835/30,829; RiceLeaf-1k/5k step-matched; RiceLeaf-full 400 ep and released recipe; RiceLeaf-full and IN$\rightarrow$RiceLeaf seeds 1, 2 & FB RL PV FL CUB & McNemar \\
20 & Local vs generic, Table 6 & 4 pairs: RiceLeaf-1k/5k/full and PlantVillage-full vs ImageNet-996/4,671/6,835/30,829 & FB RL PV FL CUB & McNemar \\
36 ($\times$4) & Forgetting-aware variants, Table 8 & 3 variants vs 2 references (ImageNet DINO, standard recipe); one family per protocol (LP, k-NN) and test set (full, dedup) & all six & McNemar \\
18 ($\times$2) & Non-DINO references, Table 10 & CLIP ViT-B/16, supervised DeiT-S/16, supervised ResNet-50; one family per protocol & all six & McNemar \\
90 & Label efficiency, Fig.~4 & 5 backbones (4 domain checkpoints, DINOv2 S/14) $\times$ 18 budget--target pairs (5 budgets on FB and RL, 4 on FL and CUB) & FB RL FL CUB & paired $t$ \\
21 & Fine-tuning, Table 12 & 7 backbones (DINOv2 S/14, B/14, MAE B/16, FloraBD-full, RiceLeaf-full, IN$\rightarrow$FloraBD, IN$\rightarrow$RiceLeaf) & RL FL CUB & Welch $t$ \\
8 & Distillation, Section S1 & 2 students $\times$ 4 teachers, each vs the plain fine-tuned student & RL & Welch $t$ \\
\hline
\end{tabular}
\end{table}

\section{Near-duplicate evidence and duplicates within training splits}
\textbf{Contact sheets.} Figs.~\ref{fig:s-top-FloraBD}--\ref{fig:s-top-CUB} show, for every dataset, the eight test images most similar to a training or validation image, and eight random test images each from nearest cosine 0.93--0.95 (just below the threshold) and 0.95--0.97 (just above it), with their nearest neighbor, cosine similarity, Hamming distance and whether the labels agree. The most similar pairs are the same photograph or the same leaf or flower taken moments apart. Near the threshold most pairs show the same plant re-framed, and some show distinct but similar images, on both sides of 0.95. Identical images do not always share a label: four of the eight most similar RiceLeafDiseaseBD pairs are visually identical (Hamming distance 0) and carry different disease labels.

\textbf{Same-label agreement by similarity.} Table~\ref{tab:s-bins} gives, per cosine bin, the share of test images whose nearest training or validation image has the same label. Agreement rises with similarity on the field datasets (RiceLeafDiseaseBD 71.5\% at 0.90--0.95, 92.1\% at 0.95--0.98, 96.6\% above 0.98; CUB 83.0\%, 100\%, 100\%). FloraBD agrees in every bin because its classes are visually distinct, and PlantDoc disagrees most for its most similar images (35.7\% above 0.98), because identical web images appear under different labels.

\textbf{PlantDoc.} The training split has 28 classes and the test split 27: \emph{Tomato two spotted spider mites leaf} (two training images, none in validation) has no test images. The class is kept in probe fitting and k-NN, so a prediction of it counts as an error, and accuracy is computed over all 236 test images. Of the 21 cosine-flagged test images, 11 are byte-identical (MD5) to a training or validation image, and 8 of these 11 carry a different label (for example, the same photograph as \emph{Potato leaf late blight} in the test split and \emph{Potato leaf early blight} in the training split). Counting resized copies and near-identical photographs, 11 of the 21 flagged images have a nearest neighbor with a different label. File names, labels and similarities are in \texttt{plantdoc\_flagged\_pairs.csv}.

\textbf{Duplicates within training splits.} The training splits of FloraBD, RiceLeafDiseaseBD and PlantVillage are the pretraining sets of the full local arms. Table~\ref{tab:s-within} reports, for each training image, whether another training image lies at cosine $\geq 0.95$ or Hamming distance $\leq 8$, and the number of groups obtained by joining images connected at cosine $\geq 0.95$. The grouping is a rough estimate of distinct scenes: chaining can merge distinct images, and the feature detector misses strongly re-framed shots.

\begin{table}[h]
\centering\small
\caption{\textbf{Same-label agreement by nearest-neighbor similarity.} For test images in each bin of nearest training/validation cosine similarity (ImageNet DINO features): percentage whose nearest neighbor has the same label (number of test images in the bin).}
\label{tab:s-bins}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\hline
Dataset & $<0.90$ & 0.90--0.95 & 0.95--0.98 & $\geq 0.98$ \\
\hline
""" + "\n".join(bin_rows) + r"""
\hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{\textbf{Near-duplicates within the training splits (= pretraining sets of the full local arms).} Extra MD5 copies: images beyond the first of each byte-identical set. Twin: another training image at cosine $\geq 0.95$ or Hamming distance $\leq 8$. Groups: connected components of the graph joining training images at cosine $\geq 0.95$.}
\label{tab:s-within}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrr}
\hline
Dataset & Images & Extra MD5 copies & cos twin (\%) & pHash twin (\%) & Groups (\% of images) \\
\hline
""" + "\n".join(within_rows) + r"""
\hline
\end{tabular}}
\end{table}

""" + "\n".join(top_figs) + r"""

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
open('supplement.tex', 'w').write(tex)
print('supplement.tex written')
