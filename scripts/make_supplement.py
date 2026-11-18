"""Build supplement.tex from the notebook CSVs. Usage: python make_supplement.py <tables_dir>"""
import sys, re
import pandas as pd

TAB = sys.argv[1]
def esc(s): return str(s).replace('_', '\\_').replace('%', '\\%').replace('→', '$\\rightarrow$').replace('≥', '$\\geq$').replace('≤', '$\\leq$').replace('∪', '$\\cup$').replace('−', '$-$')

NAME = {'imagenet_dino': 'ImageNet DINO', 'dinov2_vits14': 'DINOv2 ViT-S/14', 'dinov2_vitb14': 'DINOv2 ViT-B/14',
        'mae_vitb16': 'MAE ViT-B/16', 'clip_vitb16': 'CLIP ViT-B/16', 'deit_s16_sup': 'DeiT-S/16 (supervised)',
        'resnet50_sup': 'ResNet-50 (supervised)', 'florabd_1k': 'FloraBD-1k', 'florabd_5k': 'FloraBD-5k',
        'florabd_10k': 'FloraBD-10k', 'florabd_full_scratch': 'FloraBD-full', 'rice_1k': 'RiceLeaf-1k', 'rice_5k': 'RiceLeaf-5k',
        'rice_full': 'RiceLeaf-full', 'rice_full_s1': 'RiceLeaf-full (seed 1)', 'rice_full_s2': 'RiceLeaf-full (seed 2)',
        'rice_full_ep400': 'RiceLeaf-full, 400 ep', 'rice_full_inrecipe': 'RiceLeaf-full, IN recipe',
        'rice_1k_stepmatch': 'RiceLeaf-1k, step-matched', 'rice_5k_stepmatch': 'RiceLeaf-5k, step-matched',
        'pv_5k': 'PlantVillage-5k', 'pv_10k': 'PlantVillage-10k', 'pv_full': 'PlantVillage-full',
        'inet_n996': 'ImageNet-996', 'inet_n4671': 'ImageNet-4,671', 'inet_n6835': 'ImageNet-6,835', 'inet_n30829': 'ImageNet-30,829',
        'florabd_cont_in': 'ImageNet$\\rightarrow$FloraBD', 'rice_cont_in': 'ImageNet$\\rightarrow$RiceLeaf (seed 42)',
        'rice_cont_in_s1': 'ImageNet$\\rightarrow$RiceLeaf (seed 1)', 'rice_cont_in_s2': 'ImageNet$\\rightarrow$RiceLeaf (seed 2)'}
ORDER = list(NAME)

# ---- Table S1/S2: distillation and deployment ----
K = pd.read_csv(f'{TAB}/tab_kd.csv'); D = pd.read_csv(f'{TAB}/tab_deploy.csv'); TT = pd.read_csv(f'{TAB}/tab_kd_teachers.csv')
TEACH = ['ImageNet DINO', 'ImageNet→RiceLeaf (cont.)', 'ImageNet→FloraBD (cont.)', 'RiceLeaf-full (scratch)']
kd_rows = []
for st in ['MobileNetV3-Small', 'EfficientNet-B0']:
    k = K[K.student == st].set_index('teacher')
    ce = k['CE mean'].iloc[0]
    cells = [f'{ce:.2f}'] + [f"{k.loc[t, 'KD mean ± std'].replace('±', '$\\pm$')} ({k.loc[t, 'gain_pp']:+.2f}; {k.loc[t, 'p_holm']:.2f})" for t in TEACH]
    kd_rows.append(f'{st} & ' + ' & '.join(cells) + ' \\\\')
teach_row = 'Teacher (fine-tuned ViT-S/16) & & ' + ' & '.join(f"{TT.set_index('teacher').loc[t, 'test_acc']:.2f}" for t in TEACH) + ' \\\\'
dep_rows = [f"{r.model} & {r.params_M:.2f} & {r.size_MB:.1f} & {r.GMACs:.3f} & {r.cpu_ms_1thread_median:.1f} & {r.cpu_ms_allthreads_median:.1f} \\\\" for r in D.itertuples()]

# ---- Table S3: detector agreement ----
A = pd.read_csv(f'{TAB}/rev_dup_detectors.csv').set_index('dataset')
agr_cols = [('n_test', 'Test'), ('exact (MD5)', 'MD5'), ('cos≥0.95', 'cos $\\geq$ 0.95'), ('pHash≤8', 'pHash $\\leq$ 8'),
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
        rows.append(f'{comp_label(c)} & ' + ' & '.join(cells) + f" & {esc(r.iloc[-1])} \\\\")
    return rows
sens_lp, sens_knn = sens_table('rev_dup_sensitivity_lp.csv'), sens_table('rev_dup_sensitivity_knn.csv')
SIZES = pd.read_csv(f'{TAB}/rev_dup_test_sizes.csv', index_col=0)
size_rows = [f'{esc(i)} & ' + ' & '.join(f'{int(SIZES.loc[i, t]):,}' for t in TG) + ' \\\\' for i in SIZES.index]

det_head = ' & '.join(['Full', 'MD5', 'cos 0.90', 'cos 0.95', 'cos 0.98', 'pHash 4', 'pHash 8', 'pHash 12', 'Union'])
def figure(stem, ds):
    return f"""\\begin{{figure}}[p]
\\centering
\\includegraphics[width=\\textwidth]{{figures/supp/dup_disagree_{stem}.jpg}}
\\caption{{\\textbf{{{ds}: test--train pairs on which the two detectors disagree most.}} Rows 1--2: flagged only by cosine similarity (cos $\\geq 0.95$, Hamming $> 12$), highest cosine first; rows 3--4: flagged only by pHash (Hamming $\\leq 8$, cos $< 0.95$), lowest Hamming first. Each pair shows the test image and its nearest training or validation image.}}
\\label{{fig:s-{stem}}}
\\end{{figure}}
"""

tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,array,pdflscape,cite,caption}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}
\title{Supplementary Material for\\ ``Domain Match Helps, Scale Decides: A Budget-Matched Study of Laboratory-Scale Self-Supervised Pretraining and Near-Duplicate Inflation in Agricultural and Botanical Recognition''}
\author{[Authors]}
\date{}
\begin{document}
\maketitle
All values come from the released notebooks and their result files; the main paper's protocols, splits and statistics apply unless stated. Machine-readable versions of every table are provided as CSV files in the folder \emph{supplementary\_data}.

\section{Distillation and deployment}
\label{sec:s1}
\textbf{Protocol.} ViT-S/16 teachers are fine-tuned on RiceLeafDiseaseBD (seed 0), one per pretraining source. MobileNetV3-Small \cite{howard2019searching} and EfficientNet-B0 \cite{tan2019efficientnet} students with ImageNet weights are trained for 50 epochs (AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch 64) with
\begin{equation}
\mathcal{L} = \alpha\, T^{2}\, \mathrm{KL}\!\left(p^{T}_{\mathrm{teacher}} \,\|\, p^{T}_{\mathrm{student}}\right) + (1-\alpha)\,\mathcal{L}_{\mathrm{CE}},
\end{equation}
where $p^{T}$ is the softmax at temperature $T = 4$ and $\alpha = 0.5$ \cite{hinton2015distilling}, and compared with a plain fine-tuned student, with three seeds each (Welch's $t$-test, Holm over the eight student--teacher pairs). Deployment cost (parameters, fp32 size, MACs, PyTorch CPU latency at batch 1, median of 200 runs) was measured on an Intel Xeon CPU at 2.20~GHz. Distillation methods for rice disease are the subject of a companion manuscript \cite{agrikd}; here distillation only tests whether the pretraining source changes what a deployable student can reach.

\textbf{Results.} Students distilled from the ImageNet DINO teacher gained $+1.40$ points (MobileNetV3-Small) and $+1.02$ points (EfficientNet-B0) over plain fine-tuning (Table~\ref{tab:s-kd}); the gains were consistent in direction but not significant after Holm correction with three seeds ($p_{\mathrm{Holm}} = 0.21$ and $0.16$), and each teacher was trained with one seed. The from-scratch RiceLeaf-full teacher (82.81\%) gave no gain. The distilled MobileNetV3-Small came within 0.2 points of the ViT-S/16 teacher at one tenth of its single-thread latency and one fourteenth of its size (Table~\ref{tab:s-deploy}, Fig.~\ref{fig:s-pareto}); the distilled EfficientNet-B0 exceeded its teacher by 0.45 points. Latency was measured on a server CPU, not on a phone or embedded board.

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

\begin{figure}[h]
\centering
\includegraphics[width=0.55\textwidth]{figures/supp/fig_pareto.pdf}
\caption{\textbf{Test accuracy against single-thread CPU latency} for plain fine-tuned (open) and best distilled (filled) students on RiceLeafDiseaseBD, and the ViT-S/16 teacher.}
\label{fig:s-pareto}
\end{figure}

\clearpage
\section{Near-duplicate detector agreement}
\label{sec:s2}
Table~\ref{tab:s-agree} reports, for each dataset, how many test images each detector flags and how well the two detectors agree, together with how often a flagged test image shares its label with its nearest training or validation image. A detector that finds true near-duplicates should flag pairs that almost always share a label; the cosine detector did so in 94--100\% of cases on every dataset except PlantDoc, whereas pHash-flagged pairs shared a label in only 47.8\% of cases on PlantVillage and 14.5\% on CUB, because pHash flags different objects on similar uniform backgrounds (Figs.~\ref{fig:s-PlantVillage}--\ref{fig:s-CUB}). Pairs flagged only by the cosine detector were the same scene photographed with a small shift in framing or scale, which changes 14--18 of the 64 hash bits. The six RiceLeafDiseaseBD images flagged only by pHash were same-scene pairs just below the cosine threshold. On PlantDoc, flagged pairs included resized copies of the same stock photograph, and 11 of the 21 cosine-flagged pairs carried different labels in training and test, i.e., the same image appears under two classes. Exact-copy (MD5) counts are included in every de-duplicated set.

\begin{table}[h]
\centering\small
\caption{\textbf{Agreement of the two near-duplicate detectors.} Counts of test images flagged at cosine $\geq 0.95$ (ImageNet DINO features) and at pHash Hamming $\leq 8$; Jaccard index and Cohen's $\kappa$ of the two flags; percentage of flagged test images whose nearest training/validation image has the same label.}
\label{tab:s-agree}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
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
\caption{\textbf{Linear probe}: difference (pp, 95\% CI) under each detector and threshold.}
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
\caption{\textbf{k-NN}: difference (pp, 95\% CI) under each detector and threshold.}
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

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
open('supplement.tex', 'w').write(tex)
print('supplement.tex written')
