"""Builds supplementary_data/holm_families.csv: every member comparison of every Holm family in the paper."""
import os, json, numpy as np, pandas as pd
D = os.environ.get('SSL_SUPP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'supplementary_data'))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); a[j] = min(1.0, run)
    return a
rows = []
def add(fam, size, analysis, protocol, test_set, A, B, target, p, ph, test):
    rows.append(dict(family=fam, family_size=size, analysis=analysis, protocol=protocol, test_set=test_set,
                     model=A, reference=B, target=target, test=test, p=p, p_holm=ph))
ORIG15 = ['dinov2_vits14', 'dinov2_vitb14', 'mae_vitb16', 'florabd_1k', 'florabd_5k', 'florabd_10k', 'florabd_full_scratch',
          'rice_1k', 'rice_5k', 'rice_full', 'pv_5k', 'pv_10k', 'pv_full', 'florabd_cont_in', 'rice_cont_in']
F = pd.read_csv(f'{D}/_src_lp_family135.csv')
assert len(F) == 135 and F.backbone.nunique() == 27
f75 = F[F.backbone.isin(ORIG15)].copy(); assert len(f75) == 75
f75['ph75'] = holm(f75.p)
for r in f75.itertuples(): add('LP-75', 75, 'Table 5 (original 15 backbones)', 'linear probe', 'full', r.backbone, 'imagenet_dino', r.target, r.p, r.ph75, 'exact McNemar')
for r in F.itertuples(): add('LP-135', 135, 'all other linear-probe comparisons (27 backbones)', 'linear probe', 'full', r.backbone, 'imagenet_dino', r.target, r.p, r.p_holm, 'exact McNemar')
G = pd.read_csv(f'{D}/_src_domain_vs_generic.csv'); assert len(G) == 20
REF = {996: 'inet_n996', 4671: 'inet_n4671', 6835: 'inet_n6835', 30829: 'inet_n30829'}
for r in G.itertuples(): add('GEN-20', 20, 'local vs generic (Table 6)', 'linear probe', 'full', r._2, REF[int(r.images)], r.target, r.p_mcnemar, r.p_holm, 'exact McNemar')
C = pd.read_csv(f'{D}/continued_variants_long.csv')
for (pr, det), g in C.groupby(['protocol', 'detector']):
    assert len(g) == 36
    for r in g.itertuples(): add(f'CPT-36-{pr}-{det}', 36, 'forgetting-aware variants (Table 8)', {'lp': 'linear probe', 'knn': 'k-NN'}[pr],
                                 {'none': 'full', 'cos0.95': 'dedup cos0.95'}[det], r.variant, r.reference, r.target, r.p, r.p_holm, 'exact McNemar')
Rf = pd.read_csv(f'{D}/_src_references_long.csv')
for pr, g in Rf.groupby('protocol'):
    assert len(g) == 18
    for r in g.itertuples(): add(f'REF-18-{pr}', 18, 'non-DINO references (Table 10)', {'lp': 'linear probe', 'knn': 'k-NN'}[pr], 'full', r.backbone, 'imagenet_dino', r.target, r.p, r.p_holm, 'exact McNemar')
L = json.load(open(f'{D}/label_efficiency_tests_published.json')); assert len(L) == 90
for r in L: add('LE-90', 90, 'label efficiency (Fig. 4, Table S10)', 'label-efficiency probe', 'full', r['backbone'], 'imagenet_dino', f"{r['target']} @ {int(round(r['fraction']*100))}%", r['p_paired_t'], r['p_holm'], 'paired t over 20 subsets')
T = json.load(open(f'{D}/_src_finetune_tests.json')); assert len(T) == 21
for r in T: add('FT-21', 21, 'fine-tuning (Table 12, Table S9)', 'fine-tuning', 'full', r['backbone'], 'imagenet_dino', r['target'], r['p_welch'], r['p_holm'], "Welch's t over 3 seeds")
K = pd.read_csv(f'{D}/kd.csv'); assert len(K) == 8
for r in K.itertuples(): add('KD-8', 8, 'distillation (Section S1)', 'distillation', 'full', f'{r.student} <- {r.teacher}', f'{r.student} (plain fine-tuning)', r.target, r.p_welch, r.p_holm, "Welch's t over 3 seeds")
out = pd.DataFrame(rows); out.to_csv(f'{D}/holm_families.csv', index=False)
print(out.groupby(['family', 'family_size']).size().to_string())
