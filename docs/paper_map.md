# Paper → notebook map

Numbers refer to the revised manuscript (v8) and its Supplementary Material. "Drive" paths are under
`MyDrive/SSL_AgriStudy/`.

| Paper item | Produced by | Output |
|---|---|---|
| Table 1 Datasets and splits | 00, 11 (PlantDoc) | `results_v2/splits/`, `datasets/PlantDoc/SPLIT_INFO.json` |
| Table 2 Pretraining arms | 02, 08, 09, 10, 10b | `checkpoints/domain_dino*/`, `results_v2/08_10_pretraining_runs/` |
| Table 3 Compute (measured columns) | 16 | `paper_v2/tables/rev_compute_measured.csv` |
| Table 4 Linear probe | 01 → 07 | `tab_linear_probe.csv` |
| Fig. 1 Difference from ImageNet DINO | 07 | `fig_domain_vs_imagenet.pdf` |
| Table 5 Local vs generic at matched budget | 08 → 12 | `rev_domain_vs_generic.csv` |
| Fig. 2 Accuracy vs pretraining images | `scripts/make_figures.py` | `rev_scaling_with_generic.pdf` |
| Table 6 Training length, recipe, step matching | 10 → 12 | `rev_budget_recipe.csv`, `rev_dose_response.csv` |
| Table 7 Pretraining seeds | 09 → 12 | `rev_pretraining_seeds.csv` |
| Table 8 Continued pretraining and variants | 10b → 01 → 11 → 14 (cell 7) | `rev_continued_variants_acc.csv`, `_long.csv` |
| Table 9 k-NN and PlantDoc linear probe | 11 → 12 | `rev_knn.csv`, `rev_plantdoc_lp.csv` |
| Table 10 Non-DINO references | 15 | `rev_references_*.csv` |
| Fig. 3 Detector agreement | 14 (cell 9) | `rev_dup_detector_agreement.pdf` |
| Table 11 Sensitivity to the duplicate detector | 14 | `rev_dup_sensitivity_long.csv` |
| Table 12 Near-duplicate counts | 11 → 12 | `rev_duplicates.csv` |
| Table 13 / Fig. 4 Label efficiency | 03 → 07; `scripts/make_figures.py` | `tab_label_efficiency.csv`, `fig_label_efficiency.pdf` |
| Table 14 Full fine-tuning | 04 → 07 | `tab_finetune.csv`, `tab_finetune_tests.csv` |
| Table 15 Supervised baselines | 05 | `05_supervised_*` results |
| Supp. S1 (Tables S1–S2, Fig. S1) Distillation, deployment | 06 → 07; `scripts/make_figures.py` | `tab_kd.csv`, `tab_deploy.csv` |
| Supp. S2–S4 (Tables S3–S7, Figs. S2–S7) | 14; `scripts/make_supplement.py` | `rev_dup_*.csv`, `rev_knn_dedup.csv`, contact sheets |
| Holm family of 135 linear-probe comparisons | 14 (cell 8) | `rev_lp_family135.csv` |
