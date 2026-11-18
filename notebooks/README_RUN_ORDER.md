# SSL_AgriStudy re-run notebooks — run order

All notebooks write the same shared library (`sslcommon.py`) in their first cell, so every
experiment uses one strict backbone loader, one split definition, one linear-probe protocol and
one set of statistics. New results go to `SSL_AgriStudy/results_v2/` (old results are not touched
and are never read).

| Step | Notebook | GPU time (L4, rough) | Needed for |
|---|---|---|---|
| 1 | 00_setup_and_sanity_check | 30–45 min | everything — must pass before anything else |
| 2 | 01_linear_probe_transfer_matrix | 3–6 h | LP table + domain-vs-ImageNet matrix + scaling |
| 2 (parallel session) | 02_pretrain_domain_dino_vanilla | 2–4 h per run, 3 runs | clean FloraBD-full and continued-pretraining checkpoints |
| 3 | re-run 01 after 02 finishes | ~1 h | adds the new checkpoints |
| 4 | 03_label_efficiency | 1–2 h | label-efficiency curves with real variance |
| 5 | 04_finetune_ssl_backbones | 25–40 h total, resumable | fine-tune table |
| 6 | 06_distillation_and_deployment | 8–12 h | KD + latency/size/MACs + ONNX |
| optional | 05_supervised_baselines_OPTIONAL | 15–25 h | only if the paper keeps the CNN-vs-ViT claim |
| last | 07_tables_and_figures | minutes (CPU ok) | all paper tables (LaTeX/CSV) and figures |

GPU-time figures are estimates from the timings in your July logs, not measurements.

Rules
* Never edit numbers by hand; every paper number comes from notebook 07.
* If notebook 00's sanity assertion fails, stop — the pipeline is not trustworthy.
* Every run is saved as soon as it finishes; after a disconnect, just re-run the notebook.
* Parallel sessions are safe: every run is saved to its own small file, feature caches are
  written atomically, and checkpoints from notebook 02 are used only after it writes COMPLETE.
* 04 can be split across sessions with `ONLY_BACKBONES = [...]` (use different lists per session).
* Record the GPU type used (printed in every notebook) in the paper's Methods section.

## Revision notebooks (reviewer concerns)

| Step | Notebook | GPU time (rough, L4) | Concern |
|---|---|---|---|
| R1 | 08_imagenet_subset_controls | ~13 h | 1a domain vs amount of data |
| R1 (parallel) | 09_pretraining_seeds | ~8 h | 3 pretraining variance |
| R2 | 10_budget_recipe_and_continued_variants | ~19 h (each run optional) | 1b, 2, 5 |
| R3 | re-run 01 (adds the new checkpoints; finished results skipped) | ~1–2 h | — |
| R4 | 11_knn_plantdoc_and_duplicates | ~1–2 h | 4, 6, 8 |
| R5 | 12_revision_analysis (CPU) | minutes | tables/figures |

Notebook 11's duplicate check and PlantDoc/k-NN parts can run right away (they only need notebook 01's features).
All notebooks in this zip carry the updated shared library (new checkpoint names); use them all, not a mix.

## Second revision round (16 Nov 2026)

| Step | Notebook | Runtime | Concern |
|---|---|---|---|
| S1 | 15_clip_and_supervised_references | GPU, ~1 h (L4) | single SSL method: CLIP ViT-B/16, supervised DeiT-S/16 and ResNet-50 as frozen references (LP + k-NN, six targets) |
| S2 | 14_duplicate_detector_robustness_CPU | **CPU runtime**, ~20–40 min, no GPU units | duplicate detector relies on DINO features: adds a model-free pHash detector, detector agreement, sensitivity of key comparisons/slopes to detector and threshold, k-NN on de-duplicated test sets |

* Both need notebook 11 to have run (PlantDoc on Drive, `11_duplicates.json`, ImageNet-DINO feature cache).
* Run 15 **before** 14 if you can: 14 then also reports the three new references on the de-duplicated test sets.
  In the other order nothing breaks; just re-run 14 afterwards (it reuses its cached hashes and takes minutes).
* 14 replaces `12b_knn_dedup` — you do not need to run 12b.
* Built-in check: in 14's sensitivity table the "cos ≥ 0.95 (paper)" column must reproduce the de-duplicated
  numbers already in the paper (68.09 / 79.04 on RiceLeaf, etc.), because it uses notebook 11's exact list.
* Bring back: `paper_v2/tables/rev_dup_*.csv`, `rev_knn_dedup*.csv`, `rev_dedup_all_backbones.csv`,
  `rev_references_*.csv`, and `paper_v2/figures/rev_dup_detector_agreement.pdf` + `rev_dup_disagree_*.png`.
* Both end with the Drive flush cell — wait for the ✓ before closing the runtime.

## Third revision round (17 Nov 2026) — about 1.5 GPU-hours in total

| Step | Notebook | Runtime | Why |
|---|---|---|---|
| T1 | re-run **01_linear_probe_transfer_matrix** | GPU, ~30–45 min | picks up the three forgetting-aware variants from notebook 10 (`rice_cont_fullhead`, `rice_cont_fullhead_lowlr`, `rice_cont_replay`), which finished pretraining but were never evaluated; everything else is skipped |
| T2 | re-run **11_knn_plantdoc_and_duplicates** | GPU, ~20–30 min | k-NN and PlantDoc for the same three variants (finished results skipped) |
| T3 | **16_compute_throughput_energy** | GPU, ~20–30 min | measured seconds/step and GPU power for each DINO configuration → GPU-hours and kWh per arm |
| T4 | **14_duplicate_detector_robustness_CPU** (new version) | CPU, ~10 min (hashes are cached) | new cells 7–8: variants vs ImageNet DINO and vs standard recipe (own Holm family, full and de-duplicated), and a check that the paper's 135-comparison family is unchanged |

* T1 and T3 can run in parallel sessions; T2 after T1 is safest (it does not need T1, but run order keeps logs simple).
* Re-running 01 enlarges the Holm family inside `01_lp_vs_imagenet.json`; the paper keeps the 135-comparison family,
  which cell 8 of notebook 14 recomputes (it must reproduce p = 0.039 / 0.003 / 0.21).
* Bring back: `paper_v2/tables/rev_continued_variants_long.csv`, `rev_continued_variants_acc.csv`,
  `rev_lp_family135.csv`, `rev_compute_measured.csv` (and notebook 14/16 outputs).
