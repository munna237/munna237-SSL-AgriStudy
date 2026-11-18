# SSL-AgriStudy

Code, splits and results for

> **Domain Match Helps, Scale Decides: A Budget-Matched Study of Laboratory-Scale Self-Supervised Pretraining and
> Near-Duplicate Inflation in Agricultural and Botanical Recognition** (IEEE Access, under review).

The study holds the self-supervised method fixed (DINO, ViT-S/16) and asks whether a laboratory should pretrain on its
own unlabeled images or reuse ImageNet-scale representations. Local data and ImageNet subsets are compared at
identical image counts, recipe and optimizer steps. Continued pretraining from ImageNet DINO is tested under four
recipes. All checkpoints and seven released encoders are evaluated on six targets with paired statistics, and the
train–test near-duplicates in the evaluation data are audited with two independent detectors.

> The `archive/2026-07-initial-upload/` folder holds the first upload of this repository, made with an earlier
> pipeline that was later found to be faulty. It is superseded and kept only for transparency; see its README.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Every experiment, as Google Colab notebooks (run order in `notebooks/README_RUN_ORDER.md`) |
| `src/sslcommon.py` | Shared library: strict backbone loading, splits, feature cache, linear probe, statistics, resumable training. Every notebook writes this exact file in its first cell. |
| `src/patch_dino.py` | The only changes made to the official DINO code (commit `7c446df`): `--init_weights`, `--init_full`, `weights_only=False`, xcit hub calls removed |
| `scripts/make_figures.py` | Figs. 2 and 4 and Supplementary Fig. S1 from the result CSVs |
| `scripts/make_supplement.py` | Supplementary tables from the result CSVs |
| `scripts/export_results_from_drive.ipynb` | Exports splits, result files and per-image predictions from Drive, and builds the checkpoint manifest |
| `results/tables/` | CSVs behind the paper's tables |
| `results/` (after export) | `splits/`, result JSONs, `correct/` (per-image correctness for every backbone × dataset × protocol) |
| `docs/paper_map.md` | Which notebook produces each table and figure |

## Requirements

* Google Colab with an NVIDIA L4 (or any ≥ 24 GB GPU) for pretraining and feature extraction; a CPU runtime is
  enough for notebooks 07, 12 and 14.
* Python 3.13, PyTorch 2.14 (CUDA 13.0); see `requirements.txt`. The notebooks install what Colab lacks.
* A Google Drive folder `MyDrive/SSL_AgriStudy/` holding the datasets (below). All results, feature caches and
  checkpoints are written there. Every run is resumable after a disconnect and safe to run in parallel sessions.

## Datasets (not redistributed)

| Dataset | Source | Expected images |
|---|---|---|
| FloraBD | Mendeley Data, doi: [10.17632/r9whm6nz8n.2](https://doi.org/10.17632/r9whm6nz8n.2) | 19,426 |
| RiceLeafDiseaseBD | Mendeley Data, doi: [10.17632/86s4jzj2m4.2](https://doi.org/10.17632/86s4jzj2m4.2) | 9,769 |
| PlantVillage | Hughes & Salathé (2015) | 44,014 |
| PlantDoc | [github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (downloaded by notebook 11) | 2,578 |
| Oxford Flowers-102 | Nilsback & Zisserman (2008) | 8,189 |
| CUB-200-2011 | Wah et al. (2011) | 11,788 |

Place each as `MyDrive/SSL_AgriStudy/datasets/<name>/{train,val,test}/<class>/<image>`. For Flowers-102 and CUB, the
`val` folder holds the official test split, and a 20% validation split is carved from `train` with a fixed seed; the
carved lists are in `results/splits/`. Image counts are checked on every run. The ImageNet-1k subsets (notebook 08)
come from ImageNet train via a Drive copy or the gated Hugging Face dataset (`HF_TOKEN` as a Colab secret).

## Reproducing the paper

Run order, GPU time and what each notebook answers: `notebooks/README_RUN_ORDER.md`. In short:

1. `00` sanity check, then `01` (linear probes) and `02`, `08`, `09`, `10`/`10b` (pretraining; resumable).
2. Re-run `01` after new checkpoints exist (finished results are skipped), then `03`, `04`, `05`, `06`, `11`, `15`.
3. `07`, `12`, `14` (CPU) write every table and figure to `MyDrive/SSL_AgriStudy/paper_v2/`.
4. `16` measures pretraining throughput and GPU energy.

Measured cost on one L4: all laboratory-scale pretraining in the paper takes 87.8 GPU-hours (5.5 kWh of GPU energy);
a single 100-epoch RiceLeaf run takes 2.5 h.

## Pretrained checkpoints

The teacher backbones of all 27 laboratory-scale DINO ViT-S/16 runs (the weights every evaluation uses, ~87 MB each)
are archived on Zenodo: [Zenodo DOI]. `results/checkpoint_manifest.csv` lists each file's SHA-256, final epoch and
training arguments. To reuse one with the notebooks, save `<run>_teacher_backbone.pth` as
`MyDrive/SSL_AgriStudy/checkpoints/domain_dino_v2/<run>/checkpoint.pth` (the nine original runs `florabd_1k`–`pv_full`
go under `checkpoints/domain_dino/`), and add an empty `COMPLETE` file to each `domain_dino_v2` run folder.
`sslcommon.load_backbone` loads them with a strict key check.

## Statistics in brief

* Linear probe: frozen CLS features, StandardScaler, logistic regression with *C* chosen on validation, refit on
  train+validation, test scored once.
* k-NN: cosine, *k* = 20, *T* = 0.07, training split as memory.
* Effect sizes: paired bootstrap 95% CIs over test images (2,000 resamples). Tests: exact McNemar with Holm
  correction within each analysis family (families stated in the paper).
* De-duplicated test sets: exact MD5 copies plus near-duplicates at ImageNet-DINO cosine ≥ 0.95, with pHash (Hamming
  ≤ 4/8/12) and other thresholds as sensitivity checks. Re-scoring uses the saved per-image predictions, so no
  refitting is needed.

## Citation

See `CITATION.cff`. Please also cite DINO (Caron et al., ICCV 2021) and the datasets you use.

## License

Code: MIT (see `LICENSE`). Datasets and third-party models keep their own licences.
