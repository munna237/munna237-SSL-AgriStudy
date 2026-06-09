<div align="center">

# When Does Local Data Beat ImageNet?
## A Domain-Specific SSL Study for Agricultural Visual Recognition in Bangladesh

[![Paper](https://img.shields.io/badge/Paper-Smart_Agricultural_Technology-green)]()
[![Dataset FloraBD](https://img.shields.io/badge/Dataset-FloraBD_Mendeley-blue)](https://data.mendeley.com)
[![Dataset RiceLeaf](https://img.shields.io/badge/Dataset-RiceLeafDiseaseBD-blue)](https://doi.org/10.17632/86s4jzj2m4.2)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)]()

**Mohammad Mehedi Hasan Munna¹, Reefka Fabliha Tulona¹,
Acramul Haque Kabir²**

¹ East West University, Dhaka, Bangladesh
² North South University, Dhaka, Bangladesh

</div>

---

## 🌟 Hero Result

> **DINO pretrained on FloraBD (13,598 images) achieves
> 96.31% on Oxford Flowers-102 — surpassing ImageNet DINO
> (95.20%) using 99× fewer pretraining images.**

This is the first demonstration that locally collected
South Asian botanical data can exceed ImageNet-scale
pretraining for fine-grained botanical recognition.

---

## 📋 Overview

This repository contains the complete code, results, and
figures for our systematic empirical study of domain-specific
self-supervised pretraining for South Asian agricultural
fine-grained visual recognition.

We conduct **140 experiments** across:
- 4 SSL models (DINO, DINOv2-S, DINOv2-B, MAE)
- 4 supervised baselines (ResNet-50, EfficientNet-B0,
  ViT-Small, Swin-Tiny)
- 5 datasets (FloraBD, RiceLeafDiseaseBD, PlantVillage,
  Flowers-102, CUB-200-2011)
- 10 domain-specific DINO pretraining checkpoints
  (1K to 30K images)
- 55-entry transfer matrix

---

## 🔑 Key Findings

| Finding | Result |
|---------|--------|
| **Hero result** | FloraBD DINO (13,598 img) → 96.31% on Flowers-102 vs ImageNet DINO 95.20% |
| **Crossover threshold** | Domain-specific pretraining beats ImageNet at ALL scales ≥1,000 images |
| **Real field > Lab** | FloraBD scales better than PlantVillage (lab data saturates at 10K) |
| **Cross-domain transfer** | FloraBD DINO improves RiceLeafBD by +17.11pp over ImageNet DINO |
| **Architecture gap** | ViT-Small beats ResNet-50 by +21.88pp on real field RiceLeafBD |
| **Token subsampling** | Random = Similarity-based (p=0.72) — criterion is irrelevant |
| **DINOv2 gap** | DINOv2 ViT-S beats DINO ViT-S by +24.84pp on FloraBD (same arch) |

---

## 📊 Main Results

### Phase 2: Pretrained SSL — Linear Probe

| Model | FloraBD | RiceLeafBD | PlantVillage | Flowers-102 | CUB-200 |
|-------|---------|------------|--------------|-------------|---------|
| DINO ViT-S/16 | 75.13% | 44.16% | 64.72% | 25.00% | 4.64% |
| DINOv2 ViT-S/14 | **99.97%** | 78.06% | 99.20% | 99.48% | 87.19% |
| DINOv2 ViT-B/14 | **99.97%** | **83.02%** | **99.47%** | **99.66%** | **89.42%** |
| MAE ViT-B/16 | 96.74% | 57.20% | 96.08% | 59.70% | 25.22% |

### Phase 3: Supervised Baselines

| Model | FloraBD | RiceLeafBD | PlantVillage | Flowers-102 | CUB-200 |
|-------|---------|------------|--------------|-------------|---------|
| ResNet-50 | 99.62% | 70.04% | 97.76% | 66.63% | 59.46% |
| EfficientNet-B0 | 99.83% | 76.43% | 99.18% | 83.67% | 70.62% |
| ViT-Small/16 | **99.97%** | **91.92%** | **99.94%** | **99.27%** | **85.11%** |
| Swin-Tiny | **99.97%** | 91.64% | 99.88% | 96.91% | 84.24% |

### Phase 4: Domain-Specific Pretraining — Scaling Curves

| Source | Size | →Flowers-102 | →FloraBD | →RiceLeafBD |
|--------|------|-------------|---------|------------|
| FloraBD | 1K | 27.47%★ | 86.48%★ | 54.82%★ |
| FloraBD | 5K | 35.70%★ | 93.00%★ | 59.58%★ |
| FloraBD | 10K | 39.91%★ | 94.07%★ | 60.05%★ |
| **FloraBD** | **13,598** | **96.31%★** | **99.28%★** | **60.45%★** |
| RiceLeaf | Full | 35.14%★ | 90.81%★ | 60.80%★ |
| PlantVillage | Full | 33.52%★ | 89.16%★ | 56.52%★ |
| ImageNet DINO | 1.28M | 23.53% | 73.31% | 43.34% |

★ = Beats ImageNet DINO reference

---

## 🗂️ Datasets

### FloraBD (Novel — Ours)
- **19,426 images** | **45 species** | Dhaka, Bangladesh
- Real field conditions, smartphone cameras
- Publicly available: [Mendeley Data](https://data.mendeley.com)
  (DOI pending)
- Split: 13,598 train / 2,913 val / 2,915 test

### RiceLeafDiseaseBD (Novel — Ours)
- **9,769 images** | **6 classes** | Gazipur, Bangladesh
- Real paddy field conditions
- 74,836 bounding box annotations (YOLO format)
- Publicly available:
  [![DOI](https://img.shields.io/badge/DOI-10.17632%2F86s4jzj2m4.2-blue)](https://doi.org/10.17632/86s4jzj2m4.2)
- Split: 6,835 train / 1,462 val / 1,472 test

### External Datasets
| Dataset | Images | Classes | Download |
|---------|--------|---------|----------|
| PlantVillage | 44,014 | 38 | [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Oxford Flowers-102 | 8,189 | 102 | [Official](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| CUB-200-2011 | 11,788 | 200 | [Official](http://www.vision.caltech.edu/datasets/cub_200_2011/) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/munna237/SSL-AgriStudy.git
cd SSL-AgriStudy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download pretrained checkpoints

```bash
# DINO ViT-S/16 (ImageNet)
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
     -O checkpoints/pretrained/dino_vits16.pth

# MAE ViT-B/16 (ImageNet)
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth \
     -O checkpoints/pretrained/mae_vitbase16.pth

# DINOv2 (via torch.hub — downloaded automatically)
```

### 4. Download domain-specific checkpoints

Our trained domain-specific DINO checkpoints are available
on Google Drive:

| Checkpoint | Size | Download |
|-----------|------|----------|
| FloraBD-1K | 704 MB | [Google Drive]() |
| FloraBD-5K | 704 MB | [Google Drive]() |
| FloraBD-10K | 704 MB | [Google Drive]() |
| FloraBD-Full | 704 MB | [Google Drive]() |
| RiceLeaf-1K | 704 MB | [Google Drive]() |
| RiceLeaf-5K | 704 MB | [Google Drive]() |
| RiceLeaf-Full | 704 MB | [Google Drive]() |
| PlantVil-5K | 704 MB | [Google Drive]() |
| PlantVil-10K | 704 MB | [Google Drive]() |
| PlantVil-Full | 704 MB | [Google Drive]() |

### 5. Organise datasets

datasets/
├── FloraBD/
│   ├── train/  (13,598 images, 45 classes)
│   ├── val/    (2,913 images)
│   └── test/   (2,915 images)
├── RiceLeafDiseaseBD/
│   ├── train/  (6,835 images, 6 classes)
│   ├── val/    (1,462 images)
│   └── test/   (1,472 images)
├── PlantVillage/
│   ├── train/  (30,829 images, 38 classes)
│   ├── val/
│   └── test/
├── Flowers102/
│   ├── train/  (2,040 images, 102 classes)
│   └── val/    (6,149 images)
└── CUB/
├── train/  (5,994 images, 200 classes)
└── val/    (5,794 images)

---

## 🧪 Reproducing Experiments

### Phase 2: Evaluate pretrained SSL models

```bash
# Linear probe — DINO on FloraBD
python code/evaluate.py \
    --model dino_vits16 \
    --checkpoint checkpoints/pretrained/dino_vits16.pth \
    --data_path datasets/FloraBD \
    --output_dir results/linear/FloraBD \
    --protocol linear \
    --num_classes 45 \
    --epochs 100

# Fine-tune — DINOv2 on RiceLeafDiseaseBD
python code/evaluate.py \
    --model dinov2_vits14 \
    --data_path datasets/RiceLeafDiseaseBD \
    --output_dir results/finetune/RiceLeafDiseaseBD \
    --protocol finetune \
    --num_classes 6 \
    --epochs 50
```

### Phase 4: Domain-specific DINO pretraining

```bash
# Pretrain DINO on FloraBD from scratch
cd dino/
python main_dino.py \
    --arch vit_small \
    --data_path ../datasets/FloraBD/train \
    --output_dir ../checkpoints/domain_dino/florabd_full \
    --epochs 100 \
    --batch_size_per_gpu 64 \
    --lr 0.0005 \
    --seed 42
```

### Run all experiments (batch)

```bash
# Phase 2 (40 experiments)
bash scripts/phase2_ssl_eval.sh

# Phase 3 (20 experiments)
bash scripts/phase3_supervised.sh

# Phase 4a (10 pretraining runs)
bash scripts/phase4_pretrain.sh

# Phase 4b (55 transfer evaluations)
bash scripts/phase4_transfer.sh

# Phase 5 (15 ablation runs)
bash scripts/phase5_ablation.sh
```

---

## 📈 Generate All Figures

```bash
python code/generate_figures.py
# Saves all 11 figures to figures/ directory
```

---

## 📁 Repository Contents
SSL-AgriStudy/
├── code/                   # Python scripts
│   ├── evaluate.py         # Unified evaluation (Phase 2+3)
│   ├── generate_figures.py # All 11 publication figures
│   ├── make_subsets.py     # Create dataset subsets
│   └── token_ablation/
│       └── morph_sieve.py  # Token subsampling module
├── configs/                # Hyperparameter configs
├── scripts/                # Bash scripts per phase
├── results/                # All 140 experiment results (JSON)
├── figures/                # All 11 publication figures (PNG+PDF)
├── notebooks/              # Google Colab notebooks
└── docs/                   # Paper (added after acceptance)

---

## 🔢 Experiment Summary

| Phase | Description | Experiments | Results |
|-------|-------------|-------------|---------|
| Phase 2 | Pretrained SSL evaluation | 40 | `results/phase2_results.json` |
| Phase 3 | Supervised baselines | 20 | `results/phase3_results.json` |
| Phase 4a | Domain DINO pretraining | 10 | Checkpoints on Drive |
| Phase 4b | Transfer matrix | 55 | `results/phase4_transfer.json` |
| Phase 5 | Token ablation (5 seeds) | 15 | `results/phase5_ablation.json` |
| **Total** | | **140** | |

---

## 📦 Requirements
torch>=2.0
torchvision>=0.15
timm==0.9.16
umap-learn>=0.5
scikit-learn>=1.0
scipy>=1.9
matplotlib>=3.5
numpy>=1.23
Pillow>=9.0
tqdm>=4.64

---

## 📄 Citation

If you use this code, datasets, or findings in your
research, please cite:

```bibtex
@article{munna2025ssl_agri,
  title   = {When Does Local Data Beat {ImageNet}?
             A Domain-Specific {SSL} Study for
             Agricultural Visual Recognition
             in {Bangladesh}},
  author  = {Munna, Mohammad Mehedi Hasan and
             Tulona, Reefka Fabliha and
             Kabir, Acramul Haque},
  journal = {Smart Agricultural Technology},
  year    = {2025},
  note    = {Under review}
}
```

For FloraBD dataset:
```bibtex
@misc{munna2025florabd,
  author    = {Munna, Mohammad Mehedi Hasan and others},
  title     = {{FloraBD}: A Large-Scale {B}angladeshi
               Flower Image Dataset for Fine-Grained
               Visual Recognition},
  publisher = {Mendeley Data},
  year      = {2025}
}
```

For RiceLeafDiseaseBD dataset:
```bibtex
@misc{munna2026riceleafbd,
  author    = {Munna, Mohammad Mehedi Hasan and others},
  title     = {{RiceLeafDiseaseBD}: A Field-Based
               Annotated Smartphone Image Dataset of
               Healthy and Diseased Rice Leaves
               from {Bangladesh}},
  publisher = {Mendeley Data},
  doi       = {10.17632/86s4jzj2m4.2},
  year      = {2026}
}
```

---

## 📬 Contact

**Mohammad Mehedi Hasan Munna**
BSc in Computer Science and Engineering
East West University, Dhaka, Bangladesh

---

## 📜 License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

<div align="center">
<strong>If this work helps your research, please ⭐ the
repository and cite our paper.</strong>
</div>
