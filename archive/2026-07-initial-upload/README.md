# Archived: initial upload (July 2026 pipeline) — superseded

These files are kept for transparency only. **Do not use them to reproduce the paper.**

They come from the first, July 2026 version of the experiments. An audit before submission found problems in
that pipeline, and every experiment was re-run with the corrected code in the repository root:

* The linear-probe/fine-tuning loader dropped every `mlp` key with `strict=False`, so backbones were evaluated
  with missing weights. The corrected loader (`src/sslcommon.py`) is strict.
* The checkpoint labelled "FloraBD-pretrained" had been initialised from ImageNet DINO (and trained with a
  modified objective), so it was not a from-scratch domain model. The clean from-scratch and continued runs are in
  the paper's Table 2.
* Model selection used test data on Flowers-102 and CUB, whose `val` folder is the official test split.

Consequently, the headline in `README_initial.md` (FloraBD DINO 96.31% vs. ImageNet DINO 95.20% on Flowers-102)
does **not** hold. With the corrected pipeline, from-scratch FloraBD-full DINO reaches 62.94% and ImageNet DINO
95.72% on Flowers-102 (paper, Table 4).
