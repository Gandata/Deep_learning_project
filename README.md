# Open-World Text-Based 3D Object Search

> Open-vocabulary 3D object search using frozen point cloud encoders (Concerto / Utonia) and a CLIP-aligned MLP translation head. Validated on S3DIS Area 5 and in-the-wild Polycam scans.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Setup & Installation (Colab)](#setup--installation-colab)
- [Data Preparation](#data-preparation)
- [Training the Translation Head](#training-the-translation-head)
- [Evaluation](#evaluation)
- [In-the-Wild Demo](#in-the-wild-demo)
- [Team & Acknowledgements](#team--acknowledgements)
- [References](#references)

---

## Overview

This project builds and evaluates a pipeline for **open-world text-based object search in 3D point cloud scenes**. The core idea:

1. **Frozen 3D encoder** — We use **Concerto Small** (39M params, pretrained on large-scale 3D data) to extract per-point features from indoor scans. The backbone is never finetuned.
2. **CLIP-aligned MLP translation head** — A lightweight 2–3 layer MLP maps Concerto's 3D feature space into **CLIP's text embedding space**. This is trained with supervision on S3DIS label↔CLIP-text-embedding pairs.
3. **Open-vocabulary querying** — At inference, a user provides a free-text query (e.g., "red chair", "lamp", "whiteboard"). The query is embedded by CLIP's text encoder and matched against the translated per-point features via cosine similarity, producing a heatmap over the point cloud.
4. **In-the-wild generalization** — We test the pipeline on at least one scene captured with a mobile LiDAR/photogrammetry app (Polycam), exported as `.ply`.

**Optional extension:** If Concerto's cross-domain generalization proves insufficient on out-of-domain scans, we compare with **Utonia** — a newer encoder (March 2026) with cross-domain robustness — contingent on public weight availability.

---

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Point Cloud  │────▶│  Concerto Small  │────▶│  Per-point 3D    │
│  (XYZ + RGB)  │     │  (frozen, 39M)   │     │  features (D=256)│
└──────────────┘     └─────────────────┘     └───────┬──────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ MLP Translation│
                                              │ Head (trainable│
                                              │ 2–3 layers)    │
                                              └───────┬───────┘
                                                      │
                                                      ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Text query   │────▶│  CLIP Text       │────▶│  Cosine sim /    │
│  "red chair"  │     │  Encoder (frozen)│     │  heatmap on PC   │
└──────────────┘     └─────────────────┘     └──────────────────┘
```

---

## Repository Structure

```
Deep_learning_project/
├── README.md                      # This file
├── LICENSE
├── .gitignore
├── pyproject.toml                 # Project dependencies (uv)
│
├── docs/                          # Documentation & papers
│   ├── Concerto.pdf
│   ├── Utonia.pdf
│   ├── repository_guide.md        # Repo conventions, branching, version control
│   ├── collaboration_plan.md      # GitHub + Drive workflow
│   ├── feasibility_evaluation.md  # Risk assessment
│   ├── work_plan_leonardo.md      # Per-person task sheets
│   ├── work_plan_ricardo.md
│   ├── work_plan_adrian.md
│   └── work_plan_matteo.md
│
├── configs/                       # Training & eval config files (YAML)
│   ├── train_mlp_s3dis.yaml
│   └── eval_s3dis.yaml
│
├── src/                           # Core Python source code
│   ├── __init__.py
│   ├── encoder.py                 # Concerto feature extraction wrapper
│   ├── translation_head.py        # MLP definition & forward pass
│   ├── clip_utils.py              # CLIP text embedding helpers
│   ├── dataset.py                 # S3DIS + Polycam dataset loaders
│   ├── train.py                   # Training loop for the MLP head
│   ├── evaluate.py                # Quantitative evaluation (mIoU, top-k)
│   └── visualize.py               # 3D heatmap visualization utilities
│
├── notebooks/                     # Colab notebooks (one per workflow)
│   ├── 01_setup_and_data.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_train_mlp.ipynb
│   ├── 04_evaluate.ipynb
│   └── 05_demo.ipynb
│
├── scripts/                       # CLI utility scripts
│   ├── extract_features.py        # Batch feature extraction
│   ├── prepare_s3dis.py           # S3DIS preprocessing
│   └── export_polycam.py          # Polycam .ply → pipeline format
│
├── tests/                         # Unit & smoke tests
│   └── test_translation_head.py
│
└── presentation/                  # Final slides & demo materials
    └── .gitkeep
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.1 | Core framework |
| `pointcept` | latest | Concerto encoder & data utilities |
| `open_clip_torch` | ≥ 2.24 | CLIP text encoder |
| `spconv-cu118` / `spconv-cu12x` | ≥ 2.3 | Sparse convolution backend; exact wheel depends on the active Colab CUDA runtime |
| `open3d` | ≥ 0.18 | Point cloud I/O & visualization |
| `numpy`, `scipy` | latest | Numerical utilities |
| `pyyaml` | latest | Config parsing |
| `wandb` *(optional)* | latest | Experiment tracking |
| `plotly` *(optional)* | latest | Interactive 3D visualization |

> A `pyproject.toml` is provided. On Colab, install with:
> ```bash
> !pip install uv
> !uv pip install --system -e .
> ```
>
> `notebooks/03_train_mlp.ipynb` now skips Concerto/spconv entirely when pre-extracted `features/s3dis_area5/*.npz` already exist, and otherwise auto-tests a few `spconv` wheels before failing.

---

## Setup & Installation (Colab)

```python
# 1. Clone the repo
!git clone https://github.com/Gandata/Deep_learning_project.git
%cd Deep_learning_project

# 2. Install dependencies
!pip install uv
!uv pip install --system -e .

# 3. Mount Google Drive (for data & checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# 4. Symlink data
!ln -s /content/drive/MyDrive/DL_Project/data ./data
!ln -s /content/drive/MyDrive/DL_Project/checkpoints ./checkpoints
```

---

## Data Preparation

### S3DIS Area 5

1. Download S3DIS from the [Stanford website](http://buildingparser.stanford.edu/dataset.html) (requires form).
2. Place the raw data in `Drive > DL_Project > data > s3dis_raw/`.
3. Run preprocessing:
   ```bash
   python scripts/prepare_s3dis.py --input data/s3dis_raw --output data/s3dis_processed
   ```

### Polycam In-the-Wild Scan

1. Capture a scene using [Polycam](https://poly.cam/) on iOS/Android.
2. Export as `.ply` (point cloud mode, with RGB).
3. Place in `Drive > DL_Project > data > polycam/`.

---

## Training the Translation Head

```bash
python src/train.py --config configs/train_mlp_s3dis.yaml
```

Key hyperparameters (see config):
- **MLP layers:** 3 (256 → 512 → 512 → 512, with ReLU + dropout)
- **Loss:** MSE or cosine embedding loss between predicted embeddings and CLIP text embeddings of ground-truth labels
- **Optimizer:** AdamW, lr=1e-3, weight decay=1e-4
- **Epochs:** 50–100 (early stopping on val loss)
- **Batch size:** Adjusted to fit T4 VRAM (~15GB)

---

## Evaluation

```bash
python src/evaluate.py --config configs/eval_s3dis.yaml --split area5
```

**Metrics:**
- **mIoU** (semantic segmentation via nearest-label matching)
- **Top-k retrieval accuracy** (given a text query, what % of top-k points belong to the correct class)
- **Qualitative heatmaps** (per-query 3D visualizations)

---

## In-the-Wild Demo

The `notebooks/05_demo.ipynb` notebook provides an interactive demo:
1. Load a Polycam `.ply` scan
2. Extract Concerto features (frozen)
3. Apply the trained MLP translation head
4. Enter a free-text query → visualize the heatmap on the 3D scene

---

## Team & Acknowledgements

| Member | Role |
|--------|------|
| **Ricardo** | Lead engineer — feature extraction, evaluation, Utonia comparison |
| **Leonardo** | Encoder integration, MLP architecture, training pipeline |
| **Adrian** | Data preparation, Polycam pipeline, demo notebook |
| **Matteo** | Evaluation scripts, visualization, presentation & slides |

**Course:** Deep Learning — Master's program  
**Compute:** Google Colab free tier (NVIDIA T4 GPU)

---

## References

1. **Concerto:** *Concerto: Cooperative Contrastive Pretraining for 3D Point Cloud Understanding* — [GitHub](https://github.com/Pointcept/Concerto) | [HuggingFace](https://huggingface.co/spaces/pointcept-bot/Concerto)
2. **Utonia:** *Utonia: Universal 3D Tokenization via Neural Codec* — [GitHub](https://github.com/Pointcept/Utonia) | [HuggingFace](https://huggingface.co/spaces/pointcept-bot/Utonia)
3. **CLIP:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021
4. **S3DIS:** Armeni et al., *3D Semantic Parsing of Large-Scale Indoor Spaces*, CVPR 2016
5. **Pointcept:** [github.com/Pointcept/Pointcept](https://github.com/Pointcept/Pointcept)
