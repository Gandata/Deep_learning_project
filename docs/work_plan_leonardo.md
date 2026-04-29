# Work Plan — Leonardo

> **Role:** Encoder Integration, MLP Architecture, Training Pipeline  
> **Strength:** Strong PyTorch / deep learning implementation  
> **Primary branches:** `feat/encoder-integration`, `feat/mlp-training`

---

## Week 1: Foundation & Encoder Integration

**Goal:** Concerto Small loads and produces per-point features on Colab.

### Environment Setup & Encoder Loading
- [ ] Set up the GitHub repo structure (`src/`, `configs/`, `notebooks/`, `scripts/`, `tests/`)
- [ ] Create `requirements.txt` with pinned versions (torch, spconv-cu118, open_clip_torch, open3d, pointcept)
- [ ] Test spconv installation on a clean Colab T4 session — **this is the highest-risk dependency**
  - If `spconv-cu118` fails, try `spconv-cu121` or compile from source
  - Document the exact install sequence that works
- [ ] Load Concerto Small weights directly using `hf_token` via the huggingface_hub library. Do not save base models to Drive.
- [ ] Write `src/encoder.py`: minimal wrapper that loads Concerto Small and runs a forward pass on a dummy point cloud
- [ ] Verify output shape and feature dimensionality (should be D=256 per point)

### Feature Extraction Pipeline
- [ ] Write `src/encoder.py` to handle real S3DIS data (coordinate with Adrian on data format)
- [ ] Implement chunking/batching for large scenes that don't fit in VRAM
- [ ] Extract features for 1–2 test rooms from S3DIS Area 5 → save to Drive as `.npz`
- [ ] Verify features look reasonable (sanity check: PCA visualization of per-point features)
- [ ] **Merge `feat/encoder-integration` → `dev`** via PR (Ricardo reviews)

### MLP Architecture Design
- [ ] Write `src/translation_head.py`: MLP architecture
  - Input: Concerto feature dim (256)
  - Hidden layers: 2–3 layers (256 → 512 → 512 → 512)
  - Activation: ReLU / GELU
  - Dropout: 0.1–0.3
  - Output: CLIP embedding dim (512 for ViT-B/32)
- [ ] Write `src/clip_utils.py`: function to compute CLIP text embeddings for S3DIS class labels
  - Use multiple templates: "a photo of a {}", "a 3D point cloud of a {}", "a {}"
  - Average across templates → target embedding per class
- [ ] Add smoke test in `tests/test_translation_head.py`: MLP produces correct output shape

### Week 1 Deliverables
- ✅ Concerto loads and runs on Colab T4
- ✅ Feature extraction works on S3DIS rooms
- ✅ MLP architecture defined and tested
- ✅ `src/encoder.py`, `src/translation_head.py`, `src/clip_utils.py` merged to `dev`

---

## Week 2: Training & Iteration

**Goal:** MLP trained on S3DIS, achieving meaningful mIoU on Area 5.

### Training Loop
- [ ] Write `src/train.py`: training loop for the MLP head
  - Load pre-extracted Concerto features (`.npz` from Drive)
  - Load CLIP text embeddings for S3DIS labels
  - Train MLP to predict CLIP embedding from Concerto feature, supervised by ground-truth label
  - Losses to try: MSE, cosine embedding loss, or a combination
  - Optimizer: AdamW, lr=1e-3, weight decay=1e-4
  - Save checkpoint every 5 epochs to Drive
- [ ] Write `configs/train_mlp_s3dis.yaml` with all hyperparameters
- [ ] Write `notebooks/03_train_mlp.ipynb`: Colab notebook for training

### Training Runs & Iteration
- [ ] First training run (MSE loss, 50 epochs)
  - Target: loss decreases steadily, no NaN/Inf
  - Save `mlp_v1_mse_loss_epoch50.pth` to Drive
- [ ] Evaluate with Ricardo's evaluation script (coordinate!)
  - If mIoU < 35%: try cosine embedding loss
  - If mIoU < 35% with cosine loss: try deeper MLP (4 layers), feature normalization
- [ ] Second training run with best configuration
  - Save `mlp_v2_best_config.pth` to Drive
- [ ] **Merge `feat/mlp-training` → `dev`** via PR

### Integration Testing
- [ ] End-to-end test: raw point cloud → Concerto features → MLP → CLIP query → heatmap
- [ ] Verify the full pipeline runs on Colab without OOM
- [ ] Help Ricardo with any evaluation issues
- [ ] Help Adrian with Polycam feature extraction if needed

### Week 2 Deliverables
- ✅ MLP trained with best loss/architecture
- ✅ mIoU on S3DIS Area 5 computed (target: ≥45%)
- ✅ End-to-end pipeline verified
- ✅ `src/train.py`, `configs/`, `notebooks/03_*` merged to `dev`

---

## Week 3: Polish, Demo & Presentation

**Goal:** Pipeline robust, demo working, presentation ready.

### Polycam Integration & Robustness
- [ ] Run feature extraction on Adrian's Polycam scans
- [ ] Run MLP + CLIP query on Polycam scans — assess quality
- [ ] If quality is poor: try feature normalization, temperature scaling, or prompt engineering
- [ ] **(Optional) Utonia comparison:** If Ricardo is integrating Utonia, help with encoder wrapper adaptation

### Demo & Code Cleanup
- [ ] Review and polish `notebooks/05_demo.ipynb` (Adrian's notebook)
- [ ] Ensure the demo runs reliably on Colab (handle edge cases, add error messages)
- [ ] Code cleanup: consistent docstrings, type hints, remove dead code
- [ ] **Merge everything to `main`** via `dev` → `main` PR

### Presentation
- [ ] Prepare 3–4 slides for your section:
  - Concerto encoder: what it is, why frozen
  - MLP translation head: architecture, loss, training details
  - Key implementation challenges and solutions
- [ ] Practice your part of the presentation (3–4 minutes)
- [ ] **Record a backup demo video** in case Colab fails during the live presentation
- [ ] Review all slides for consistency

### Week 3 Deliverables
- ✅ Pipeline works on both S3DIS and Polycam scans
- ✅ Code cleaned up and merged to `main`
- ✅ Presentation slides done
- ✅ Backup demo video recorded

---

## Key Responsibilities Summary

| What | When | Depends On |
|------|------|------------|
| spconv install verification | Day 1 | — |
| Concerto encoder wrapper | Days 1–3 | spconv working |
| Feature extraction pipeline | Days 3–4 | Adrian's data prep |
| MLP architecture + training loop | Days 5–9 | Extracted features |
| Training iteration | Days 8–9 | Ricardo's evaluation script |
| End-to-end integration | Day 10 | All components |
| Polycam feature extraction | Days 11–12 | Adrian's Polycam export |
| Code cleanup + demo polish | Day 13 | — |
| Presentation | Days 14–15 | Matteo's slide template |

---

## Dependencies on Others

- **Adrian:** S3DIS preprocessed data, Polycam scans
- **Ricardo:** Evaluation script, feature extraction help
- **Matteo:** Slide template

## What Others Depend on From You

- **Ricardo:** Working encoder wrapper, trained MLP checkpoint
- **Adrian:** Feature extraction script, pipeline for demo
- **Matteo:** Evaluation results for visualization
