# Work Plan — Ricardo

> **Role:** Lead Engineer — Feature Extraction Pipeline, Evaluation, Utonia Comparison  
> **Strength:** Strong PyTorch / deep learning implementation  
> **Primary branches:** `feat/feature-extraction`, `feat/evaluation`, `exp/utonia-comparison`

---

## Week 1: Feature Extraction at Scale & CLIP Baselines

**Goal:** All Concerto features extracted for S3DIS Area 5, evaluation framework ready.

### Environment Setup & CLIP Baseline
- [ ] Set up personal Colab environment: verify spconv, torch, open_clip install (parallel with Leonardo)
- [ ] Write `src/clip_utils.py` — CLIP text embedding utilities (coordinate with Leonardo — he may start this too):
  - Load CLIP ViT-B/32 model
  - Function: `get_text_embedding(text: str) → torch.Tensor` (512-dim)
  - Function: `get_class_embeddings(class_names: List[str], templates: List[str]) → Dict[str, torch.Tensor]`
  - Precompute and save CLIP embeddings for all 13 S3DIS classes using multiple templates
  - Save to Drive: `data/s3dis_processed/label_to_clip_embeddings.npy`
- [ ] Review Concerto paper Table 5 (language probing section) — note which metrics and baselines to reproduce

### Batch Feature Extraction
- [ ] Write `scripts/extract_features.py`:
  - Uses Leonardo's `src/encoder.py` wrapper
  - Iterates over S3DIS Area 5 rooms
  - Extracts per-point Concerto features
  - Saves per-room `.npz` files to Drive: `features/s3dis_area5/room_N.npz`
  - Handles interruptions: skip rooms that already have saved features (resume-safe)
- [ ] Extract features for **all rooms in Area 5** across multiple Colab sessions:
  - Session 1: Rooms 1–5
  - Session 2: Rooms 6–10 (or however many there are)
  - Each room saved independently → session drops only lose one room
- [ ] Verify feature quality: check shapes, no NaN values, consistent dimensionality
- [ ] Write `notebooks/02_feature_extraction.ipynb`: Colab notebook for feature extraction
- [ ] **Merge `feat/feature-extraction` → `dev`** via PR (Leonardo reviews)

### Week 1 Deliverables
- ✅ CLIP text embeddings precomputed for all S3DIS classes
- ✅ Concerto features extracted for all S3DIS Area 5 rooms
- ✅ Features saved to Drive, resume-safe extraction script
- ✅ `scripts/extract_features.py`, `notebooks/02_*` merged to `dev`

---

## Week 2: Evaluation Framework & Quantitative Results

**Goal:** Full evaluation pipeline producing mIoU, top-k accuracy, and comparison tables.

### Evaluation Script
- [ ] Write `src/evaluate.py`:
  - Load pre-extracted features and ground-truth labels
  - Load trained MLP checkpoint
  - Run MLP on features → predicted CLIP embeddings
  - Compute cosine similarity between predicted embeddings and all class CLIP embeddings
  - Assign each point to the highest-similarity class
  - Compute metrics:
    - **mIoU** (mean Intersection over Union, 13 classes)
    - **Overall Accuracy (OA)**
    - **Per-class IoU** (table)
    - **Top-k accuracy** (for k=1, 3, 5: what % of points have the correct class in top-k)
- [ ] Write `configs/eval_s3dis.yaml` with paths and settings
- [ ] Handle edge cases: classes with 0 predictions, very small classes

### Run Evaluation & Iterate with Leonardo
- [ ] Evaluate Leonardo's first MLP checkpoint (`mlp_v1_mse_loss_epoch50.pth`)
  - Report mIoU and per-class breakdown to Leonardo
  - Identify which classes perform worst — investigate why
- [ ] Evaluate subsequent checkpoints as Leonardo iterates
- [ ] Write `notebooks/04_evaluate.ipynb` (coordinate with Matteo — he'll extend it with visualizations)
- [ ] Produce a comparison table:
  | Method | mIoU | OA | Best Class | Worst Class |
  |--------|------|----|------------|-------------|
  | Linear probe (baseline) | ? | ? | ? | ? |
  | MLP v1 (MSE loss) | ? | ? | ? | ? |
  | MLP v2 (cosine loss) | ? | ? | ? | ? |

### Open-Vocabulary Evaluation
- [ ] Test the MLP with **novel text queries** not in the 13 S3DIS classes:
  - "fire extinguisher", "whiteboard", "computer monitor", "bookshelf", "trash can"
  - This tests whether the CLIP alignment generalizes beyond training classes
- [ ] Design a qualitative evaluation protocol:
  - For each novel query, visualize the heatmap and manually assess quality (good / partial / fail)
  - Document results in a table
- [ ] **Merge `feat/evaluation` → `dev`** via PR (Leonardo reviews)

### Week 2 Deliverables
- ✅ Evaluation pipeline producing mIoU, OA, per-class IoU, top-k accuracy
- ✅ Results on at least 2 MLP variants
- ✅ Open-vocabulary qualitative evaluation on novel queries
- ✅ `src/evaluate.py`, `configs/eval_s3dis.yaml`, `notebooks/04_*` merged to `dev`

---

## Week 3: Polycam Evaluation, Utonia (Optional), & Presentation

**Goal:** Evaluate on Polycam scans, optionally compare with Utonia, finalize results for presentation.

### Polycam Evaluation
- [ ] Extract Concerto features for Adrian's Polycam scans (use existing extraction pipeline)
- [ ] Run MLP + evaluation on Polycam scans
  - Note: no ground truth for Polycam → qualitative evaluation only
  - Generate heatmaps for common queries: "chair", "table", "wall", "floor"
  - Assess domain gap: are features noisier? Do queries work?
- [ ] Document findings:
  - Which queries work on Polycam?
  - How does point density / RGB quality affect results?
  - Is the domain gap severe?

### Utonia Comparison (Optional — Stretch Goal)
- [ ] **Decision gate:** Check if Utonia weights are publicly available
  - If YES and time permits:
    - [ ] Download Utonia weights
    - [ ] Adapt Leonardo's `src/encoder.py` to support Utonia (or write `src/encoder_utonia.py`)
    - [ ] Extract Utonia features for Area 5 and Polycam scans
    - [ ] Train a separate MLP for Utonia features (if time) or use the same MLP architecture
    - [ ] Compare Concerto vs. Utonia on both S3DIS and Polycam
  - If NO or no time:
    - [ ] Skip Utonia — document in presentation as "future work"
    - [ ] Focus on polishing existing results and helping Matteo with figures

### Final Results & Figures
- [ ] Compile all quantitative results into final tables
- [ ] Generate final figures for presentation (work with Matteo)
- [ ] Save all results to Drive: `results/final_metrics.json`, `results/figures/`

### Presentation
- [ ] Prepare 3–4 slides for your section:
  - Evaluation methodology: metrics, protocol
  - Quantitative results: mIoU table, per-class breakdown
  - Open-vocabulary results: novel query heatmaps
  - Polycam results: domain gap analysis
  - (Optional) Utonia comparison
- [ ] Practice your part (3–4 minutes)
- [ ] Review all slides for consistency
- [ ] Be prepared for Q&A on: evaluation metrics, generalization, limitations

### Week 3 Deliverables
- ✅ Polycam evaluation complete (qualitative)
- ✅ (Optional) Utonia comparison results
- ✅ Final results compiled
- ✅ Presentation slides done

---

## Key Responsibilities Summary

| What | When | Depends On |
|------|------|------------|
| CLIP text embeddings | Days 1–2 | — |
| Batch feature extraction | Days 3–5 | Leonardo's encoder wrapper, Adrian's data |
| Evaluation script | Days 6–8 | Extracted features, Leonardo's MLP checkpoint |
| Run evaluation + iterate | Days 8–9 | Leonardo's training runs |
| Open-vocabulary evaluation | Day 10 | Trained MLP |
| Polycam evaluation | Days 11–12 | Adrian's Polycam export |
| Utonia comparison (optional) | Days 12–13 | Utonia weights availability |
| Final results + figures | Day 13 | All evaluations done |
| Presentation | Days 14–15 | Matteo's slide template |

---

## Dependencies on Others

- **Leonardo:** Encoder wrapper, MLP checkpoints
- **Adrian:** Preprocessed S3DIS data, Polycam scans
- **Matteo:** Visualization utilities for evaluation notebook, slide template

## What Others Depend on From You

- **Leonardo:** Evaluation results to guide MLP iteration
- **Matteo:** Quantitative results and per-class breakdowns for visualization
- **Adrian:** Feature extraction on Polycam scans
- **All:** Final results tables for presentation
