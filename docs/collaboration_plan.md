# GitHub + Google Drive Collaboration Plan

> How four people work in parallel without overwriting each other, managing code on GitHub and data/checkpoints on Google Drive.

---

## 1. The Two-System Split

| What | Where | Why |
|------|-------|-----|
| Code (`.py`, `.ipynb`, `.yaml`, `.md`) | **GitHub** | Version control, PRs, code review |
| Data (S3DIS, Polycam `.ply`, preprocessed) | **Google Drive** | Too large for git, persistent across Colab sessions |
| Checkpoints (`.pth`, `.pt`) | **Google Drive** | Large binary files, need persistence |
| Extracted features (`.npy`, `.npz`) | **Google Drive** | Intermediate artifacts, shared across team |
| Final figures/slides | **Google Drive** (editable) + **GitHub** (final) | Live editing in Slides, final exports committed |

### Golden Rule
> **If it's code or config → GitHub. If it's data or a binary artifact → Drive.**

---

## 2. Google Drive Structure

Create a **shared team folder** in Google Drive:

```
DL_Project/                                    (Shared with all 4 members)
├── data/
│   ├── s3dis_raw/                             # Original S3DIS download
│   │   └── Stanford3dDataset_v1.2/
│   ├── s3dis_processed/                       # After preprocessing script
│   │   ├── Area_5/
│   │   │   ├── points.npy
│   │   │   ├── colors.npy
│   │   │   └── labels.npy
│   │   └── label_to_clip_embeddings.npy       # Precomputed CLIP embeds for 13 classes
│   ├── polycam/                               # In-the-wild scans
│   │   ├── scan_001.ply
│   │   └── scan_002.ply
│   └── README_data.txt                        # Notes on data versions/sources
│
├── features/                                  # Extracted Concerto features
│   ├── s3dis_area5_concerto_small.npz
│   └── polycam_scan_001_concerto_small.npz
│
├── checkpoints/                               # Trained MLP weights
│   ├── mlp_v1_epoch50.pth
│   ├── mlp_v2_cosine_loss.pth
│   └── best_model.pth
│
├── results/                                   # Evaluation outputs
│   ├── metrics_s3dis_area5.json
│   ├── heatmaps/
│   │   ├── query_chair.html
│   │   └── query_lamp.html
│   └── figures/                               # For presentation
│
├── pretrained/                                # Downloaded pretrained weights
│   ├── concerto_small.pth                     # From HuggingFace
│   └── clip_vitb32.pth                        # If needed locally
│
└── presentation/                              # Google Slides + exports
    ├── slides_link.txt                        # Link to Google Slides doc
    └── exported_figures/
```

### Drive Access Rules

- **All 4 members** have Editor access to the shared `DL_Project/` folder.
- **Never delete files** — rename with suffix `_deprecated` or move to a `_archive/` subfolder.
- **Use clear naming** with version tags:
  - `mlp_v1_epoch50.pth` not `model.pth`
  - `s3dis_area5_concerto_small_v2.npz` not `features.npz`
- **Add a `README_data.txt`** in each subfolder noting what the files are and when they were created.

---

## 3. Colab ↔ Drive ↔ GitHub Workflow

Every Colab session should start with this pattern:

```python
# === 1. Mount Drive ===
from google.colab import drive
drive.mount('/content/drive')

# === 2. Clone repo (or pull latest) ===
import os
REPO_DIR = '/content/Deep_learning_project'
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/Gandata/Deep_learning_project.git {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull origin dev

%cd {REPO_DIR}

# === 3. Install dependencies ===
!pip install -q -r requirements.txt

# === 4. Symlink Drive data ===
!ln -sf /content/drive/MyDrive/DL_Project/data ./data
!ln -sf /content/drive/MyDrive/DL_Project/checkpoints ./checkpoints
!ln -sf /content/drive/MyDrive/DL_Project/features ./features
!ln -sf /content/drive/MyDrive/DL_Project/pretrained ./pretrained
```

### Saving Work from Colab

**For code changes:**
```python
# Commit and push from Colab
!cd /content/Deep_learning_project && \
  git add -A && \
  git commit -m "feat(train): add cosine loss option" && \
  git push origin feat/mlp-training
```

**For data/checkpoints:**
```python
# Save directly to Drive (already symlinked)
torch.save(model.state_dict(), './checkpoints/mlp_v2_cosine_loss.pth')
# This writes to Drive automatically via symlink
```

---

## 4. Parallel Work — Avoiding Conflicts

### File Ownership

Each person primarily owns specific files. This minimizes merge conflicts:

| Person | Primary files | Shared files (coordinate!) |
|--------|---------------|----------------------------|
| **Leonardo** | `src/encoder.py`, `src/translation_head.py`, `src/train.py` | `configs/*.yaml`, `requirements.txt` |
| **Ricardo** | `src/evaluate.py`, `scripts/extract_features.py` | `configs/*.yaml`, `src/dataset.py` |
| **Adrian** | `scripts/prepare_s3dis.py`, `scripts/export_polycam.py`, `notebooks/01_*.ipynb`, `notebooks/05_*.ipynb` | `src/dataset.py` |
| **Matteo** | `src/visualize.py`, `notebooks/04_*.ipynb`, `presentation/` | `src/evaluate.py` (metrics display) |

### Rules for Shared Files

1. **`src/dataset.py`** — Adrian creates the initial version. Ricardo and Leonardo add their specific data loading needs via PR.
2. **`configs/*.yaml`** — One person updates, others pull before modifying.
3. **`requirements.txt`** — Only update on `dev` via PR. Never on feature branches independently.

### Communication Protocol

- **Before editing a shared file:** Post in the team chat: "I'm editing `src/dataset.py` on `feat/data-prep`"
- **Daily (async):** Each person posts a 2-line status in the team chat:
  - What I did today
  - What I'm doing tomorrow / any blockers
- **Weekly sync (15 min):** Quick call to merge `dev` → `main` and align on next week's goals.

---

## 5. Branching in Practice

### Scenario: Leonardo and Ricardo both need to modify `configs/`

1. Leonardo works on `feat/mlp-training`, adds `configs/train_mlp_s3dis.yaml`.
2. Ricardo works on `feat/evaluation`, adds `configs/eval_s3dis.yaml`.
3. Both open PRs to `dev`. Since they're editing **different files** in `configs/`, no conflict.
4. If both need to edit the **same** config file, whoever merges second rebases and resolves.

### Scenario: Adrian creates `src/dataset.py`, Ricardo needs to extend it

1. Adrian creates `src/dataset.py` with S3DIS loading on `feat/data-prep`.
2. Adrian merges to `dev`.
3. Ricardo pulls `dev` into `feat/evaluation`:
   ```bash
   git checkout feat/evaluation
   git merge dev
   ```
4. Ricardo extends `src/dataset.py` with evaluation-specific data loading.

### Scenario: Notebook conflict (avoid this!)

Notebooks should **never** be edited by two people simultaneously. Each notebook has a single owner:

| Notebook | Owner |
|----------|-------|
| `01_setup_and_data.ipynb` | Adrian |
| `02_feature_extraction.ipynb` | Ricardo |
| `03_train_mlp.ipynb` | Leonardo |
| `04_evaluate.ipynb` | Matteo |
| `05_demo.ipynb` | Adrian |

---

## 6. Checkpoint & Feature Versioning on Drive

Since Drive doesn't have git-style versioning, we use **naming conventions**:

```
<artifact>_<version>_<key_detail>.ext
```

**Examples:**
```
mlp_v1_mse_loss_epoch50.pth
mlp_v2_cosine_loss_epoch30.pth
mlp_v3_deeper_arch_epoch80.pth
s3dis_area5_concerto_small_v1.npz
polycam_scan001_concerto_small_v1.npz
```

**Keep a changelog** in `DL_Project/checkpoints/CHANGELOG.txt`:
```
2026-04-18  Leonardo  mlp_v1_mse_loss_epoch50.pth    - First training run, MSE loss, 50 epochs, mIoU=42.3
2026-04-20  Leonardo  mlp_v2_cosine_loss_epoch30.pth  - Switched to cosine loss, mIoU=48.1
2026-04-23  Leonardo  mlp_v3_deeper_arch_epoch80.pth  - Added 3rd layer, mIoU=51.7
```

---

## 7. Git Configuration for the Team

Each team member should set up:

```bash
# Identity
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Default branch
git config init.defaultBranch main

# Auto-rebase on pull (avoids merge commits on pull)
git config pull.rebase true

# nbstripout (strip notebook outputs)
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

### GitHub Repository Settings (Admin)

- [ ] Set `main` as default branch
- [ ] Enable branch protection on `main`: require 1 review, no force push
- [ ] Enable branch protection on `dev`: no force push
- [ ] Add all 4 members as collaborators with Write access

---

## 8. Emergency Procedures

### "I accidentally committed a large file"

```bash
# Remove from git history (if not yet pushed)
git rm --cached path/to/large_file.pth
echo "*.pth" >> .gitignore
git add .gitignore
git commit --amend --no-edit

# If already pushed, use BFG or git filter-branch
# Ask Leonardo or Ricardo for help
```

### "My Colab session died mid-training"

1. Check if checkpoint was saved to Drive (should save every N epochs).
2. Resume from latest checkpoint:
   ```python
   model.load_state_dict(torch.load('./checkpoints/mlp_v2_cosine_loss_epoch30.pth'))
   # Continue training from epoch 31
   ```

### "I have a merge conflict in a notebook"

1. **Do not try to resolve it manually** (JSON merge conflicts are horrible).
2. Pick one version:
   ```bash
   git checkout --theirs notebooks/03_train_mlp.ipynb  # or --ours
   ```
3. Re-apply your changes manually in Colab.

---

## 9. Summary Cheat Sheet

| Task | Tool | Command |
|------|------|---------|
| Start working | Colab | Mount Drive, clone/pull repo, install deps, symlink |
| Save code | Git | `git add`, `commit`, `push` to feature branch |
| Save data/checkpoints | Drive | Save to symlinked Drive path |
| Share code | GitHub | Open PR to `dev`, get review, merge |
| Share data | Drive | Save to shared `DL_Project/` folder |
| Track experiments | Drive | Naming convention + CHANGELOG.txt |
| Coordinate | Chat | Daily async status + weekly sync call |
