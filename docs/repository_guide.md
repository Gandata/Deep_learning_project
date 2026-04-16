# Repository Guide

> Conventions, folder layout, branching strategy, and version control best practices for the team.

---

## 1. Repository Name & Description

- **Name:** `Deep_learning_project` (already set)
- **GitHub Description:** *"Open-vocabulary 3D object search using frozen Concerto/Utonia encoders and a CLIP-aligned MLP translation head. Validated on S3DIS and in-the-wild Polycam scans."*
- **Topics/Tags:** `deep-learning`, `point-cloud`, `clip`, `3d-search`, `concerto`, `open-vocabulary`, `s3dis`

---

## 2. Folder Structure Rationale

```
Deep_learning_project/
├── README.md                 # Entry point — what, why, how
├── requirements.txt          # Pinned deps (torch, open_clip, spconv, etc.)
├── .gitignore                # Excludes data/, checkpoints/, *.ply, *.pth
│
├── docs/                     # All planning & reference documents
├── configs/                  # YAML configs (training, eval)
├── src/                      # All Python modules — importable, testable
├── notebooks/                # Colab notebooks (numbered workflow steps)
├── scripts/                  # Standalone CLI scripts
├── tests/                    # Unit tests
└── presentation/             # Slides and demo assets
```

### Why this structure?

| Decision | Reason |
|----------|--------|
| **`src/` for modules** | Keeps code importable (`from src.encoder import ...`), testable, and separate from notebooks |
| **`notebooks/` numbered** | Enforces a clear execution order; each notebook imports from `src/` |
| **`configs/` in YAML** | Avoids hardcoded hyperparams; easy to diff and track in git |
| **`scripts/` separate** | One-off preprocessing/export scripts that aren't part of the core pipeline |
| **`tests/` present** | Even minimal smoke tests catch breakage early (e.g., MLP output shape) |
| **No data in repo** | Data lives on Google Drive, symlinked at runtime. Git ignores `data/` and `checkpoints/` |

---

## 3. Branching Strategy

We use a simplified **GitHub Flow** adapted for a 4-person, 3-week sprint:

### Branches

| Branch | Purpose | Protected? |
|--------|---------|------------|
| `main` | Stable, working code. Only merged via PR with at least 1 review. | ✅ Yes |
| `dev` | Integration branch. All feature branches merge here first. Weekly sync to `main`. | ⚠️ Soft-protected (no force push) |
| `feat/encoder-integration` | Leonardo — Concerto encoder wrapper + feature extraction | No |
| `feat/mlp-training` | Leonardo — MLP head architecture + training loop | No |
| `feat/feature-extraction` | Ricardo — Batch feature extraction pipeline | No |
| `feat/evaluation` | Ricardo — Evaluation scripts + metrics | No |
| `feat/data-prep` | Adrian — S3DIS preprocessing + Polycam export | No |
| `feat/demo` | Adrian — Demo notebook + interactive visualization | No |
| `feat/visualization` | Matteo — 3D heatmap visualization utilities | No |
| `feat/presentation` | Matteo — Slides, figures, final report | No |
| `fix/*` | Bug fixes (e.g., `fix/spconv-install`) | No |
| `exp/*` | Experimental branches (e.g., `exp/utonia-comparison`) | No |

### Workflow

```
main ◄── dev ◄── feat/encoder-integration
              ◄── feat/mlp-training
              ◄── feat/evaluation
              ◄── feat/data-prep
              ◄── feat/demo
              ◄── feat/visualization
              ◄── feat/presentation
```

1. **Create feature branch** from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feat/my-feature
   ```

2. **Work on your branch**, commit frequently with descriptive messages:
   ```bash
   git add -A
   git commit -m "feat(encoder): add Concerto Small wrapper with frozen weights"
   ```

3. **Push and open a PR** to `dev`:
   ```bash
   git push origin feat/my-feature
   # Open PR on GitHub: feat/my-feature → dev
   ```

4. **Get at least 1 review** (Leonardo or Ricardo review each other; Adrian and Matteo review each other or get reviewed by a lead).

5. **Merge to `dev`** via squash merge (keeps history clean).

6. **Weekly: merge `dev` → `main`** after the team confirms everything works.

---

## 4. Commit Message Convention

Use **Conventional Commits** for clarity:

```
<type>(<scope>): <short description>

[optional body]
```

**Types:**
- `feat` — New feature or functionality
- `fix` — Bug fix
- `data` — Data preparation or processing changes
- `docs` — Documentation only
- `refactor` — Code restructuring without behavior change
- `test` — Adding or modifying tests
- `chore` — Build, CI, or tooling changes

**Examples:**
```
feat(encoder): add Concerto Small feature extraction wrapper
fix(train): correct learning rate schedule for cosine annealing
data(s3dis): add Area 5 preprocessing script
docs(readme): update installation instructions for spconv
test(mlp): add smoke test for MLP output dimensions
```

---

## 5. Handling Notebooks in Git

Jupyter notebooks (`.ipynb`) produce **terrible diffs** because they store cell outputs, execution counts, and metadata as JSON. Here's how we manage this:

### Rules

1. **Always clear outputs before committing:**
   ```bash
   # Install nbstripout (one time)
   pip install nbstripout
   nbstripout --install  # Auto-strips on git add
   ```

2. **Alternative: use the Colab "Clear all outputs" menu** before downloading and committing.

3. **Keep notebooks thin:** Notebooks should only contain:
   - Setup cells (imports, Drive mount, symlinks)
   - Calls to `src/` functions
   - Visualization/display cells
   
   **All logic lives in `src/`**, not in notebooks. This makes code reviewable and testable.

4. **One notebook per workflow step** (numbered `01_`, `02_`, etc.) — no monolithic "do everything" notebooks.

5. **Never merge conflicting notebooks** — if two people edit the same notebook, one person re-applies their changes on top of the other's version. Notebook merge conflicts are essentially unresolvable.

### nbstripout Setup (Recommended)

```bash
# Install globally in the repo
pip install nbstripout
cd Deep_learning_project
nbstripout --install --attributes .gitattributes
git add .gitattributes
git commit -m "chore: add nbstripout for clean notebook diffs"
```

This automatically strips outputs from `.ipynb` files on `git add`, so you never accidentally commit cell outputs.

---

## 6. What NOT to Commit

The `.gitignore` should exclude:

```gitignore
# Data & model artifacts (live on Google Drive)
data/
checkpoints/
outputs/
*.pth
*.pt
*.ckpt
*.ply
*.h5
*.npy
*.npz

# Colab / Jupyter artifacts
.ipynb_checkpoints/
*.ipynb_tmp

# Large files
*.zip
*.tar.gz
*.tar

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

### What TO commit

- All `.py` source files
- All `.ipynb` notebooks (with outputs stripped)
- Config files (`.yaml`)
- `requirements.txt`
- Documentation (`.md`)
- Small visualization outputs (`.png`, `.html`) if under 1MB

---

## 7. Code Style

- **Python 3.10+** (Colab default)
- **PEP 8** formatting — use `black` formatter if desired
- **Type hints** encouraged for function signatures
- **Docstrings** for all public functions (Google style)
- **No hardcoded paths** — use configs or argparse

---

## 8. PR Review Checklist

Before approving a PR, the reviewer should check:

- [ ] Code runs without errors on Colab
- [ ] No hardcoded paths (uses config or CLI args)
- [ ] Notebooks have outputs cleared
- [ ] No data files or checkpoints accidentally included
- [ ] Commit messages follow convention
- [ ] README or docs updated if needed
- [ ] At least one smoke test passes (if applicable)

---

## 9. Release Tags

We tag milestones on `main`:

| Tag | Meaning |
|-----|---------|
| `v0.1` | End of Week 1 — data pipeline + encoder extraction working |
| `v0.2` | End of Week 2 — MLP trained + evaluation on S3DIS |
| `v1.0` | End of Week 3 — Full pipeline + demo + presentation ready |
