# Work Plan — Adrian

> **Role:** Data Pipeline & Demo — S3DIS preprocessing, Polycam export, demo notebook  
> **Strength:** Capable PyTorch skills, good at practical/hands-on tasks  
> **Primary branches:** `feat/data-prep`, `feat/demo`

---

## Week 1: Data Acquisition & Preprocessing

**Goal:** S3DIS Area 5 preprocessed and ready for feature extraction. Polycam test scan captured.

### Day 1–2: S3DIS Data Acquisition
- [ ] Submit the S3DIS download request form on the [Stanford website](http://buildingparser.stanford.edu/dataset.html)
  - **Critical:** This requires manual approval and can take 24–48h. Do this FIRST.
  - Use a university email for faster approval
- [ ] While waiting for S3DIS approval, set up the shared Google Drive structure:
  ```
  DL_Project/
  ├── data/
  │   ├── s3dis_raw/
  │   ├── s3dis_processed/
  │   └── polycam/
  ├── features/
  ├── checkpoints/
  ├── pretrained/
  ├── results/
  └── presentation/
  ```
- [ ] Share the Drive folder with all team members (Editor access)
- [ ] Set up Colab environment: test spconv install (parallel with Leonardo and Ricardo)
- [ ] Create `notebooks/01_setup_and_data.ipynb` with the standard Colab setup boilerplate:
  - Drive mount
  - Repo clone/pull
  - Dependency install
  - Symlink creation

### Day 3–4: S3DIS Preprocessing
- [ ] Download S3DIS dataset once approved → upload to Drive (`data/s3dis_raw/`)
  - **Storage concern:** S3DIS is ~14GB. If Drive free tier is insufficient:
    - Use a team member's university Google Workspace account
    - Or process areas one at a time and delete raw data after processing
- [ ] Write `scripts/prepare_s3dis.py`:
  - Parse S3DIS raw format (each room as separate `.txt` files per object class)
  - Convert to unified format per room: `points.npy` (XYZ), `colors.npy` (RGB), `labels.npy` (class IDs)
  - Focus on **Area 5 only** (this is the standard test split)
  - Reference Pointcept's S3DIS processing script for the correct format
- [ ] Run preprocessing → save to Drive (`data/s3dis_processed/Area_5/`)
- [ ] Verify: load processed data, check shapes, visualize a room in Open3D
- [ ] Create a `data/s3dis_processed/README_data.txt` documenting:
  - Source and version of S3DIS
  - Processing date
  - Number of rooms, number of points per room
  - Class label mapping (0=ceiling, 1=floor, ...)

### Day 5: Polycam Test Capture
- [ ] Download [Polycam](https://poly.cam/) on iOS or Android
- [ ] Capture a test scan of an indoor scene (e.g., your room, a lab, a common area)
  - Use LiDAR mode if device supports it (iPhone Pro/iPad Pro)
  - Otherwise use photogrammetry mode (take many photos)
- [ ] Export as `.ply` file (Point Cloud mode, with RGB colors)
- [ ] Upload to Drive (`data/polycam/scan_001.ply`)
- [ ] Write `scripts/export_polycam.py`:
  - Load `.ply` with Open3D
  - Extract XYZ + RGB arrays
  - Optionally subsample to match S3DIS point density (~1000 points/m²)
  - Save in the same format as S3DIS processed data
- [ ] Test: visualize the Polycam scan, verify it looks reasonable
- [ ] **Merge `feat/data-prep` → `dev`** via PR (Matteo reviews)

### Week 1 Deliverables
- ✅ S3DIS Area 5 preprocessed and on Drive
- ✅ Polycam test scan captured and converted
- ✅ Shared Drive structure set up
- ✅ `scripts/prepare_s3dis.py`, `scripts/export_polycam.py`, `notebooks/01_*` merged to `dev`

---

## Week 2: Dataset Integration & Demo Prototype

**Goal:** `src/dataset.py` working, demo notebook prototype ready.

### Day 6–7: Dataset Loader
- [ ] Write `src/dataset.py`:
  - `S3DISDataset` class: loads preprocessed rooms (points, colors, labels)
    - Returns dict: `{"points": np.array, "colors": np.array, "labels": np.array}`
    - Supports loading from Drive path
    - Supports filtering by area (Area 5 for test)
  - `PolycamDataset` class: loads exported Polycam scans
    - Same interface as `S3DISDataset` but no labels
  - `FeatureDataset` class: loads pre-extracted features + labels for MLP training
    - Returns: `{"features": np.array, "labels": np.array}`
    - Used by Leonardo's training loop
- [ ] Test all dataset classes on Colab
- [ ] Coordinate with Leonardo and Ricardo: confirm the interface they need
- [ ] **Merge changes to `dev`** via PR

### Day 8–9: Demo Notebook Prototype
- [ ] Write `notebooks/05_demo.ipynb` (prototype):
  - Cell 1: Setup (Drive mount, repo clone, install deps)
  - Cell 2: Load a point cloud (S3DIS room or Polycam scan)
  - Cell 3: Extract Concerto features (using Leonardo's encoder)
  - Cell 4: Run MLP translation head (using Leonardo's MLP)
  - Cell 5: Enter a text query → compute CLIP embedding
  - Cell 6: Compute cosine similarity heatmap
  - Cell 7: Visualize the heatmap on the 3D point cloud
    - Use Matteo's visualization utilities (coordinate!)
    - Fallback: use simple Open3D or Plotly scatter3d
- [ ] Test with placeholder/dummy data if real pipeline isn't ready yet
- [ ] Make the demo **interactive**: user types a query in a text input cell

### Day 10: Additional Polycam Scans (if needed)
- [ ] Capture 1–2 more Polycam scans of different indoor scenes for variety
- [ ] Process and upload to Drive
- [ ] Test the export pipeline on different scan qualities
- [ ] Help Matteo with evaluation visualization if needed
- [ ] **Merge `feat/demo` prototype → `dev`** via PR

### Week 2 Deliverables
- ✅ `src/dataset.py` with S3DIS, Polycam, and Feature dataset loaders
- ✅ Demo notebook prototype (may use placeholder data)
- ✅ 2–3 Polycam scans captured and processed

---

## Week 3: Demo Polish & Presentation Support

**Goal:** Demo notebook fully working and polished. Presentation materials ready.

### Day 11–12: Demo Finalization
- [ ] Update `notebooks/05_demo.ipynb` with real trained MLP checkpoint (from Leonardo)
- [ ] Test end-to-end demo on both S3DIS and Polycam scans
- [ ] Add polish:
  - Clear instructions in markdown cells
  - Error handling (what if CLIP model fails to load, what if features aren't extracted)
  - Multiple example queries with expected outputs
  - Side-by-side comparison: S3DIS query vs. Polycam query
- [ ] Optimize for presentation: minimize loading time, preload models
- [ ] **Record a backup demo video** (screen recording of the notebook running):
  - Use Loom, OBS, or Colab's built-in screen recording
  - Show: loading a scene → typing a query → heatmap appears
  - Save to Drive: `presentation/demo_video.mp4`

### Day 13: Demo Documentation
- [ ] Write clear instructions in `notebooks/05_demo.ipynb` for anyone to reproduce:
  - What to install
  - What files to have on Drive
  - What to type
  - Expected output
- [ ] Test that someone else on the team (e.g., Matteo) can run the demo without help
- [ ] Merge final demo to `dev` → `main`

### Day 14–15: Presentation
- [ ] Prepare 2–3 slides for your section:
  - Data pipeline: S3DIS, what it is, how we preprocessed it
  - Polycam: how we captured in-the-wild scans, the export process
  - Demo walkthrough: what the user sees (screenshots)
- [ ] **Run the live demo** during the presentation (or play the backup video)
- [ ] Practice your part (2–3 minutes)
- [ ] Be prepared for Q&A on: data preprocessing, Polycam compatibility, demo details

### Week 3 Deliverables
- ✅ Demo notebook fully working on S3DIS + Polycam
- ✅ Backup demo video recorded
- ✅ Presentation slides done
- ✅ Demo tested by another team member

---

## Key Responsibilities Summary

| What | When | Depends On |
|------|------|------------|
| S3DIS download request | Day 1 | — |
| Drive folder setup | Day 1–2 | — |
| S3DIS preprocessing | Days 3–4 | S3DIS download approval |
| Polycam test capture | Day 5 | Polycam app |
| `src/dataset.py` | Days 6–7 | Preprocessed data |
| Demo notebook prototype | Days 8–9 | Leonardo's encoder + MLP (can use placeholders) |
| Additional Polycam scans | Day 10 | — |
| Demo finalization | Days 11–12 | Leonardo's trained MLP, Matteo's viz utils |
| Backup demo video | Day 12 | Working demo |
| Presentation | Days 14–15 | Matteo's slide template |

---

## Dependencies on Others

- **Leonardo:** Encoder wrapper (for demo, by Day 8), trained MLP checkpoint (for demo, by Day 11)
- **Ricardo:** Feature extraction on Polycam scans (by Day 12)
- **Matteo:** Visualization utilities for demo (by Day 8)

## What Others Depend on From You

- **Leonardo + Ricardo:** Preprocessed S3DIS data (by Day 3–4)
- **Ricardo:** Polycam scans exported (by Day 5, more by Day 10)
- **All:** Working demo (by Day 12)
- **All:** Shared Drive structure (by Day 2)

---

## Tips & Notes

- **S3DIS download is the #1 blocker for the whole team.** Submit the form within hours of project kickoff.
- **Test Polycam export early** — if `.ply` export doesn't include RGB, try a different export format or a different scanning app.
- **Keep the demo notebook simple.** It should be a showcase, not a development environment. All logic lives in `src/`.
- **Communicate proactively** with Leonardo about the data format: agree on the interface (what shape, what dtype, what file format) before writing the dataset loader.
