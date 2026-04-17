# Work Plan — Matteo

> **Role:** Evaluation Visualization, Quality Assurance & Presentation Lead  
> **Strength:** Capable PyTorch skills, detail-oriented, good communicator  
> **Primary branches:** `feat/visualization`, `feat/presentation`

---

## Week 1: Visualization Framework & Presentation Setup

**Goal:** Visualization utilities ready, presentation template created, contribution to evaluation design.

### Environment Setup & Research
- [ ] Set up personal Colab environment (parallel with others)
- [ ] Read the Concerto paper (focus on evaluation sections: Table 5, language probing)
- [ ] Read the Utonia paper (brief overview — understand key differences from Concerto)
- [ ] Study prior work on 3D-language alignment for the presentation:
  - OpenScene, LERF, ConceptFusion (brief notes on approach and results)
  - This gives context for the "related work" section of the presentation
- [ ] Note key numbers from Concerto paper to cite in the presentation:
  - Linear probe language mIoU: 44.56%
  - Supervised mIoU: 77.3%
  - Our MLP should land between these

### Visualization Utilities
- [ ] Write `src/visualize.py`:
  - `plot_point_cloud(points, colors, labels=None)` — basic 3D point cloud viewer
    - Use Plotly `scatter3d` for interactive Colab-compatible visualization
    - Color by: RGB, class label, or custom heatmap
  - `plot_heatmap(points, scores, query_text)` — overlay cosine similarity scores on point cloud
    - Color map: blue (low similarity) → red (high similarity)
    - Add title with query text
    - Optional: threshold to highlight top-k% of points
  - `plot_class_comparison(points, pred_labels, gt_labels)` — side-by-side predicted vs. ground truth
  - `save_figure(fig, path)` — export Plotly figures as HTML and PNG
  - Consider lightweight alternatives if Plotly is too heavy:
    - `matplotlib` 3D scatter (less interactive but lighter)
    - `open3d.visualization` (only works locally, not in Colab)
- [ ] Test all visualization functions with dummy data
- [ ] Create a small notebook to demo the visualization utils: `notebooks/viz_test.ipynb` (temporary, not committed)
- [ ] **Merge `feat/visualization` → `dev`** via PR (Adrian reviews)

### Week 1 Deliverables
- ✅ `src/visualize.py` with heatmap and point cloud plotting functions
- ✅ Paper notes for presentation context
- ✅ Visualization utilities tested and merged

---

## Week 2: Evaluation Visualization & Presentation Drafting

**Goal:** Evaluation results beautifully visualized, presentation first draft complete.

### Evaluation Notebook & Figures
- [ ] Extend `notebooks/04_evaluate.ipynb` (Ricardo creates the base version):
  - Add visualization cells that use `src/visualize.py`
  - Per-class IoU bar chart (horizontal bars, sorted by performance)
  - Confusion matrix heatmap (predicted class vs. ground truth)
  - mIoU comparison table (different MLP variants)
  - Qualitative heatmap gallery: grid of 4–6 queries on the same scene
- [ ] Generate publication-quality figures for the presentation:
  - Save all figures to Drive: `results/figures/`
  - Use consistent color scheme across all figures
  - Export as both PNG (for slides) and HTML (for notebook)
- [ ] Work with Ricardo: as he produces evaluation results, you visualize them
  - Quick turnaround: Ricardo gives you metrics → you produce figures → Ricardo/Leonardo iterate

### Presentation First Draft
- [ ] Create presentation structure in Google Slides (shared with team):
  - **Slide 1:** Title + team names
  - **Slide 2:** Problem statement — what is open-vocabulary 3D search?
  - **Slide 3:** Related work — OpenScene, LERF, Concerto (brief)
  - **Slide 4:** Our approach — architecture diagram (from README)
  - **Slide 5:** Concerto encoder — what it does, why frozen (Leonardo's content)
  - **Slide 6:** MLP translation head — architecture, loss (Leonardo's content)
  - **Slide 7:** Data — S3DIS, Area 5, Polycam (Adrian's content)
  - **Slide 8:** Training details — hyperparameters, Colab setup (Leonardo's content)
  - **Slide 9:** Quantitative results — mIoU table, per-class bar chart (Ricardo's content)
  - **Slide 10:** Open-vocabulary results — novel query heatmaps (Ricardo's content)
  - **Slide 11:** In-the-wild results — Polycam heatmaps (Adrian/Ricardo's content)
  - **Slide 12:** (Optional) Utonia comparison (Ricardo's content)
  - **Slide 13:** Live demo (Adrian runs it)
  - **Slide 14:** Limitations & future work
  - **Slide 15:** Q&A / Thank you
- [ ] Create a consistent slide template: university branding (if applicable), clean fonts, consistent colors
- [ ] Add placeholder content for each section — the team fills in their details in Week 3
- [ ] Share the Google Slides link in Drive: `presentation/slides_link.txt`
- [ ] **Merge any notebook updates to `dev`**

### Week 2 Deliverables
- ✅ Evaluation notebook with professional visualizations
- ✅ Figures generated for presentation
- ✅ Presentation template with structure and placeholders
- ✅ Google Slides shared with team

---

## Week 3: Final Figures, Presentation Polish & Quality Assurance

**Goal:** All figures finalized, presentation polished, everything tested.

### Final Visualizations
- [ ] Generate final heatmap visualizations for:
  - S3DIS Area 5 (best room, 3–4 different queries)
  - Polycam scan (same queries for comparison)
  - Novel/open-vocabulary queries (2–3 creative ones)
- [ ] Create the **architecture diagram** for the presentation:
  - Point cloud → Concerto → MLP → CLIP space → query → heatmap
  - Clean, professional diagram (use draw.io, Figma, or PowerPoint shapes)
- [ ] Generate a **t-SNE / UMAP visualization** (stretch goal):
  - Plot MLP-projected features in 2D, colored by class
  - Shows how well the MLP separates classes in CLIP space
  - If this doesn't look good, skip it

### Presentation Polish
- [ ] Finalize all slides with real content (coordinate with each team member)
- [ ] Ensure consistent formatting: font sizes, colors, figure placement
- [ ] Add speaker notes for each slide
- [ ] Review the flow: does the story make sense? Is it compelling?
- [ ] Time the presentation: should be ~13–14 minutes (leave buffer)
- [ ] Prepare potential Q&A questions and answers:
  - "Why not finetune the backbone?" → Colab constraints, frozen encoder is the whole point
  - "Why MLP instead of linear probe?" → Professor feedback, Concerto paper shows gap
  - "How does this compare to OpenScene?" → Different approach (they use 2D→3D distillation)
  - "What about outdoor scenes?" → Concerto is indoor-trained, mention Utonia as future work
  - "What's the compute cost?" → Feature extraction: X min, MLP training: Y min

### Quality Assurance
- [ ] **Full pipeline test**: run the entire pipeline from scratch on Colab:
  1. Clone repo
  2. Install deps
  3. Load preprocessed data
  4. Load pre-extracted features
  5. Load trained MLP
  6. Run a query → visualize heatmap
  - If anything fails, file a bug and coordinate fix with the responsible person
- [ ] Verify all notebooks run end-to-end without errors
- [ ] Check that `requirements.txt` is complete and pinned
- [ ] **Merge everything to `main`** via `dev` → `main` PR (with team approval)

### Presentation Day Preparation
- [ ] Prepare your slides (you present the overview):
  - Problem statement (Slide 2)
  - Related work (Slide 3)
  - Limitations & future work (Slide 14)
- [ ] Practice the full presentation as a team (dry run):
  - Each person presents their section
  - Time it
  - Give each other feedback
- [ ] Prepare the demo environment:
  - Open Colab notebook before the presentation
  - Pre-run the slow cells (model loading, feature loading)
  - Have the backup video ready
- [ ] Final presentation delivery

### Week 3 Deliverables
- ✅ All figures finalized and in slides
- ✅ Presentation polished and rehearsed
- ✅ Pipeline QA'd end-to-end
- ✅ `main` branch clean and tagged `v1.0`

---

## Key Responsibilities Summary

| What | When | Depends On |
|------|------|------------|
| Paper reading & notes | Days 1–2 | — |
| `src/visualize.py` | Days 3–5 | — |
| Evaluation visualizations | Days 6–8 | Ricardo's eval results |
| Presentation template | Days 9–10 | — |
| Final heatmap figures | Days 11–12 | Trained MLP, Polycam scans |
| Architecture diagram | Day 12 | — |
| Presentation polish | Days 12–13 | All team members' slide content |
| QA full pipeline | Day 13 | All components merged |
| Presentation rehearsal | Day 14 | All slides done |

---

## Dependencies on Others

- **Ricardo:** Evaluation metrics and results
- **Leonardo:** Trained MLP checkpoint (for heatmap generation)
- **Adrian:** Polycam scans (for visualization)
- **All:** Slide content for their respective sections

## What Others Depend on From You

- **Ricardo + Adrian:** Visualization utilities
- **All:** Presentation template and structure
- **All:** Final polished slides
- **All:** QA confirmation that the pipeline works

---

## Tips & Notes

- **Start the visualization utilities early** — everyone will need them for their evaluation and demo work.
- **Use Plotly for Colab compatibility** — matplotlib 3D plots are static and hard to navigate. Plotly gives interactive 3D views directly in Colab.
- **Keep figures consistent** — use the same color palette across all visualizations. Define a color map for the 13 S3DIS classes once and reuse it everywhere.
- **The presentation is your key deliverable** — you're the "quality gate" for the project. If something doesn't work, you're the first to catch it during QA.
- **Prepare for Q&A proactively** — compile a shared doc with potential questions and answers. Everyone should study it before the presentation.
