# Feasibility Evaluation

> Honest assessment of risks, constraints, and mitigations for the project.

---

## 1. Overall Feasibility Verdict

| Aspect | Verdict | Confidence |
|--------|---------|------------|
| Core pipeline (Concerto → MLP → CLIP query) | ✅ **Feasible** | High |
| S3DIS Area 5 evaluation | ✅ **Feasible** | High |
| In-the-wild Polycam demo | ⚠️ **Feasible with risks** | Medium |
| Utonia comparison | ⚠️ **Stretch goal only** | Low–Medium |
| 15-min presentation + demo | ✅ **Feasible** | High |

**Bottom line:** The core project (items 1–2 + presentation) is well within reach. The in-the-wild demo has moderate risk (Polycam export quality, domain gap). Utonia should remain a strict bonus.

---

## 2. Compute Constraints (Colab Free Tier T4)

### What We Have

| Resource | Limit |
|----------|-------|
| GPU | NVIDIA T4 (16GB VRAM) |
| RAM | ~12.7GB system RAM |
| Disk | ~78GB local, unlimited Drive |
| Session length | ~90 min (can be shorter under load) |
| Daily GPU quota | ~4–6 hours (varies, not guaranteed) |
| CUDA | 11.8 (Colab default) |

### Will It Fit?

| Operation | VRAM Estimate | Fits T4? | Notes |
|-----------|---------------|----------|-------|
| Concerto Small forward pass (39M params) | ~4–6 GB | ✅ Yes | Frozen, inference only. May need chunking for very large scenes. |
| CLIP ViT-B/32 text encoder | ~0.5 GB | ✅ Yes | Very lightweight for text-only |
| MLP translation head (3 layers, 256→512) | < 0.1 GB | ✅ Yes | Tiny |
| Training the MLP (batch of point features) | ~2–4 GB total | ✅ Yes | Features pre-extracted, so no backbone in memory during training |
| Full pipeline inference (encoder + MLP + CLIP) | ~6–8 GB | ✅ Yes | Sequential, not all in memory at once |

**Key insight:** Because the backbone is **frozen**, we can **pre-extract features once** and save them to Drive. Training the MLP then only requires loading pre-extracted features (numpy arrays), which is extremely lightweight. This is the critical design decision that makes Colab feasible.

### Session Drop Risk

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Session drops during feature extraction | Medium | Medium | Save features per-room/per-scene incrementally to Drive. Resume from last saved room. |
| Session drops during MLP training | Medium | Low | Save checkpoint every 5 epochs to Drive. Training is fast (<30 min total), so loss is minimal. |
| Session drops during evaluation | Low | Low | Evaluation is fast (minutes). Re-run if dropped. |
| GPU quota exhausted for the day | Medium | Medium | Schedule heavy GPU work (feature extraction) across multiple days. MLP training is CPU-feasible as backup. |

### Mitigation: Pre-extraction Strategy

```
Day 1: Extract Concerto features for S3DIS Area 5, Room 1–5   → save to Drive
Day 2: Extract Concerto features for S3DIS Area 5, Room 6–10  → save to Drive  
Day 3: Extract Concerto features for Polycam scans             → save to Drive
Day 4+: All MLP training uses pre-extracted features (no GPU backbone needed)
```

This makes us **resilient to session drops** — we never lose more than one room's worth of computation.

---

## 3. Data Risks

### S3DIS (Primary Evaluation)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Download requires Stanford form (manual approval) | High | Medium | Submit form immediately (Day 1). Usually approved within 24h. If delayed, use alternative download mirrors or pre-processed versions from Pointcept. |
| 14GB is large for Drive free tier (15GB) | High | **High** | Process and discard raw data after preprocessing. Keep only processed Area 5 (~2–3 GB). Or use a team member's Drive with more space. |
| Preprocessing fails or format mismatch with Pointcept | Low | Medium | Pointcept repo has S3DIS preprocessing scripts — use those directly. Test on a small subset first. |

### Drive Storage Budget

| Item | Size Estimate |
|------|---------------|
| S3DIS raw (temporary) | ~14 GB |
| S3DIS processed (Area 5 only) | ~2–3 GB |
| Concerto features (Area 5) | ~1–2 GB |
| Polycam scans | ~0.1–0.5 GB |
| Checkpoints | < 0.1 GB |
| **Total (after cleanup)** | **~4–6 GB** |

**Recommendation:** Use one team member's Drive account that has sufficient space (15 GB free, or a university Google Workspace with more). Delete raw S3DIS data after preprocessing.

### Polycam In-the-Wild Scans

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Polycam export format incompatible with pipeline | Medium | High | Test export early (Week 1). Polycam exports `.ply` with XYZ+RGB, which Open3D reads natively. Write a conversion script. |
| Point density / quality too different from S3DIS | Medium | Medium | Expected — this is partly the point of the demo (show domain gap). Subsample to match S3DIS density if needed. |
| Polycam requires paid subscription for full export | Low | Low | Free tier supports `.ply` export. Verify early. Alternatives: 3D Scanner App (iOS), Meshroom (desktop photogrammetry). |
| No access to a suitable indoor space to scan | Low | Low | Any room works — a lab, apartment, classroom. The scan doesn't need to be high quality; it just needs to be a real scan. |

---

## 4. Software / Dependency Risks

### spconv Installation on Colab

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `spconv-cu118` install fails or version conflict | Medium | **High** | spconv is the #1 installation headache in point cloud research. Pin the exact version in `requirements.txt`. Test install in a clean Colab session on Day 1. If it fails, try `spconv-cu121` or build from source. |
| CUDA version mismatch (Colab updates CUDA) | Low–Medium | High | Colab sometimes updates CUDA silently. Pin `spconv-cu118` and check `nvcc --version` at session start. |

### Pointcept / Concerto Compatibility

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Concerto weights not loadable (format change) | Low | High | Download weights early, test loading on Day 1. The HuggingFace Space has a working demo, so weights should be fine. |
| Pointcept data pipeline too complex to adapt | Medium | Medium | We don't need the full Pointcept training pipeline — only the encoder forward pass. Write a minimal wrapper that loads weights and runs inference. |
| Concerto requires custom C++ ops not available on Colab | Low–Medium | High | Concerto uses spconv (handled above) and standard PyTorch ops. Check if any custom CUDA kernels are needed. |

### CLIP

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CLIP installation or version issues | Very Low | Low | `open_clip_torch` is well-maintained and Colab-friendly. No expected issues. |

---

## 5. Timeline Risks

### Week-by-Week Risk Assessment

| Week | Goal | Primary Risk | Mitigation |
|------|------|-------------|------------|
| **Week 1** | Data pipeline + feature extraction working | spconv install, S3DIS download delay | Start both immediately on Day 1. Have backup plan (pre-processed data). |
| **Week 2** | MLP trained + evaluation on S3DIS | Training instability, poor mIoU | MLP training is fast (minutes–hours). Iterate on architecture/loss quickly. If mIoU is very low, try deeper MLP or different loss. |
| **Week 3** | Polycam demo + presentation | Domain gap too large, demo not compelling | Accept that some degradation is expected. Frame it as a finding, not a failure. Have S3DIS demo as the primary showcase. |

### "What If Everything Goes Wrong" Minimum Viable Project

Even in the worst case, the team can deliver:

1. ✅ Concerto feature extraction on S3DIS (pre-extracted, saved)
2. ✅ MLP trained on S3DIS labels → CLIP space (even with modest mIoU)
3. ✅ Quantitative evaluation table (mIoU, top-k accuracy)
4. ✅ Qualitative heatmap visualizations on S3DIS
5. ✅ Presentation explaining the approach, results, and limitations

This is a **complete, presentable project** even without the Polycam demo or Utonia comparison.

---

## 6. MLP Translation Head — Will It Work?

This is the central technical risk.

### Evidence For

- Concerto's own paper (Table 5) shows that **linear probing on language** reaches 44.56% mIoU. An MLP should do better.
- The professor explicitly approved the MLP approach over linear probing.
- Similar CLIP-3D alignment work (e.g., OpenScene, LERF) shows that learned projections between 3D features and CLIP space are effective.

### Evidence Against

- 44.56% linear probe mIoU suggests the gap between Concerto's feature space and CLIP's is significant.
- An MLP may overfit on S3DIS's 13 classes and fail to generalize to open-vocabulary queries.
- The "open-vocabulary" claim is only as good as the training supervision — we train on 13 class labels, so truly novel queries (e.g., "fire extinguisher") may not work.

### Realistic Expectations

| Metric | Optimistic | Realistic | Pessimistic |
|--------|-----------|-----------|-------------|
| S3DIS mIoU (13 classes) | 55–60% | 45–55% | 35–45% |
| Open-vocab queries (qualitative) | Works for most common objects | Works for S3DIS classes, weak for novel | Only works for trained classes |
| Polycam demo (qualitative) | Clear heatmaps | Noisy but visible | Fails due to domain gap |

### What to Do If mIoU Is Low

1. **Try deeper MLP** (4 layers instead of 3)
2. **Try different loss** (cosine embedding loss instead of MSE)
3. **Try feature normalization** (L2-normalize before MLP)
4. **Try ensembling** multiple CLIP text templates ("a photo of a {}", "a 3D scan of a {}", etc.)
5. **Lower the bar** — report the number honestly and discuss why in the presentation. The professor will respect honest analysis over inflated results.

---

## 7. Utonia — Feasibility as Optional Extension

| Factor | Assessment |
|--------|------------|
| Public weights available? | **Unknown** — as of the project start, Utonia (March 2026) may not have released weights yet. Check the GitHub repo and HuggingFace space. |
| Colab-feasible? | **Likely yes** — if architecture is similar to Concerto (transformer-based point cloud encoder), inference should fit T4. |
| Time to integrate? | **1–3 days** — if the API is similar to Concerto, the wrapper can be adapted. If the architecture is very different, it could take longer. |
| Value for the project? | **High** — showing that a cross-domain encoder improves Polycam results would be a strong finding. |

### Decision Rule

- **Week 2 checkpoint:** If Concerto results on S3DIS are good (mIoU > 45%) AND Utonia weights are public → Ricardo spends 2 days on Utonia comparison.
- **Otherwise:** Skip Utonia, focus on polishing the demo and presentation.

---

## 8. Presentation & Demo Feasibility

| Requirement | Feasibility | Notes |
|-------------|-------------|-------|
| 15-min presentation | ✅ Easy | 4 people, each presents ~3–4 minutes |
| Working demo | ✅ Feasible | Run notebook live on Colab, or pre-record a video as backup |
| 15-min Q&A | ✅ Manageable | Everyone should understand the full pipeline, not just their part |

### Demo Strategy

- **Primary:** Live Colab notebook (`05_demo.ipynb`) showing a text query on a 3D scene
- **Backup:** Pre-recorded video/GIF of the demo (in case Colab is slow or crashes during presentation)
- **Double backup:** Static screenshots with heatmap overlays

---

## 9. Risk Summary Matrix

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|-----------|--------|------------|-------|
| spconv install fails | Medium | High | Test Day 1, pin version, have fallback | Leonardo |
| S3DIS download delayed | Medium | Medium | Submit form Day 1, use Pointcept preprocessed | Adrian |
| Drive storage insufficient | High | High | Use university account, delete raw data after preprocessing | Adrian |
| Colab session drops during extraction | Medium | Medium | Save incrementally per-room | Ricardo |
| MLP mIoU too low | Medium | Medium | Iterate on architecture/loss, lower expectations honestly | Leonardo |
| Polycam export incompatible | Medium | Medium | Test export Week 1, have conversion script | Adrian |
| Polycam domain gap too large | Medium | Low | Frame as a finding, not a failure | Ricardo |
| Utonia weights not available | High | Low | It's optional — skip if unavailable | Ricardo |
| Notebook merge conflicts | Medium | Low | One owner per notebook, nbstripout | All |
| Team member unavailable (illness, other projects) | Low–Medium | Medium | Clear task ownership, overlap on critical path | All |

---

## 10. Go/No-Go Checkpoints

| Date | Checkpoint | Go Condition | No-Go Action |
|------|-----------|-------------|--------------|
| **End of Day 2** | Environment setup | spconv installs, Concerto loads, S3DIS form submitted | Escalate spconv issue; consider alternative encoder |
| **End of Week 1** | Feature extraction | Concerto features extracted for ≥3 rooms | Re-scope to fewer rooms; simplify pipeline |
| **Mid Week 2** | MLP training | MLP trains without errors, loss decreases | Debug training loop; simplify to linear probe as fallback |
| **End of Week 2** | Evaluation | mIoU computed on Area 5, ≥35% | Analyze failure modes; adjust architecture |
| **Mid Week 3** | Demo | End-to-end query works on at least one scene | Use S3DIS only; skip Polycam |
| **End of Week 3** | Presentation | Slides done, demo recorded as backup | Present with static results |
