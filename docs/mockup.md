# Mockups and Incomplete Parts

| File with Mockup | What is Missing | Required to Complete |
|---|---|---|
| `scripts/extract_features.py` | Concerto Small encoder forward pass (mocked with random noise) | `src/encoder.py` (Leonardo) and Concerto weights |
| `src/evaluate.py` | Translation head (MLP) forward pass (mocked with random projection to CLIP space) | `src/translation_head.py` (Leonardo) and trained MLP checkpoint |
| `notebooks/05_demo.ipynb` | Concerto Small encoder feature extraction (mocked with random features) | `src/encoder.py` (Leonardo) |
| `notebooks/05_demo.ipynb` | MLP translation head inference (mocked with random CLIP-space projection) | `src/translation_head.py` (Leonardo) and trained MLP checkpoint |
