"""
dataset.py
──────────
PyTorch Dataset for the Pointcept-formatted S3DIS data.

Each room folder contains:
    coord.npy    (N, 3)  float32  XYZ positions
    color.npy    (N, 3)  uint8    RGB [0, 255]
    normal.npy   (N, 3)  float32  surface normals [-1, 1]
    segment.npy  (N, 1)  int16    semantic label id [0, 12]
    instance.npy (N, 1)  int16    instance id (not used here)

The dataset returns per-room point clouds as dicts:
    {
        "coord":    float32 (N, 3)   XYZ, centroid-normalised
        "color":    float32 (N, 3)   RGB scaled to [0, 1]
        "normal":   float32 (N, 3)   surface normals
        "label":    int64   (N,)     semantic label ids
        "room":     str              e.g. "Area_5/conferenceRoom_1"
    }

Usage:
    from src.dataset import S3DISDataset

    dataset = S3DISDataset(
        root="data/s3dis_raw",
        areas=[1, 2, 3, 4, 6],   # training areas
    )
    val_dataset = S3DISDataset(
        root="data/s3dis_raw",
        areas=[5],                # eval area
    )

    sample = dataset[0]
    print(sample["coord"].shape)   # (N, 3)
    print(sample["label"].shape)   # (N,)

Author: Adrian (Data preparation)
"""

import os
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

# ── Label vocabulary (matches Pointcept's S3DIS ordering) ────────────────────
LABEL_MAP = {
    0:  "ceiling",
    1:  "floor",
    2:  "wall",
    3:  "beam",
    4:  "column",
    5:  "window",
    6:  "door",
    7:  "sofa",
    8:  "table",
    9:  "chair",
    10: "bookcase",
    11: "board",
    12: "clutter",
}
NUM_CLASSES = len(LABEL_MAP)

# CLIP-friendly text descriptions for each label (used by clip_utils.py)
LABEL_TEXT = {
    0:  "ceiling of a room",
    1:  "floor of a room",
    2:  "wall of a room",
    3:  "beam on the ceiling",
    4:  "column or pillar",
    5:  "window",
    6:  "door",
    7:  "sofa or couch",
    8:  "table or desk",
    9:  "chair",
    10: "bookcase or shelf",
    11: "whiteboard or board",
    12: "clutter or miscellaneous object",
}


class S3DISDataset(Dataset):
    """
    PyTorch Dataset for Pointcept-formatted S3DIS point clouds.

    Args:
        root        (str)        Path to s3dis_raw folder containing Area_1..Area_6
        areas       (list[int])  Which areas to include, e.g. [1,2,3,4,6] for train
        normalize_xyz (bool)     Subtract room centroid from XYZ (default: True)
        use_normal  (bool)       Include surface normals in output (default: True)
        max_points  (int|None)   If set, randomly subsample rooms to this many points
    """

    def __init__(
        self,
        root: str,
        areas: list = [1, 2, 3, 4, 5, 6],
        normalize_xyz: bool = True,
        use_normal: bool = True,
        max_points: int = None,
    ):
        self.root          = Path(root)
        self.areas         = areas
        self.normalize_xyz = normalize_xyz
        self.use_normal    = use_normal
        self.max_points    = max_points
        self.label_map     = LABEL_MAP
        self.label_text    = LABEL_TEXT
        self.num_classes   = NUM_CLASSES

        # Collect all room paths
        self.rooms = self._collect_rooms()
        if len(self.rooms) == 0:
            raise RuntimeError(
                f"No rooms found in {self.root} for areas {self.areas}. "
                f"Check that the path is correct and the data is downloaded."
            )
        print(f"S3DISDataset: {len(self.rooms)} rooms loaded from areas {self.areas}")

    def _collect_rooms(self):
        """Walk the area folders and return sorted list of room paths."""
        rooms = []
        for area_id in self.areas:
            area_path = self.root / f"Area_{area_id}"
            if not area_path.exists():
                print(f"  Warning: {area_path} not found, skipping.")
                continue
            for room_dir in sorted(area_path.iterdir()):
                if room_dir.is_dir() and (room_dir / "coord.npy").exists():
                    rooms.append(room_dir)
        return rooms

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx: int) -> dict:
        room_path = self.rooms[idx]

        # ── Load arrays ───────────────────────────────────────────────────
        coord  = np.load(room_path / "coord.npy").astype(np.float32)   # (N, 3)
        color  = np.load(room_path / "color.npy").astype(np.float32)   # (N, 3) uint8→float
        label  = np.load(room_path / "segment.npy").astype(np.int64)   # (N, 1)

        # Flatten label to (N,)
        label = label.squeeze(-1)

        # ── Normalise RGB: [0, 255] → [0, 1] ─────────────────────────────
        color = color / 255.0
        color = np.clip(color, 0.0, 1.0)

        # ── Normalise XYZ: subtract centroid ──────────────────────────────
        if self.normalize_xyz:
            coord = coord - coord.mean(axis=0)

        # ── Optional: subsample points ────────────────────────────────────
        if self.max_points is not None and len(coord) > self.max_points:
            idx_sample = np.random.choice(len(coord), self.max_points, replace=False)
            coord = coord[idx_sample]
            color = color[idx_sample]
            label = label[idx_sample]

        # ── Build output dict ─────────────────────────────────────────────
        sample = {
            "coord": coord,                                          # (N, 3)
            "color": color,                                          # (N, 3)
            "label": label,                                          # (N,)
            "room":  f"{room_path.parent.name}/{room_path.name}",   # e.g. "Area_5/conferenceRoom_1"
        }

        # Optionally include surface normals
        if self.use_normal:
            normal = np.load(room_path / "normal.npy").astype(np.float32)  # (N, 3)
            if self.max_points is not None and "idx_sample" in dir():
                normal = normal[idx_sample]
            sample["normal"] = normal

        return sample

    def get_label_name(self, label_id: int) -> str:
        """Return the category name for a label id, e.g. 9 → 'chair'."""
        return self.label_map.get(int(label_id), "unknown")

    def get_label_text(self, label_id: int) -> str:
        """Return the CLIP-friendly text description for a label id."""
        return self.label_text.get(int(label_id), "unknown object")

    def get_all_label_texts(self) -> list:
        """Return list of all CLIP text descriptions ordered by label id."""
        return [self.label_text[i] for i in range(self.num_classes)]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/s3dis_raw"

    print("Loading val dataset (Area 5)...")
    val_ds = S3DISDataset(root=root, areas=[5], max_points=50000)

    sample = val_ds[0]
    print(f"\nSample room : {sample['room']}")
    print(f"coord shape : {sample['coord'].shape}")
    print(f"color shape : {sample['color'].shape}")
    print(f"label shape : {sample['label'].shape}")
    print(f"normal shape: {sample['normal'].shape}")
    print(f"XYZ range   : {sample['coord'].min():.3f} to {sample['coord'].max():.3f}")
    print(f"RGB range   : {sample['color'].min():.3f} to {sample['color'].max():.3f}")
    print(f"Labels      : {np.unique(sample['label'])}")
    print(f"\nLabel names : {[val_ds.get_label_name(l) for l in np.unique(sample['label'])]}")
    print(f"\nAll CLIP texts:")
    for i, text in enumerate(val_ds.get_all_label_texts()):
        print(f"  {i:2d}: {text}")
