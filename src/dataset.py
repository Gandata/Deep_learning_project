"""
dataset.py
----------
PyTorch dataset for processed S3DIS point clouds stored as one `.npy` file per room.

Expected room format:
    Area_5/conferenceRoom_1.npy

Each room file must have columns:
    X Y Z R G B label_id
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


LABEL_MAP = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "sofa",
    8: "table",
    9: "chair",
    10: "bookcase",
    11: "board",
    12: "clutter",
}
NUM_CLASSES = len(LABEL_MAP)

LABEL_TEXT = {
    0: "ceiling of a room",
    1: "floor of a room",
    2: "wall of a room",
    3: "beam on the ceiling",
    4: "column or pillar",
    5: "window",
    6: "door",
    7: "sofa or couch",
    8: "table or desk",
    9: "chair",
    10: "bookcase or shelf",
    11: "whiteboard or board",
    12: "clutter or miscellaneous object",
}


class S3DISDataset(Dataset):
    """
    PyTorch dataset for processed S3DIS point clouds.

    Args:
        root: path to processed S3DIS root containing Area_1..Area_6
        areas: which areas to include
        normalize_xyz: subtract room centroid from XYZ
        max_points: optional random subsampling cap
    """

    def __init__(
        self,
        root: str,
        areas: list = [1, 2, 3, 4, 5, 6],
        normalize_xyz: bool = True,
        max_points: int | None = None,
    ):
        self.root = Path(root)
        self.areas = areas
        self.normalize_xyz = normalize_xyz
        self.max_points = max_points
        self.label_map = LABEL_MAP
        self.label_text = LABEL_TEXT
        self.num_classes = NUM_CLASSES

        self.rooms = self._collect_rooms()
        if len(self.rooms) == 0:
            raise RuntimeError(
                f"No rooms found in {self.root} for areas {self.areas}. "
                "Check that the path is correct and the data is downloaded."
            )
        print(f"S3DISDataset: {len(self.rooms)} rooms loaded from areas {self.areas}")

    def _collect_rooms(self) -> list[Path]:
        rooms: list[Path] = []
        for area_id in self.areas:
            area_path = self.root / f"Area_{area_id}"
            if not area_path.exists():
                print(f"  Warning: {area_path} not found, skipping.")
                continue
            for room_file in sorted(area_path.glob("*.npy")):
                rooms.append(room_file)
        return rooms

    def __len__(self):
        return len(self.rooms)

    @staticmethod
    def _normalize_color(color: np.ndarray) -> np.ndarray:
        if color.size > 0 and color.max() > 1.0:
            color = color / 255.0
        return np.clip(color, 0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        room_path = self.rooms[idx]
        room_array = np.load(room_path).astype(np.float32)

        if room_array.ndim != 2 or room_array.shape[1] < 7:
            raise ValueError(
                f"Expected room file {room_path} to have shape (N, >=7), got {room_array.shape}."
            )

        coord = room_array[:, :3].astype(np.float32)
        color = self._normalize_color(room_array[:, 3:6].astype(np.float32))
        label = room_array[:, 6].astype(np.int64)

        if self.normalize_xyz:
            coord = coord - coord.mean(axis=0)

        if self.max_points is not None and len(coord) > self.max_points:
            idx_sample = np.random.choice(len(coord), self.max_points, replace=False)
            coord = coord[idx_sample]
            color = color[idx_sample]
            label = label[idx_sample]

        return {
            "coord": coord,
            "color": color,
            "label": label,
            "room": f"{room_path.parent.name}/{room_path.stem}",
        }

    def get_label_name(self, label_id: int) -> str:
        return self.label_map.get(int(label_id), "unknown")

    def get_label_text(self, label_id: int) -> str:
        return self.label_text.get(int(label_id), "unknown object")

    def get_all_label_texts(self) -> list[str]:
        return [self.label_text[i] for i in range(self.num_classes)]


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/s3dis_processed"

    print("Loading val dataset (Area 5)...")
    val_ds = S3DISDataset(root=root, areas=[5], max_points=50000)

    sample = val_ds[0]
    print(f"\nSample room : {sample['room']}")
    print(f"coord shape : {sample['coord'].shape}")
    print(f"color shape : {sample['color'].shape}")
    print(f"label shape : {sample['label'].shape}")
    print(f"XYZ range   : {sample['coord'].min():.3f} to {sample['coord'].max():.3f}")
    print(f"RGB range   : {sample['color'].min():.3f} to {sample['color'].max():.3f}")
    print(f"Labels      : {np.unique(sample['label'])}")
