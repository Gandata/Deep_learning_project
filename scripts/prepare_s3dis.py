"""
prepare_s3dis.py
────────────────
Preprocesses the raw S3DIS dataset (Stanford3dDataset_v1.2_Aligned_Version)
into clean .npy files ready for the Concerto feature extraction pipeline.

Usage (Colab or terminal):
    python scripts/prepare_s3dis.py \
        --input  data/s3dis_raw/Stanford3dDataset_v1.2_Aligned_Version \
        --output data/s3dis_processed

Output per room:
    data/s3dis_processed/
    ├── Area_1/
    │   ├── conferenceRoom_1.npy   # float32 array shape (N, 7): X Y Z R G B label_id
    │   └── ...
    ├── Area_2/
    │   └── ...
    └── ...

    data/s3dis_processed/
    └── label_map.json             # {"ceiling": 0, "floor": 1, ..., "clutter": 12}

Author: Adrian (Data preparation)
"""

import os
import re
import json
import argparse
import logging
import numpy as np
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── S3DIS label vocabulary (12 categories + clutter) ─────────────────────────
LABEL_MAP = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 3,
    "column": 4,
    "window": 5,
    "door": 6,
    "sofa": 7,
    "table": 8,
    "chair": 9,
    "bookcase": 10,
    "board": 11,
    "clutter": 12,
}
NUM_CLASSES = len(LABEL_MAP)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw S3DIS data.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to Stanford3dDataset_v1.2_Aligned_Version folder",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Where to write processed .npy files",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["1", "2", "3", "4", "5", "6"],
        help="Which areas to process (default: all 6)",
    )
    parser.add_argument(
        "--normalize_xyz",
        action="store_true",
        help="Shift each room so its centroid is at the origin (dataset.py does this on the fly, so leave this off)",
    )
    parser.add_argument(
        "--normalize_rgb",
        action="store_true",
        help="Scale RGB values from [0,255] to [0,1] (dataset.py does this on the fly, so leave this off)",
    )
    return parser.parse_args()


# ── Core helpers ──────────────────────────────────────────────────────────────


def label_from_annotation_folder(folder_name: str) -> int:
    """
    S3DIS stores each object as a subfolder named e.g. 'chair_1', 'wall_3'.
    Extract the category name and return the integer label id.
    Unknown categories are mapped to 'clutter'.
    """
    # strip trailing _<number>
    category = re.sub(r"_\d+$", "", folder_name).lower().strip()
    return LABEL_MAP.get(category, LABEL_MAP["clutter"])


def load_room(room_path: Path, normalize_xyz: bool, normalize_rgb: bool) -> np.ndarray:
    """
    Parse one room directory and return a float32 array of shape (N, 7):
        columns: X  Y  Z  R  G  B  label_id

    S3DIS room structure:
        room_path/
        ├── room_name.txt          ← full merged point cloud (X Y Z R G B)
        └── Annotations/
            ├── chair_1.txt        ← per-object files (X Y Z R G B)
            ├── chair_2.txt
            └── ...

    We rebuild the labelled cloud from the Annotations folder so each point
    carries the correct semantic label.
    """
    annotations_dir = room_path / "Annotations"
    if not annotations_dir.exists():
        log.warning(f"  No Annotations folder in {room_path}, skipping.")
        return None

    point_chunks = []

    for obj_file in sorted(annotations_dir.glob("*.txt")):
        # Derive label from filename (e.g. "chair_1.txt" → "chair" → 9)
        label_id = label_from_annotation_folder(obj_file.stem)

        try:
            # Each line: X Y Z R G B  (space-separated, sometimes has stray chars)
            raw = np.loadtxt(obj_file, dtype=np.float32)
        except Exception as e:
            log.warning(f"    Could not parse {obj_file.name}: {e} — skipping")
            continue

        if raw.ndim == 1:
            raw = raw[np.newaxis, :]  # single-point edge case

        if raw.shape[1] < 6:
            log.warning(
                f"    {obj_file.name} has only {raw.shape[1]} columns, skipping"
            )
            continue

        xyz = raw[:, :3]
        rgb = raw[:, 3:6]
        labels = np.full((len(raw), 1), label_id, dtype=np.float32)

        point_chunks.append(np.hstack([xyz, rgb, labels]))

    if not point_chunks:
        log.warning(f"  Room {room_path.name} produced no points.")
        return None

    cloud = np.vstack(point_chunks)  # (N, 7)

    # ── Normalise XYZ: subtract room centroid ──────────────────────────────
    if normalize_xyz:
        cloud[:, :3] -= cloud[:, :3].mean(axis=0)

    # ── Normalise RGB: [0,255] → [0,1] ────────────────────────────────────
    if normalize_rgb:
        # Guard: some versions already have values in [0,1]
        if cloud[:, 3:6].max() > 1.0:
            cloud[:, 3:6] /= 255.0
        cloud[:, 3:6] = np.clip(cloud[:, 3:6], 0.0, 1.0)

    return cloud


def process_area(
    area_path: Path, output_area: Path, normalize_xyz: bool, normalize_rgb: bool
):
    """Process all rooms in one area folder."""
    output_area.mkdir(parents=True, exist_ok=True)
    room_dirs = sorted([d for d in area_path.iterdir() if d.is_dir()])

    if not room_dirs:
        log.warning(f"No room directories found in {area_path}")
        return

    stats = {"rooms": 0, "points": 0, "skipped": 0}

    for room_dir in room_dirs:
        out_room_dir = output_area / room_dir.name

        if out_room_dir.exists() and (out_room_dir / "coord.npy").exists():
            log.info(f"  [skip] {room_dir.name} already processed")
            stats["rooms"] += 1
            continue

        log.info(f"  Processing room: {room_dir.name}")
        cloud = load_room(room_dir, normalize_xyz, normalize_rgb)

        if cloud is None:
            stats["skipped"] += 1
            continue
            
        out_room_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into coord, color, segment
        coord = cloud[:, :3].astype(np.float32)
        color = cloud[:, 3:6].astype(np.float32)
        segment = cloud[:, 6:7].astype(np.int64)
        
        np.save(out_room_dir / "coord.npy", coord)
        np.save(out_room_dir / "color.npy", color)
        np.save(out_room_dir / "segment.npy", segment)

        stats["rooms"] += 1
        stats["points"] += len(cloud)
        log.info(f"    → saved {len(cloud):,} points  ({out_room_dir.name}/)")

    log.info(
        f"  Area done: {stats['rooms']} rooms, "
        f"{stats['points']:,} total points, "
        f"{stats['skipped']} skipped"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Save label map once
    label_map_path = output_root / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    log.info(f"Label map saved → {label_map_path}")

    # Process each requested area
    for area_id in args.areas:
        area_name = f"Area_{area_id}"
        area_path = input_root / area_name

        if not area_path.exists():
            log.warning(f"Area not found, skipping: {area_path}")
            continue

        log.info(f"\n{'─'*50}")
        log.info(f"Processing {area_name} ...")
        output_area = output_root / area_name
        process_area(area_path, output_area, args.normalize_xyz, args.normalize_rgb)

    log.info(f"\n{'─'*50}")
    log.info(f"All done. Processed data saved to: {output_root}")
    log.info("Each .npy file is float32 with columns: X Y Z R G B label_id")


if __name__ == "__main__":
    main()
