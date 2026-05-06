"""
export_polycam.py
─────────────────
Converts a raw .ply point cloud (e.g. from Polycam or any 3D scanner)
into the same format used by S3DISDataset:
    coord.npy    (N, 3)  float32  XYZ, centroid-normalised
    color.npy    (N, 3)  float32  RGB [0, 1]  (gray if no color in source)
    normal.npy   (N, 3)  float32  surface normals (estimated if missing)

Output is saved as a folder of .npy files, ready to be loaded by
the demo notebook (05_demo.ipynb) and fed into Concerto.

Usage (Colab or terminal):
    python scripts/export_polycam.py \
        --input  data/polycam/scenario1_room1_indoor.ply \
        --output data/polycam_processed/room1

Author: Adrian (Polycam pipeline)
"""

import os
import argparse
import logging
import numpy as np
import open3d as o3d
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a raw .ply scan into pipeline-ready .npy files."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input .ply file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output folder where coord/color/normal.npy will be saved",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling in metres (default: 0.02 = 2cm). "
             "Set to 0 to skip downsampling.",
    )
    parser.add_argument(
        "--normal_radius",
        type=float,
        default=0.1,
        help="Search radius for normal estimation in metres (default: 0.1)",
    )
    parser.add_argument(
        "--normal_max_nn",
        type=int,
        default=30,
        help="Max neighbours for normal estimation (default: 30)",
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        default=True,
        help="Remove statistical outliers (default: True)",
    )
    parser.add_argument(
        "--s3dis_input",
        action="store_true",
        help="If set, treats --input as a pre-processed S3DIS directory containing coord.npy and color.npy",
    )
    return parser.parse_args()


# ── Processing steps ──────────────────────────────────────────────────────────

def load_ply(path: Path) -> o3d.geometry.PointCloud:
    """Load a .ply file and report what fields it contains."""
    log.info(f"Loading: {path}")
    pcd = o3d.io.read_point_cloud(str(path))

    n = len(pcd.points)
    if n == 0:
        raise RuntimeError(f"No points loaded from {path}. Check the file.")

    log.info(f"  Loaded {n:,} points")
    log.info(f"  Has color:   {pcd.has_colors()}")
    log.info(f"  Has normals: {pcd.has_normals()}")
    return pcd


def remove_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Remove statistical outliers to clean up noise."""
    log.info("Removing statistical outliers...")
    before = len(pcd.points)
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    after = len(pcd_clean.points)
    log.info(f"  {before:,} → {after:,} points ({before - after:,} removed)")
    return pcd_clean


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Voxel downsample to reduce density while keeping structure."""
    log.info(f"Downsampling with voxel size {voxel_size}m...")
    before = len(pcd.points)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    after = len(pcd_down.points)
    log.info(f"  {before:,} → {after:,} points")
    return pcd_down


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
) -> o3d.geometry.PointCloud:
    """Estimate surface normals from local geometry."""
    log.info(f"Estimating normals (radius={radius}m, max_nn={max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Orient normals consistently (towards camera / positive Z)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 10.0]))
    log.info("  Normals estimated and oriented.")
    return pcd


def assign_default_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Assign neutral gray color when the scan has no RGB data."""
    log.info("No color found in file — assigning default gray (0.5, 0.5, 0.5)")
    n = len(pcd.points)
    gray = np.full((n, 3), 0.5, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(gray)
    return pcd


def normalize_xyz(coords: np.ndarray) -> np.ndarray:
    """Subtract centroid so the scene is centred at the origin."""
    centroid = coords.mean(axis=0)
    log.info(f"Normalising XYZ — centroid: {centroid.round(3)}")
    return coords - centroid


def save_npy(output_dir: Path, coord, color, normal):
    """Save the three arrays as .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "coord.npy",  coord.astype(np.float32))
    np.save(output_dir / "color.npy",  color.astype(np.float32))
    np.save(output_dir / "normal.npy", normal.astype(np.float32))

    log.info(f"Saved to {output_dir}/")
    log.info(f"  coord.npy  : {coord.shape}  float32")
    log.info(f"  color.npy  : {color.shape}  float32")
    log.info(f"  normal.npy : {normal.shape} float32")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if args.s3dis_input:
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path must be a directory in s3dis mode: {input_path}")
        coord_path = input_path / "coord.npy"
        color_path = input_path / "color.npy"
        if not coord_path.exists() or not color_path.exists():
            raise FileNotFoundError(f"coord.npy and color.npy must exist in {input_path}")
        
        log.info(f"Loading S3DIS processed room from {input_path}")
        coord = np.load(coord_path).astype(np.float64)
        color = np.load(color_path).astype(np.float64)
        if color.max() > 1.0:
            color /= 255.0
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(color)
        log.info(f"  Loaded {len(coord):,} points")
    else:
        # 1. Load
        pcd = load_ply(input_path)

    # 2. Remove outliers
    if args.remove_outliers:
        pcd = remove_outliers(pcd)

    # 3. Downsample
    if args.voxel_size > 0:
        pcd = downsample(pcd, args.voxel_size)

    # 4. Handle color
    if not pcd.has_colors():
        pcd = assign_default_color(pcd)

    # 5. Estimate normals if missing
    if not pcd.has_normals():
        pcd = estimate_normals(pcd, args.normal_radius, args.normal_max_nn)

    # 6. Extract numpy arrays
    coord  = np.asarray(pcd.points,  dtype=np.float32)   # (N, 3)
    color  = np.asarray(pcd.colors,  dtype=np.float32)   # (N, 3) in [0,1]
    normal = np.asarray(pcd.normals, dtype=np.float32)   # (N, 3)

    # 7. Normalise XYZ
    coord = normalize_xyz(coord)

    # 8. Sanity checks
    log.info("\nFinal stats:")
    log.info(f"  Points : {len(coord):,}")
    log.info(f"  XYZ    : {coord.min():.3f} to {coord.max():.3f}")
    log.info(f"  RGB    : {color.min():.3f} to {color.max():.3f}")
    log.info(f"  Normals: {normal.min():.3f} to {normal.max():.3f}")

    # 9. Save
    save_npy(output_path, coord, color, normal)

    log.info("\nDone! You can now load this scan in the demo notebook.")


if __name__ == "__main__":
    main()
