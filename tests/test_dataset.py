import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import S3DISDataset

def test_s3dis_dataset_loading(tmp_path):
    # Create a dummy structure
    # Area_1/room_1.npy
    area_dir = tmp_path / "Area_1"
    area_dir.mkdir()
    
    # Create a dummy .npy file with 100 points and 7 columns (X, Y, Z, R, G, B, label)
    num_points = 100
    dummy_data = np.random.rand(num_points, 7).astype(np.float32)
    # Ensure labels are integers (last column)
    dummy_data[:, 6] = np.random.randint(0, 13, size=num_points)
    
    room_file = area_dir / "room_1.npy"
    np.save(room_file, dummy_data)
    
    # Initialize dataset
    dataset = S3DISDataset(root=str(tmp_path), areas=[1], normalize_xyz=True)
    
    assert len(dataset) == 1
    
    # Get a sample
    sample = dataset[0]
    
    assert "coord" in sample
    assert "color" in sample
    assert "label" in sample
    assert "room" in sample
    
    assert sample["coord"].shape == (num_points, 3)
    assert sample["color"].shape == (num_points, 3)
    assert sample["label"].shape == (num_points,)
    
    # Check color normalization (should be in [0, 1])
    assert sample["color"].max() <= 1.0
    assert sample["color"].min() >= 0.0

def test_s3dis_dataset_subsampling(tmp_path):
    area_dir = tmp_path / "Area_1"
    area_dir.mkdir()
    
    num_points = 1000
    max_points = 100
    dummy_data = np.random.rand(num_points, 7).astype(np.float32)
    room_file = area_dir / "room_1.npy"
    np.save(room_file, dummy_data)
    
    dataset = S3DISDataset(root=str(tmp_path), areas=[1], max_points=max_points)
    sample = dataset[0]
    
    assert sample["coord"].shape == (max_points, 3)
    assert sample["color"].shape == (max_points, 3)
    assert sample["label"].shape == (max_points,)
