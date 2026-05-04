import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_metrics

def test_compute_metrics():
    # Perfect prediction
    true_labels = np.array([0, 1, 2, 0, 1, 2])
    pred_labels = np.array([0, 1, 2, 0, 1, 2])
    
    oa, miou, ious = compute_metrics(pred_labels, true_labels, num_classes=3)
    
    assert oa == 1.0
    assert miou == 1.0
    assert len(ious) == 3
    assert all(iou == 1.0 for iou in ious)

def test_compute_metrics_partial():
    # Class 0: 2 TP, 1 FP (pred 0 instead of 1) -> IOU = 2 / (2+1) = 0.66
    # Class 1: 1 TP, 1 FN (true 1 is pred 0) -> IOU = 1 / (1+1) = 0.5
    # Class 2: 2 TP -> IOU = 1.0
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    pred_labels = np.array([0, 0, 0, 1, 2, 2])
    
    oa, miou, ious = compute_metrics(pred_labels, true_labels, num_classes=3)
    
    assert oa == 5/6
    
    # Class 0: TP=2, FP=1, FN=0 -> IOU = 2/3
    # Class 1: TP=1, FP=0, FN=1 -> IOU = 1/2
    # Class 2: TP=2, FP=0, FN=0 -> IOU = 1/1
    
    assert np.allclose(ious[0], 2/3)
    assert np.allclose(ious[1], 1/2)
    assert np.allclose(ious[2], 1.0)
    
    expected_miou = (2/3 + 0.5 + 1.0) / 3
    assert np.allclose(miou, expected_miou)

def test_compute_metrics_nan():
    # Class with no samples in true or pred should be ignored in miou
    true_labels = np.array([0, 0])
    pred_labels = np.array([0, 0])
    
    oa, miou, ious = compute_metrics(pred_labels, true_labels, num_classes=2)
    
    assert ious[0] == 1.0
    assert np.isnan(ious[1])
    assert miou == 1.0 # nanmean should ignore it
