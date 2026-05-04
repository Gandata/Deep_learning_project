import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.train import FeatureDataset, compute_loss

def test_feature_dataset():
    num_samples = 100
    feat_dim = 256
    target_dim = 512
    num_classes = 13
    
    features = torch.randn(num_samples, feat_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    target_table = torch.randn(num_classes, target_dim)
    
    dataset = FeatureDataset(
        features=features,
        labels=labels,
        target_table=target_table,
        normalize_features=True
    )
    
    assert len(dataset) == num_samples
    
    feat, target, label = dataset[0]
    
    assert feat.shape == (feat_dim,)
    assert target.shape == (target_dim,)
    assert isinstance(label, torch.Tensor)
    
    # Check normalization
    assert torch.allclose(torch.norm(feat, p=2), torch.tensor(1.0), atol=1e-6)
    
    # Check target mapping
    expected_target = target_table[labels[0]]
    assert torch.equal(target, expected_target)

def test_compute_loss_mse():
    pred = torch.randn(10, 512)
    target = torch.randn(10, 512)
    
    loss = compute_loss(pred, target, "mse")
    expected = F.mse_loss(pred, target)
    assert torch.allclose(loss, expected)

def test_compute_loss_cosine():
    pred = F.normalize(torch.randn(10, 512), dim=-1)
    target = F.normalize(torch.randn(10, 512), dim=-1)
    
    loss = compute_loss(pred, target, "cosine")
    # cosine similarity is in [-1, 1], loss is (1 - sim).mean()
    sim = F.cosine_similarity(pred, target, dim=-1)
    expected = (1.0 - sim).mean()
    assert torch.allclose(loss, expected)

def test_compute_loss_combined():
    pred = torch.randn(10, 512)
    target = torch.randn(10, 512)
    
    mse_w = 0.5
    cos_w = 2.0
    loss = compute_loss(pred, target, "combined", mse_weight=mse_w, cosine_weight=cos_w)
    
    mse_loss = F.mse_loss(pred, target)
    cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    expected = mse_w * mse_loss + cos_w * cosine_loss
    assert torch.allclose(loss, expected)
