import torch
from src.encoder import ConcertoEncoder

def test_encoder_initialization_defaults():
    # Only tests initialization to ensure no syntax errors and default device assignment works
    encoder = ConcertoEncoder(
        feature_dim=256,
        checkpoint_path=None,  # We won't trigger HF download if we just mock the load
        device="cpu",
    )
    assert encoder.feature_dim == 256
    assert encoder.backend == "concerto"
