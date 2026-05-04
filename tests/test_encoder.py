import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock, patch

# Mock concerto before importing src.encoder
mock_concerto = MagicMock()
sys.modules["concerto"] = mock_concerto

from src.encoder import ConcertoEncoder

def test_encoder_initialization_defaults():
    with patch("src.encoder.hf_hub_download") as mock_hf:
        mock_hf.return_value = "dummy_checkpoint.pth"
        mock_concerto.model.load.return_value = MagicMock()
        
        encoder = ConcertoEncoder(
            feature_dim=256,
        checkpoint_path=None,  # We won't trigger HF download if we just mock the load
            device="cpu",
        )
        assert encoder.feature_dim == 256
        assert encoder.backend == "concerto"
        
        # Verify HF was called if no local checkpoint
        mock_hf.assert_called_once()
        mock_concerto.model.load.assert_called_once()
    
    print("test_encoder_initialization_defaults passed!")

if __name__ == "__main__":
    test_encoder_initialization_defaults()
