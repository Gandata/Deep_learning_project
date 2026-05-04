import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.clip_utils import get_text_embedding

def test_get_text_embedding():
    # Mock model
    embedding_dim = 512
    mock_model = MagicMock()
    # model.encode_text should return a tensor of shape (batch_size, embedding_dim)
    # get_text_embedding calls tokenizer([text]), so batch_size=1
    mock_model.encode_text.return_value = torch.randn(1, embedding_dim)
    
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = torch.zeros((1, 77), dtype=torch.long) # dummy tokens
    
    text = "a photo of a chair"
    device = "cpu"
    
    embedding = get_text_embedding(mock_model, mock_tokenizer, text, device=device)
    
    # Check type
    assert isinstance(embedding, torch.Tensor)
    
    # Check shape
    # Based on the user request, they expect [1, 512]. 
    # However, currently the function returns text_features[0] which is (512,)
    # I will assert (512,) and if it fails or if I should follow user's [1, 512], I might need to adjust.
    # Actually, the user's recommendation is "ensure ... correctly returns a torch.Tensor of shape [1, 512]".
    # Let's see if I should change the implementation or if the user is just describing what they WANT it to be.
    # Usually [1, 512] is better for consistency if it's used in batches.
    
    assert embedding.shape == (1, embedding_dim)
    
    # Check L2 normalization
    norm = torch.norm(embedding, p=2, dim=-1)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
    
    # Verify mock calls
    mock_tokenizer.assert_called_once_with([text])
    mock_model.encode_text.assert_called_once()
