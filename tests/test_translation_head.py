import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.translation_head import MLPTranslationHead


def test_translation_head_single_cloud_shape():
    model = MLPTranslationHead(
        input_dim=256,
        hidden_dims=(512, 512),
        output_dim=512,
        normalize_output=True,
    )

    features = torch.randn(1024, 256)
    output = model(features)

    assert output.shape == (1024, 512)


def test_translation_head_batched_shape():
    model = MLPTranslationHead(
        input_dim=256,
        hidden_dims=(512, 512),
        output_dim=512,
        normalize_output=True,
    )

    features = torch.randn(2, 1024, 256)
    output = model(features)

    assert output.shape == (2, 1024, 512)


def test_translation_head_output_is_normalized():
    model = MLPTranslationHead(
        input_dim=256,
        hidden_dims=(512, 512),
        output_dim=512,
        normalize_output=True,
    )

    features = torch.randn(128, 256)
    output = model(features)
    norms = output.norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_translation_head_raises_on_wrong_input_dim():
    model = MLPTranslationHead(
        input_dim=256,
        hidden_dims=(512, 512),
        output_dim=512,
        normalize_output=True,
    )

    wrong_features = torch.randn(128, 128)

    try:
        model(wrong_features)
        assert False, "Expected ValueError for wrong input dimension."
    except ValueError as error:
        assert "Last dimension must be 256" in str(error)
