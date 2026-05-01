from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a 3D point cloud of a {}",
    "a {}",
)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_embedding_dim(model: torch.nn.Module) -> int:
    projection = getattr(model, "text_projection", None)
    if projection is None:
        return int(model.ln_final.weight.shape[0])
    if projection.ndim == 2:
        return int(projection.shape[1])
    return int(projection.shape[0])


class CLIPTextEncoder:
    """
    Thin wrapper around OpenCLIP text encoding utilities.

    Designed for:
    - class label embeddings
    - free-text query embeddings
    - template-averaged embeddings for training targets
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim = _infer_embedding_dim(self.model)

        self._single_text_cache: dict[tuple[str, bool], Tensor] = {}

    @torch.no_grad()
    def encode_text(
        self,
        text: str,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> Tensor:
        cache_key = (text, normalize)
        if use_cache and cache_key in self._single_text_cache:
            return self._single_text_cache[cache_key].clone()

        embedding = self.encode_texts([text], normalize=normalize)[0]
        if use_cache:
            self._single_text_cache[cache_key] = embedding.detach().clone()
        return embedding

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Tensor:
        if len(texts) == 0:
            raise ValueError("`texts` must contain at least one string.")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")

        outputs: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(batch).to(self.device)
            text_features = self.model.encode_text(tokens)
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
            outputs.append(text_features)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_with_templates(
        self,
        label: str,
        templates: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> Tensor:
        templates = tuple(templates or DEFAULT_PROMPT_TEMPLATES)
        if len(templates) == 0:
            raise ValueError("`templates` must contain at least one prompt template.")

        prompts = [template.format(label) for template in templates]
        prompt_embeddings = self.encode_texts(prompts, normalize=True)
        label_embedding = prompt_embeddings.mean(dim=0)

        if normalize:
            label_embedding = F.normalize(label_embedding.unsqueeze(0), dim=-1).squeeze(0)

        return label_embedding

    @torch.no_grad()
    def encode_labels(
        self,
        labels: Sequence[str],
        templates: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> Tensor:
        if len(labels) == 0:
            raise ValueError("`labels` must contain at least one class name.")

        embeddings = [
            self.encode_with_templates(label, templates=templates, normalize=normalize)
            for label in labels
        ]
        return torch.stack(embeddings, dim=0)


def get_text_embedding(
    text: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | torch.device | None = None,
    normalize: bool = True,
) -> Tensor:
    encoder = CLIPTextEncoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    return encoder.encode_text(text, normalize=normalize)


def get_class_embeddings(
    class_names: Sequence[str],
    templates: Sequence[str] | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | torch.device | None = None,
    normalize: bool = True,
) -> Tensor:
    encoder = CLIPTextEncoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    return encoder.encode_labels(
        class_names,
        templates=templates,
        normalize=normalize,
    )


def save_class_embeddings_numpy(
    output_path: str | Path,
    class_names: Sequence[str],
    templates: Sequence[str] | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | torch.device | None = None,
) -> np.ndarray:
    embeddings = get_class_embeddings(
        class_names,
        templates=templates,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        normalize=True,
    )
    array = embeddings.detach().cpu().numpy().astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    return array


if __name__ == "__main__":
    labels = ["chair", "table", "whiteboard"]
    encoder = CLIPTextEncoder()
    embeddings = encoder.encode_labels(labels)
    print(f"Labels: {labels}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Embedding dim: {encoder.embedding_dim}")
