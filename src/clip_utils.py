from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch import Tensor


DEFAULT_PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a 3D point cloud of a {}",
    "a {}",
)


load_dotenv()


def init_hf() -> None:
    """Authenticate to Hugging Face if an HF token is available."""
    token = os.getenv("HF_TOKEN")
    if token and token != "your_huggingface_token_here":
        login(token=token)
    else:
        print(
            "Warning: No valid HF_TOKEN found in .env. "
            "Downloads may fail if authentication is required."
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


@torch.no_grad()
def get_text_embedding(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Backward-compatible helper used by existing notebooks/scripts.
    Returns a single normalized text embedding.
    """
    device = torch.device(device)
    text_tokens = tokenizer([text]).to(device)

    autocast_context = (
        torch.amp.autocast(device_type="cuda")
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    with autocast_context:
        text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def get_class_embeddings(
    model: torch.nn.Module,
    tokenizer,
    class_names: Sequence[str],
    templates: Sequence[str],
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """
    Backward-compatible helper used by existing notebooks/scripts.
    Returns a dict mapping class description -> numpy embedding.
    """
    class_embeddings: dict[str, np.ndarray] = {}
    for class_name in class_names:
        embeddings = []
        for template in templates:
            text = template.format(class_name)
            embedding = get_text_embedding(model, tokenizer, text, device=device)
            embeddings.append(embedding)
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding = F.normalize(avg_embedding.unsqueeze(0), dim=-1).squeeze(0)
        class_embeddings[class_name] = avg_embedding.cpu().numpy()
    return class_embeddings


class CLIPTextEncoder:
    """
    Higher-level OpenCLIP wrapper for training/inference code.

    This keeps the older functional helpers intact while exposing a cleaner API
    for Leonardo's training pipeline.
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


def save_class_embeddings_numpy(
    output_path: str | Path,
    class_names: Sequence[str],
    templates: Sequence[str] | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | torch.device | None = None,
) -> np.ndarray:
    encoder = CLIPTextEncoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    embeddings = encoder.encode_labels(class_names, templates=templates, normalize=True)
    array = embeddings.detach().cpu().numpy().astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    return array


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.dataset import LABEL_TEXT, NUM_CLASSES

    init_hf()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    templates = [
        "a photo of a {}",
        "a 3D point cloud of a {}",
        "a {}",
        "this is a {}",
        "an indoor scene with a {}",
    ]

    print("Computing CLIP embeddings for S3DIS classes...")
    class_descriptions = [LABEL_TEXT[i] for i in range(NUM_CLASSES)]
    class_embeddings_dict = get_class_embeddings(
        model,
        tokenizer,
        class_descriptions,
        templates,
        device=device,
    )

    embeddings_array = np.zeros((NUM_CLASSES, 512), dtype=np.float32)
    for i, description in enumerate(class_descriptions):
        embeddings_array[i] = class_embeddings_dict[description]

    out_dir = Path("data/s3dis_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "label_to_clip_embeddings.npy"
    np.save(out_file, embeddings_array)
    print(f"Saved CLIP embeddings to {out_file} (shape: {embeddings_array.shape})")
