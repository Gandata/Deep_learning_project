from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.clip_utils import DEFAULT_PROMPT_TEMPLATES, CLIPTextEncoder
from src.translation_head import MLPTranslationHead

try:
    from src.dataset import LABEL_TEXT as DEFAULT_LABEL_TEXT
except Exception:
    DEFAULT_LABEL_TEXT = {
        0: "ceiling of a room",
        1: "floor of a room",
        2: "wall of a room",
        3: "beam on the ceiling",
        4: "column or pillar",
        5: "window",
        6: "door",
        7: "sofa or couch",
        8: "table or desk",
        9: "chair",
        10: "bookcase or shelf",
        11: "whiteboard or board",
        12: "clutter or miscellaneous object",
    }


FEATURE_KEYS = ("features", "feature", "feat")
LABEL_KEYS = ("labels", "label", "segment")


class FeatureDataset(Dataset):
    """Point-wise dataset built from pre-extracted 3D features and label ids."""

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
        target_table: Tensor,
        normalize_features: bool = False,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"`features` must have shape (N, D), got {tuple(features.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"`labels` must have shape (N,), got {tuple(labels.shape)}.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "Features and labels must have the same number of rows, "
                f"got {features.shape[0]} and {labels.shape[0]}."
            )

        max_label = int(labels.max().item()) if labels.numel() > 0 else -1
        if max_label >= target_table.shape[0]:
            raise ValueError(
                f"Target embedding table has {target_table.shape[0]} rows, "
                f"but labels contain id {max_label}."
            )

        self.features = features.float()
        if normalize_features:
            self.features = F.normalize(self.features, dim=-1)

        self.labels = labels.long()
        self.target_table = target_table.float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        feature = self.features[index]
        label = self.labels[index]
        target = self.target_table[label]
        return feature, target, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLP translation head.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_key(candidates: tuple[str, ...], names: list[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(
        f"Could not find any of {candidates} in {path}. Available keys: {names}"
    )


def load_feature_file(path: Path) -> tuple[Tensor, Tensor]:
    if path.suffix != ".npz":
        raise ValueError(f"Feature file must be .npz, got: {path}")

    with np.load(path) as data:
        names = list(data.keys())
        feature_key = choose_key(FEATURE_KEYS, names, path)
        label_key = choose_key(LABEL_KEYS, names, path)

        features = data[feature_key].astype(np.float32)
        labels = data[label_key].astype(np.int64)

    if labels.ndim > 1:
        labels = labels.reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape (N, D) in {path}, got {features.shape}.")
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"Feature/label row mismatch in {path}: {features.shape[0]} vs {labels.shape[0]}."
        )

    return torch.from_numpy(features), torch.from_numpy(labels)


def collect_feature_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Feature path not found: {path}")

    files = sorted(path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz feature files found under: {path}")
    return files


def load_feature_split(path: str | Path) -> tuple[Tensor, Tensor]:
    files = collect_feature_files(path)
    feature_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    for file_path in files:
        features, labels = load_feature_file(file_path)
        feature_chunks.append(features)
        label_chunks.append(labels)

    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def resolve_label_texts(config: dict[str, Any]) -> list[str]:
    data_cfg = config.get("data", {})
    label_texts = data_cfg.get("label_texts")
    if label_texts is None:
        return [DEFAULT_LABEL_TEXT[index] for index in sorted(DEFAULT_LABEL_TEXT)]

    if isinstance(label_texts, dict):
        return [label_texts[index] for index in sorted(label_texts)]

    if isinstance(label_texts, list):
        return label_texts

    raise TypeError("`data.label_texts` must be a list or dict if provided.")


def build_target_table(config: dict[str, Any], device: torch.device) -> Tensor:
    data_cfg = config.get("data", {})
    clip_cfg = config.get("clip", {})

    label_embedding_path = data_cfg.get("label_embeddings_path")
    if label_embedding_path:
        array = np.load(label_embedding_path).astype(np.float32)
        target_table = torch.from_numpy(array)
        return F.normalize(target_table, dim=-1).to(device)

    label_texts = resolve_label_texts(config)
    templates = clip_cfg.get("templates", list(DEFAULT_PROMPT_TEMPLATES))

    encoder = CLIPTextEncoder(
        model_name=clip_cfg.get("model_name", "ViT-B-32"),
        pretrained=clip_cfg.get("pretrained", "openai"),
        device=device,
    )
    target_table = encoder.encode_labels(label_texts, templates=templates, normalize=True)
    return target_table.to(device)


def build_model(config: dict[str, Any], target_dim: int) -> MLPTranslationHead:
    model_cfg = config.get("model", {})
    return MLPTranslationHead(
        input_dim=model_cfg.get("input_dim", 256),
        hidden_dims=model_cfg.get("hidden_dims", [512, 512]),
        output_dim=model_cfg.get("output_dim", target_dim),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "gelu"),
        normalize_output=model_cfg.get("normalize_output", True),
    )


def compute_loss(
    predictions: Tensor,
    targets: Tensor,
    loss_name: str,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> Tensor:
    loss_name = loss_name.lower()

    if loss_name == "mse":
        return F.mse_loss(predictions, targets)

    if loss_name == "cosine":
        return (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()

    if loss_name in {"mse+cosine", "combined", "hybrid"}:
        mse_loss = F.mse_loss(predictions, targets)
        cosine_loss = (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        return mse_weight * mse_loss + cosine_weight * cosine_loss

    raise ValueError(f"Unsupported loss type: {loss_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    optimizer: AdamW | None = None,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for features, targets, _labels in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        predictions = model(features)
        loss = compute_loss(
            predictions,
            targets,
            loss_name=loss_name,
            mse_weight=mse_weight,
            cosine_weight=cosine_weight,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    config: dict[str, Any],
    train_loss: float,
    val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = resolve_device(training_cfg.get("device"))
    print(f"Using device: {device}")

    target_table = build_target_table(config, device=device)
    model = build_model(config, target_dim=target_table.shape[1]).to(device)

    train_features, train_labels = load_feature_split(data_cfg["train_features_path"])
    val_path = data_cfg.get("val_features_path")
    val_split = load_feature_split(val_path) if val_path else None

    normalize_features = data_cfg.get("normalize_features", False)
    train_dataset = FeatureDataset(
        train_features,
        train_labels,
        target_table=target_table.detach().cpu(),
        normalize_features=normalize_features,
    )
    val_dataset = None
    if val_split is not None:
        val_features, val_labels = val_split
        val_dataset = FeatureDataset(
            val_features,
            val_labels,
            target_table=target_table.detach().cpu(),
            normalize_features=normalize_features,
        )

    batch_size = training_cfg.get("batch_size", 4096)
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )

    resume_from = training_cfg.get("resume_from")
    start_epoch = 1
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resumed training from epoch {start_epoch}.")

    epochs = training_cfg.get("epochs", 50)
    loss_name = training_cfg.get("loss", "mse")
    mse_weight = training_cfg.get("mse_weight", 1.0)
    cosine_weight = training_cfg.get("cosine_weight", 1.0)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    save_every = training_cfg.get("save_every", 5)
    save_best = training_cfg.get("save_best", True)
    metrics_path = training_cfg.get("metrics_path")

    best_val_loss = float("inf")
    history: list[dict[str, float | int | None]] = []

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples:   {len(val_dataset)}")
    print(f"Model params:  {num_parameters:,}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            loss_name=loss_name,
            optimizer=optimizer,
            mse_weight=mse_weight,
            cosine_weight=cosine_weight,
        )

        val_loss = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    device=device,
                    loss_name=loss_name,
                    optimizer=None,
                    mse_weight=mse_weight,
                    cosine_weight=cosine_weight,
                )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
            }
        )

        if val_loss is None:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if epoch % save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        improved = val_loss is not None and val_loss < best_val_loss
        if save_best and improved:
            best_val_loss = float(val_loss)
            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                train_loss=train_loss,
                val_loss=val_loss,
            )

    final_path = checkpoint_dir / "last_model.pth"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        config=config,
        train_loss=history[-1]["train_loss"],
        val_loss=history[-1]["val_loss"],
    )
    print(f"Saved final checkpoint to: {final_path}")

    if metrics_path:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(f"Saved metrics to: {metrics_file}")


if __name__ == "__main__":
    main()
