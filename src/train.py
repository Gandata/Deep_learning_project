from __future__ import annotations

import argparse
import gc
import json
import math
import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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


class FileFeatureBatcher:
    """Stream batches from feature files without concatenating the full split in RAM."""

    def __init__(
        self,
        files: list[Path],
        target_table: Tensor,
        batch_size: int,
        normalize_features: bool = False,
        shuffle_files: bool = False,
        shuffle_points: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        self.files = files
        self.target_table = target_table.float()
        self.batch_size = int(batch_size)
        self.normalize_features = normalize_features
        self.shuffle_files = shuffle_files
        self.shuffle_points = shuffle_points

    def __iter__(self):
        file_indices = list(range(len(self.files)))
        if self.shuffle_files:
            random.shuffle(file_indices)

        for file_index in file_indices:
            features, labels = load_feature_file(self.files[file_index])
            features = features.float()
            if self.normalize_features:
                features = F.normalize(features, dim=-1)
            labels = labels.long()

            if self.shuffle_points:
                order = torch.randperm(features.shape[0])
            else:
                order = torch.arange(features.shape[0])

            for start in range(0, features.shape[0], self.batch_size):
                batch_index = order[start : start + self.batch_size]
                batch_features = features[batch_index]
                batch_labels = labels[batch_index]
                batch_targets = self.target_table[batch_labels]
                yield batch_features, batch_targets, batch_labels

            del features, labels, order
            gc.collect()


class GPUBufferedBatcher:
    """Load multiple rooms directly onto GPU, batch from GPU-resident data.

    This maximises GPU memory usage and eliminates per-batch CPU→GPU transfer.
    Rooms are loaded one-by-one; before each upload we check free GPU memory.
    Batches are yielded via random-index slicing (no full-tensor shuffle copy).
    """

    def __init__(
        self,
        files: list[Path],
        target_table: Tensor,
        batch_size: int,
        device: torch.device,
        normalize_features: bool = False,
        shuffle_files: bool = False,
        shuffle_points: bool = False,
        gpu_budget_gb: float = 10.0,
        store_dtype: torch.dtype = torch.float16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        self.files = files
        self.target_table = target_table.to(device=device, dtype=store_dtype)
        self.batch_size = int(batch_size)
        self.device = device
        self.normalize_features = normalize_features
        self.shuffle_files = shuffle_files
        self.shuffle_points = shuffle_points
        self.gpu_budget_bytes = int(gpu_budget_gb * 1024**3)
        self.store_dtype = store_dtype

    def _gpu_free(self) -> int:
        """Return genuinely free GPU memory (bytes)."""
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info(self.device)[0]  # (free, total)

    def __iter__(self):
        file_indices = list(range(len(self.files)))
        if self.shuffle_files:
            random.shuffle(file_indices)

        idx = 0
        cpu_features = None
        cpu_labels = None

        while idx < len(file_indices):
            # 1. Determine safe allocation size based on genuinely free memory
            torch.cuda.empty_cache()
            free_mem = self._gpu_free()
            # Keep 2 GB free for training (activations, etc.)
            safe_alloc = min(self.gpu_budget_bytes, free_mem - 2 * 1024**3)
            
            if safe_alloc <= 0:
                raise torch.OutOfMemoryError(
                    f"GPU completely full; cannot allocate buffer. Free memory: {free_mem/1024**3:.2f} GB."
                )

            all_features = None
            all_labels = None
            max_points = 0
            current_points = 0

            # 2. Fill the pre-allocated buffer
            while idx < len(file_indices):
                fi = file_indices[idx]
                
                if cpu_features is None:
                    cpu_features, cpu_labels = load_feature_file(self.files[fi])
                    
                n = cpu_features.shape[0]
                feat_dim = cpu_features.shape[1]
                
                # Pre-allocate GPU tensors on the first file of the chunk
                if all_features is None:
                    bytes_per_point = feat_dim * (2 if self.store_dtype == torch.float16 else 4) + 8
                    max_points = safe_alloc // bytes_per_point
                    
                    if n > max_points:
                        # Edge case: A single room is larger than safe_alloc. 
                        # We must allocate exactly n points to avoid an infinite loop.
                        max_points = n  
                        
                    all_features = torch.empty((max_points, feat_dim), device=self.device, dtype=self.store_dtype)
                    all_labels = torch.empty((max_points,), device=self.device, dtype=torch.int64)

                # Check if this room fits in the remaining pre-allocated space
                if current_points + n > max_points:
                    if current_points == 0:
                        pass # Handled by the edge case above
                    else:
                        break # Buffer full, process what we have

                # CPU processing
                if self.normalize_features:
                    cpu_features = F.normalize(cpu_features.float(), dim=-1)
                
                # Cast to float16 on CPU before transfer
                cpu_features = cpu_features.to(self.store_dtype)
                
                # Direct Host-to-Device copy into the pre-allocated GPU slice (ZERO double allocation)
                all_features[current_points:current_points+n].copy_(cpu_features, non_blocking=True)
                all_labels[current_points:current_points+n].copy_(cpu_labels, non_blocking=True)
                
                current_points += n
                idx += 1
                cpu_features = None
                cpu_labels = None

            if current_points == 0:
                break
                
            # 3. Truncate buffer to actually used size (creates a view, no memory overhead)
            buffer_features = all_features[:current_points]
            buffer_labels = all_labels[:current_points]

            # 4. Yield random batches via index slicing
            if self.shuffle_points:
                order = torch.randperm(current_points, device=self.device)
            else:
                order = torch.arange(current_points, device=self.device)

            for start in range(0, current_points, self.batch_size):
                end = min(start + self.batch_size, current_points)
                batch_idx = order[start:end]
                batch_features = buffer_features[batch_idx]
                batch_labels = buffer_labels[batch_idx]
                batch_targets = self.target_table[batch_labels]
                yield batch_features, batch_targets, batch_labels

            del all_features, all_labels, buffer_features, buffer_labels, order
            torch.cuda.empty_cache()


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


def infer_room_type(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[0] == "Area":
        room_type_parts = parts[2:-1]
        if room_type_parts:
            return "_".join(room_type_parts)
    return path.stem


def select_files_by_room_type_fraction(
    files: list[Path],
    fraction: float,
    seed: int,
    split_name: str,
) -> list[Path]:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"`{split_name}_room_type_fraction` must be in (0, 1], got {fraction}."
        )
    if fraction >= 1.0:
        return files

    grouped: dict[str, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(infer_room_type(file_path), []).append(file_path)

    rng = random.Random(seed)
    selected: list[Path] = []
    print(
        f"Selecting {fraction:.0%} of {split_name} files per room type "
        f"(rounded up, seed={seed})"
    )
    for room_type in sorted(grouped):
        room_files = sorted(grouped[room_type])
        shuffled = room_files[:]
        rng.shuffle(shuffled)
        keep_count = math.ceil(len(room_files) * fraction)
        chosen = sorted(shuffled[:keep_count])
        selected.extend(chosen)
        print(f"  {room_type}: keeping {len(chosen)}/{len(room_files)} files")

    selected = sorted(selected)
    print(f"Selected {len(selected)}/{len(files)} {split_name} files total")
    return selected


def holdout_split_by_room_type(
    files: list[Path],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split files into train/val sets, stratified by room type.

    Ensures every room type with ≥2 files has at least 1 file in the
    validation set, giving a representative held-out evaluation.
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(
            f"`val_holdout_fraction` must be in (0, 1), got {holdout_fraction}."
        )

    grouped: dict[str, list[Path]] = {}
    for file_path in files:
        grouped.setdefault(infer_room_type(file_path), []).append(file_path)

    rng = random.Random(seed)
    train_files: list[Path] = []
    val_files: list[Path] = []

    print(
        f"Holdout validation split: {holdout_fraction:.0%} of files "
        f"per room type (seed={seed})"
    )
    for room_type in sorted(grouped):
        room_files = sorted(grouped[room_type])
        shuffled = room_files[:]
        rng.shuffle(shuffled)
        val_count = max(1, math.ceil(len(room_files) * holdout_fraction))
        # If there's only 1 file of this type, keep it for training
        if len(room_files) == 1:
            train_files.extend(room_files)
            print(f"  {room_type}: 1 file → all train (no val)")
            continue
        val_chosen = sorted(shuffled[:val_count])
        train_chosen = sorted(shuffled[val_count:])
        val_files.extend(val_chosen)
        train_files.extend(train_chosen)
        print(
            f"  {room_type}: {len(train_chosen)} train / "
            f"{len(val_chosen)} val (of {len(room_files)})"
        )

    train_files = sorted(train_files)
    val_files = sorted(val_files)
    print(
        f"Holdout result: {len(train_files)} train files, "
        f"{len(val_files)} val files"
    )
    return train_files, val_files


def load_feature_split(path: str | Path) -> tuple[Tensor, Tensor]:
    files = collect_feature_files(path)
    feature_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    bad_files: list[str] = []

    for file_path in files:
        try:
            features, labels = load_feature_file(file_path)
        except Exception as error:
            bad_files.append(f"{file_path}: {error}")
            continue
        feature_chunks.append(features)
        label_chunks.append(labels)

    if bad_files:
        joined = "\n".join(bad_files[:10])
        if len(bad_files) > 10:
            joined += f"\n... and {len(bad_files) - 10} more"
        raise RuntimeError(
            "Some extracted feature files are invalid or incomplete. "
            "They likely need to be deleted and regenerated.\n"
            f"{joined}"
        )

    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def inspect_feature_files(
    path: str | Path,
    split_name: str,
    room_type_fraction: float = 1.0,
    room_type_fraction_seed: int = 42,
    verbose: bool = True,
    do_inspect: bool = True,
    expected_dim: int | None = None,
) -> tuple[list[Path], int | None, int, dict[Path, int]]:
    files = collect_feature_files(path)
    files = select_files_by_room_type_fraction(
        files,
        fraction=room_type_fraction,
        seed=room_type_fraction_seed,
        split_name=split_name,
    )

    if not do_inspect:
        if verbose:
            print(f"Skipping inspection of {split_name} files ({len(files)} found)")
        return files, expected_dim, 0, {}

    detected_input_dim: int | None = expected_dim
    total_samples = 0
    file_counts: dict[Path, int] = {}
    bad_files: list[str] = []

    if verbose:
        print(f"Inspecting {split_name} files: {len(files)} found")

    for index, file_path in enumerate(files, start=1):
        try:
            features, labels = load_feature_file(file_path)
        except Exception as error:
            bad_files.append(f"{file_path}: {error}")
            continue

        file_dim = int(features.shape[1])
        num_samples = int(features.shape[0])
        file_counts[file_path] = num_samples

        if detected_input_dim is None:
            detected_input_dim = file_dim
        elif file_dim != detected_input_dim:
            bad_files.append(
                f"{file_path}: feature dim {file_dim} does not match expected {detected_input_dim}"
            )
        total_samples += num_samples

        if verbose:
            print(
                f"  [{index}/{len(files)}] {file_path.name}: "
                f"{num_samples} samples, dim={file_dim}"
            )
        del features, labels
        gc.collect()

    if detected_input_dim is None and do_inspect:
        raise FileNotFoundError(f"No valid {split_name} feature files found under: {path}")

    return files, detected_input_dim, total_samples, file_counts


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


def build_model(
    config: dict[str, Any],
    target_dim: int,
    detected_input_dim: int | None = None,
) -> MLPTranslationHead:
    model_cfg = config.get("model", {})
    configured_input_dim = model_cfg.get("input_dim")
    if configured_input_dim is None:
        if detected_input_dim is None:
            configured_input_dim = 256
        else:
            configured_input_dim = detected_input_dim
    elif detected_input_dim is not None and int(configured_input_dim) != int(detected_input_dim):
        raise ValueError(
            "Config/model input_dim does not match extracted features: "
            f"config says {configured_input_dim}, but loaded features have dim {detected_input_dim}."
        )

    return MLPTranslationHead(
        input_dim=int(configured_input_dim),
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
    loader,
    device: torch.device,
    loss_name: str,
    optimizer: AdamW | None = None,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for features, targets, _labels in loader:
        # GPUBufferedBatcher yields GPU tensors; FileFeatureBatcher yields CPU tensors
        features = features.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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


def build_scheduler(
    optimizer: AdamW,
    training_cfg: dict[str, Any],
    start_epoch: int,
) -> CosineAnnealingLR | None:
    """Build an optional LR scheduler from the training config."""
    scheduler_name = training_cfg.get("scheduler", None)
    if scheduler_name is None or scheduler_name == "none":
        return None

    epochs = training_cfg.get("epochs", 50)
    if scheduler_name == "cosine":
        min_lr = training_cfg.get("scheduler_min_lr", 1e-6)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
        # Fast-forward scheduler if resuming
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(
            f"Using CosineAnnealingLR scheduler "
            f"(T_max={epochs}, eta_min={min_lr})"
        )
        return scheduler

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


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
    # --- Data Inspection & Split ---
    inspect_files = data_cfg.get("inspect_files", True)
    verbose_inspection = data_cfg.get("verbose_inspection", True)
    configured_input_dim = config.get("model", {}).get("input_dim")

    train_room_type_fraction = float(data_cfg.get("train_room_type_fraction", 1.0))
    train_room_type_fraction_seed = int(
        data_cfg.get("train_room_type_fraction_seed", training_cfg.get("seed", seed))
    )

    train_files, detected_input_dim, train_samples, train_counts = inspect_feature_files(
        data_cfg["train_features_path"],
        split_name="train",
        room_type_fraction=train_room_type_fraction,
        room_type_fraction_seed=train_room_type_fraction_seed,
        do_inspect=inspect_files,
        verbose=verbose_inspection,
        expected_dim=configured_input_dim,
    )

    # --- Validation: explicit path, holdout split, or none ---
    val_path = data_cfg.get("val_features_path")
    val_files = None
    val_samples = 0
    val_counts: dict[Path, int] = {}

    if val_path:
        # Explicit validation directory
        val_files, val_input_dim, val_samples, val_counts = inspect_feature_files(
            val_path,
            split_name="val",
            do_inspect=inspect_files,
            verbose=verbose_inspection,
            expected_dim=detected_input_dim,
        )
        if detected_input_dim is not None and val_input_dim is not None:
            if int(val_input_dim) != int(detected_input_dim):
                raise ValueError(
                    "Validation features do not match training feature dim: "
                    f"{val_input_dim} vs {detected_input_dim}."
                )
    else:
        # Random holdout from train files (stratified by room type)
        holdout_fraction = float(data_cfg.get("val_holdout_fraction", 0.0))
        if holdout_fraction > 0.0:
            holdout_seed = int(
                data_cfg.get("val_holdout_seed", training_cfg.get("seed", seed))
            )
            train_files, val_files = holdout_split_by_room_type(
                train_files,
                holdout_fraction=holdout_fraction,
                seed=holdout_seed,
            )
            # Re-calculate samples using counts from inspection (avoiding re-loading)
            train_samples = sum(train_counts.get(f, 0) for f in train_files)
            val_samples = sum(train_counts.get(f, 0) for f in val_files)

    model = build_model(
        config,
        target_dim=target_table.shape[1],
        detected_input_dim=detected_input_dim,
    ).to(device)

    # Optional torch.compile for PyTorch 2.x speedup
    use_compile = training_cfg.get("compile_model", False)
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    normalize_features = data_cfg.get("normalize_features", False)
    target_table_cpu = target_table.detach().cpu()

    # --- GPU buffer & AMP settings ---
    use_gpu_buffer = training_cfg.get("gpu_buffer", False) and device.type == "cuda"
    gpu_budget_gb = training_cfg.get("gpu_budget_gb", 12.0)
    use_amp = training_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_gpu_buffer:
        print(
            f"GPU-buffered training enabled: budget={gpu_budget_gb:.1f} GB, "
            f"AMP={'on' if use_amp else 'off'}"
        )

    batch_size = training_cfg.get("batch_size", 4096)

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

    scheduler = build_scheduler(optimizer, training_cfg, start_epoch)

    epochs = training_cfg.get("epochs", 50)
    loss_name = training_cfg.get("loss", "mse")
    mse_weight = training_cfg.get("mse_weight", 1.0)
    cosine_weight = training_cfg.get("cosine_weight", 1.0)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    save_every = training_cfg.get("save_every", 5)
    save_best = training_cfg.get("save_best", True)
    metrics_path = training_cfg.get("metrics_path")

    # --- Diagnostic: Verify checkpoint path (especially for Drive symlinks) ---
    print(f"Checkpoint directory: {checkpoint_dir.absolute()}")
    try:
        # Check if any parent is a symlink
        current = checkpoint_dir
        while current != current.parent:
            if current.is_symlink():
                target = os.readlink(current)
                print(f"  [Diagnostic] Component '{current}' is a symlink to: {target}")
                if not Path(target).exists() and not Path(current).resolve().exists():
                    print(f"  [WARNING] Symlink '{current}' appears to be BROKEN!")
            current = current.parent
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        test_file = checkpoint_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"  [Diagnostic] Directory is writable.")
    except Exception as e:
        print(f"  [ERROR] Checkpoint directory issue: {e}")
        print("  Checkpoints might NOT be saved correctly!")

    best_val_loss = float("inf")
    history: list[dict[str, float | int | None]] = []

    train_samples_str = str(train_samples) if inspect_files else "unknown (inspection skipped)"
    val_samples_str = str(val_samples) if inspect_files else "unknown (inspection skipped)"

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Train files:   {len(train_files)}")
    print(f"Train samples: {train_samples_str}")
    if val_files is not None:
        print(f"Val files:     {len(val_files)}")
        print(f"Val samples:   {val_samples_str}")
    print(f"Feature dim:   {detected_input_dim}")
    print(f"Model params:  {num_parameters:,}")

    for epoch in range(start_epoch, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        if use_gpu_buffer:
            train_loader = GPUBufferedBatcher(
                train_files,
                target_table=target_table_cpu,
                batch_size=batch_size,
                device=device,
                normalize_features=normalize_features,
                shuffle_files=True,
                shuffle_points=True,
                gpu_budget_gb=gpu_budget_gb,
            )
        else:
            train_loader = FileFeatureBatcher(
                train_files,
                target_table=target_table_cpu,
                batch_size=batch_size,
                normalize_features=normalize_features,
                shuffle_files=True,
                shuffle_points=True,
            )
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            loss_name=loss_name,
            optimizer=optimizer,
            mse_weight=mse_weight,
            cosine_weight=cosine_weight,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss = None
        if val_files is not None:
            if use_gpu_buffer:
                val_loader = GPUBufferedBatcher(
                    val_files,
                    target_table=target_table_cpu,
                    batch_size=batch_size,
                    device=device,
                    normalize_features=normalize_features,
                    shuffle_files=False,
                    shuffle_points=False,
                    gpu_budget_gb=gpu_budget_gb,
                )
            else:
                val_loader = FileFeatureBatcher(
                    val_files,
                    target_table=target_table_cpu,
                    batch_size=batch_size,
                    normalize_features=normalize_features,
                    shuffle_files=False,
                    shuffle_points=False,
                )
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    device=device,
                    loss_name=loss_name,
                    optimizer=None,
                    mse_weight=mse_weight,
                    cosine_weight=cosine_weight,
                    scaler=None,
                    use_amp=use_amp,
                )

        # Step LR scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "lr": current_lr,
            }
        )

        if val_loss is None:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
                f"| lr={current_lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
                f"| val_loss={val_loss:.6f} | lr={current_lr:.2e}"
            )

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
