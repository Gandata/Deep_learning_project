import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.translation_head import MLPTranslationHead
from src.clip_utils import CLIPTextEncoder, init_hf, DEFAULT_PROMPT_TEMPLATES
from src.dataset import LABEL_TEXT, NUM_CLASSES

FEATURE_KEYS = ("features", "feature", "feat")
LABEL_KEYS = ("labels", "label", "segment")


def compute_metrics(pred_labels, true_labels, num_classes=13):
    oa = np.mean(pred_labels == true_labels)

    ious = []
    for c in range(num_classes):
        intersection = np.sum((pred_labels == c) & (true_labels == c))
        union = np.sum((pred_labels == c) | (true_labels == c))
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)

    miou = np.nanmean(ious)
    return oa, miou, ious


def choose_key(candidates: tuple[str, ...], names: list[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in {path}. Available keys: {names}")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_checkpoint(
    checkpoint_path: str | None,
    device: torch.device,
    inferred_input_dim: int | None = None,
) -> MLPTranslationHead:
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint and "config" in checkpoint:
        model_cfg = checkpoint["config"].get("model", {})
        model = MLPTranslationHead(
            input_dim=model_cfg.get("input_dim", inferred_input_dim or 256),
            hidden_dims=model_cfg.get("hidden_dims", [512, 512]),
            output_dim=model_cfg.get("output_dim", 512),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "gelu"),
            normalize_output=model_cfg.get("normalize_output", True),
        )
    else:
        model = MLPTranslationHead(
            input_dim=inferred_input_dim or 256,
            hidden_dims=[512, 512],
            output_dim=512,
        )

    if checkpoint:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--features_dir", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16384,
        help="Number of points to evaluate at once on GPU.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=None,
        help="CLIP model to use for generating text embeddings.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=None,
        help="Pretrained weights for CLIP.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained translation-head checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {args.batch_size}.")

    checkpoint_config = None
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None

    run_config = checkpoint_config or {}
    if args.config:
        run_config = load_yaml(args.config)

    data_cfg = run_config.get("data", {})
    clip_cfg = run_config.get("clip", {})

    features_dir = Path(args.features_dir or data_cfg.get("train_features_path", "features/s3dis_area5"))
    if not features_dir.exists():
        print(f"Features directory {features_dir} not found. Run extract_features.py first.")
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    init_hf()
    clip_model = args.clip_model or clip_cfg.get("model_name", "ViT-B-32")
    clip_pretrained = args.clip_pretrained or clip_cfg.get("pretrained", "openai")
    clip_templates = clip_cfg.get("templates", DEFAULT_PROMPT_TEMPLATES)
    class_descriptions = data_cfg.get(
        "label_texts",
        [LABEL_TEXT[i] for i in range(NUM_CLASSES)],
    )
    normalize_features = bool(data_cfg.get("normalize_features", False))

    print(f"Loading CLIP model {clip_model} ({clip_pretrained})...")
    clip_encoder = CLIPTextEncoder(
        model_name=clip_model,
        pretrained=clip_pretrained,
        device=device,
    )
    clip_embeddings_torch = clip_encoder.encode_labels(
        class_descriptions,
        templates=clip_templates,
        normalize=True,
    ).to(device)

    all_preds = []
    all_labels = []

    feature_files = sorted(features_dir.glob("*.npz"))
    if not feature_files:
        print("No features found to evaluate.")
        return

    with np.load(feature_files[0]) as first_data:
        names = list(first_data.keys())
        feature_key = choose_key(FEATURE_KEYS, names, feature_files[0])
        inferred_input_dim = int(first_data[feature_key].shape[1])

    model = build_model_from_checkpoint(
        args.checkpoint,
        device=device,
        inferred_input_dim=inferred_input_dim,
    )
    if args.checkpoint:
        print(f"Loaded translation head from {args.checkpoint}")
    else:
        print("Warning: evaluating with an untrained translation head.")
    print(f"Detected feature dim: {inferred_input_dim}")
    print(f"Normalize input features: {normalize_features}")

    for npz_file in feature_files:
        with np.load(npz_file) as data:
            names = list(data.keys())
            feature_key = choose_key(FEATURE_KEYS, names, npz_file)
            label_key = choose_key(LABEL_KEYS, names, npz_file)
            feature_array = data[feature_key]
            labels = np.asarray(data[label_key]).reshape(-1)

            pred_chunks = []
            for start in range(0, feature_array.shape[0], args.batch_size):
                end = min(start + args.batch_size, feature_array.shape[0])
                features = torch.from_numpy(feature_array[start:end]).float().to(device)
                if normalize_features:
                    features = F.normalize(features, dim=-1)

                with torch.no_grad():
                    pred_clip = model(features)
                    pred_clip = F.normalize(pred_clip, dim=-1)
                    sims = pred_clip @ clip_embeddings_torch.T
                    pred_classes = torch.argmax(sims, dim=-1).cpu().numpy()

                pred_chunks.append(pred_classes)
                del features, pred_clip, sims

            pred_classes = np.concatenate(pred_chunks)

        all_preds.append(pred_classes)
        all_labels.append(labels)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Evaluated {npz_file.name}")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    oa, miou, ious = compute_metrics(all_preds, all_labels, num_classes=len(class_descriptions))

    print("\n--- EVALUATION RESULTS ---")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Mean IoU (mIoU):       {miou:.4f}")
    print("Per-class IoU:")
    for c, iou in enumerate(ious):
        val = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"  Class {c:2d}: {val}")


if __name__ == "__main__":
    main()
