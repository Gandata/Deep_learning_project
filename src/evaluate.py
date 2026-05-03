import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.translation_head import MLPTranslationHead
from src.clip_utils import CLIPTextEncoder, init_hf, DEFAULT_PROMPT_TEMPLATES
from src.dataset import LABEL_TEXT, NUM_CLASSES


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


def build_model_from_checkpoint(checkpoint_path: str | None, device: torch.device) -> MLPTranslationHead:
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint and "config" in checkpoint:
        model_cfg = checkpoint["config"].get("model", {})
        model = MLPTranslationHead(
            input_dim=model_cfg.get("input_dim", 256),
            hidden_dims=model_cfg.get("hidden_dims", [512, 512]),
            output_dim=model_cfg.get("output_dim", 512),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "gelu"),
            normalize_output=model_cfg.get("normalize_output", True),
        )
    else:
        model = MLPTranslationHead(input_dim=256, hidden_dims=[512, 512], output_dim=512)

    if checkpoint:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="features/s3dis_area5")
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="CLIP model to use for generating text embeddings.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
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

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"Features directory {features_dir} not found. Run extract_features.py first.")
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    init_hf()
    print(f"Loading CLIP model {args.clip_model} ({args.clip_pretrained})...")
    clip_encoder = CLIPTextEncoder(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    class_descriptions = [LABEL_TEXT[i] for i in range(NUM_CLASSES)]
    clip_embeddings_torch = clip_encoder.encode_labels(
        class_descriptions,
        templates=DEFAULT_PROMPT_TEMPLATES,
        normalize=True,
    ).to(device)

    model = build_model_from_checkpoint(args.checkpoint, device=device)
    if args.checkpoint:
        print(f"Loaded translation head from {args.checkpoint}")
    else:
        print("Warning: evaluating with an untrained translation head.")

    all_preds = []
    all_labels = []

    for npz_file in sorted(features_dir.glob("*.npz")):
        data = np.load(npz_file)
        features = torch.from_numpy(data["features"]).float().to(device)
        labels = data["labels"]

        with torch.no_grad():
            pred_clip = model(features)
            pred_clip = F.normalize(pred_clip, dim=-1)
            sims = pred_clip @ clip_embeddings_torch.T
            pred_classes = torch.argmax(sims, dim=-1).cpu().numpy()

        all_preds.append(pred_classes)
        all_labels.append(labels)
        print(f"Evaluated {npz_file.name}")

    if not all_preds:
        print("No features found to evaluate.")
        return

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    oa, miou, ious = compute_metrics(all_preds, all_labels)

    print("\n--- EVALUATION RESULTS ---")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Mean IoU (mIoU):       {miou:.4f}")
    print("Per-class IoU:")
    for c, iou in enumerate(ious):
        val = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"  Class {c:2d}: {val}")


if __name__ == "__main__":
    main()
