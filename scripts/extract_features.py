import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import S3DISDataset
from src.encoder import ConcertoEncoder


def main():
    parser = argparse.ArgumentParser(description="Extract per-point features for S3DIS Area 5.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/s3dis_processed",
        help="Path to processed S3DIS directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="features/s3dis_area5",
        help="Output directory for .npz feature files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for feature extraction, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional local checkpoint path. If omitted, the encoder downloads "
            "Pointcept/Concerto -> concerto_small.pth from Hugging Face."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )
    parser.add_argument(
        "--feature-dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help=(
            "Storage dtype for saved features. "
            "`float16` is much smaller on disk and is sufficient for notebook 03, "
            "which casts features back to float32 during training."
        ),
    )
    parser.add_argument(
        "--save-coord",
        action="store_true",
        help=(
            "Also save XYZ coordinates into the .npz file. "
            "Notebook 03 does not need them, so leaving this off keeps files smaller."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Area 5 from {args.data_dir}...")
    try:
        dataset = S3DISDataset(root=args.data_dir, areas=[5])
    except RuntimeError as error:
        print(error)
        print("Make sure Adrian's data preprocessing is done.")
        return

    encoder = ConcertoEncoder(
        device=device,
        checkpoint_path=args.checkpoint,
    )
    print(f"Using encoder backend: {encoder.backend} on {device}")

    for i in range(len(dataset)):
        sample = dataset[i]
        room_name = sample["room"].replace("/", "_")
        out_path = out_dir / f"{room_name}.npz"

        if out_path.exists() and not args.overwrite:
            print(f"Skipping {room_name}, already exists.")
            continue

        encoder_input = {
            "coord": sample["coord"],
            "color": sample["color"],
        }
        if "normal" in sample:
            encoder_input["normal"] = sample["normal"]

        with torch.no_grad():
            features = encoder(encoder_input)

        feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32
        features_np = features.detach().cpu().numpy().astype(feature_dtype, copy=False)
        labels_np = sample["label"].astype(np.uint8, copy=False)
        print(f"{room_name}: feature dim = {features_np.shape[1]}")
        payload = {
            "features": features_np,
            "labels": labels_np,
        }
        if args.save_coord:
            payload["coord"] = sample["coord"].astype(np.float32, copy=False)

        np.savez_compressed(out_path, **payload)
        approx_mb = out_path.stat().st_size / (1024 ** 2) if out_path.exists() else 0.0
        print(
            f"Saved features for {room_name} -> {out_path} "
            f"({args.feature_dtype}, coord={'yes' if args.save_coord else 'no'}, {approx_mb:.1f} MB)"
        )


if __name__ == "__main__":
    main()
