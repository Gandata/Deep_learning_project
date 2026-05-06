import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import S3DISDataset
from src.encoder import ConcertoEncoder


def main():
    parser = argparse.ArgumentParser(description="Extract per-point features for selected S3DIS areas.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/s3dis_processed",
        help="Path to processed S3DIS directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory for .npz feature files. "
            "If omitted, a directory like `features/s3dis_area4` is derived from `--areas`."
        ),
    )
    parser.add_argument(
        "--areas",
        type=int,
        nargs="+",
        default=[5],
        help="S3DIS area ids to extract, for example `--areas 4` or `--areas 1 2 3`.",
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
    parser.add_argument(
        "--max-points-per-chunk",
        type=int,
        default=100_000,
        help=(
            "Maximum number of points passed to Concerto in one forward pass. "
            "Large S3DIS rooms can OOM on a T4 if encoded all at once, so extraction "
            "is chunked by default."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    area_ids = [int(area_id) for area_id in args.areas]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        area_suffix = (
            f"area{area_ids[0]}"
            if len(area_ids) == 1
            else "areas_" + "_".join(str(area_id) for area_id in area_ids)
        )
        out_dir = Path(f"features/s3dis_{area_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading S3DIS areas {area_ids} from {args.data_dir}...")
    try:
        dataset = S3DISDataset(root=args.data_dir, areas=area_ids)
    except RuntimeError as error:
        print(error)
        print("Make sure Adrian's data preprocessing is done.")
        return

    encoder = ConcertoEncoder(
        device=device,
        checkpoint_path=args.checkpoint,
    )
    print(f"Using encoder backend: {encoder.backend} on {device}")

    def encode_room_in_chunks(sample: dict[str, np.ndarray], room_name: str) -> torch.Tensor:
        chunk_size = args.max_points_per_chunk
        if chunk_size is None or chunk_size <= 0:
            chunk_size = int(sample["coord"].shape[0])

        while True:
            try:
                feature_chunks: list[torch.Tensor] = []
                total_points = int(sample["coord"].shape[0])
                num_chunks = (total_points + chunk_size - 1) // chunk_size
                print(
                    f"{room_name}: encoding {total_points:,} points "
                    f"in {num_chunks} chunk(s) of up to {chunk_size:,}"
                )

                for chunk_idx, start in enumerate(range(0, total_points, chunk_size), start=1):
                    end = min(start + chunk_size, total_points)
                    encoder_input = {
                        "coord": sample["coord"][start:end],
                        "color": sample["color"][start:end],
                    }
                    if "normal" in sample:
                        encoder_input["normal"] = sample["normal"][start:end]

                    with torch.no_grad():
                        chunk_features = encoder(encoder_input)
                    feature_chunks.append(chunk_features.detach().cpu())
                    print(
                        f"{room_name}: chunk {chunk_idx}/{num_chunks} "
                        f"-> points [{start}:{end}) -> {tuple(chunk_features.shape)}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                return torch.cat(feature_chunks, dim=0)

            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_size <= 16_000:
                    raise
                chunk_size = max(chunk_size // 2, 16_000)
                print(
                    f"{room_name}: CUDA OOM during extraction, retrying with smaller chunks "
                    f"(new chunk size: {chunk_size:,})"
                )

    for i in range(len(dataset)):
        sample = dataset[i]
        room_name = sample["room"].replace("/", "_")
        out_path = out_dir / f"{room_name}.npz"

        if out_path.exists() and not args.overwrite:
            print(f"Skipping {room_name}, already exists.")
            continue

        features = encode_room_in_chunks(sample, room_name)

        feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32
        features_np = features.numpy().astype(feature_dtype, copy=False)
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
