import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.clip_utils import CLIPTextEncoder, DEFAULT_PROMPT_TEMPLATES, init_hf
from src.dataset import LABEL_MAP, LABEL_TEXT, NUM_CLASSES
from src.translation_head import MLPTranslationHead

FEATURE_KEYS = ("features", "feature", "feat")
LABEL_KEYS = ("labels", "label", "segment")


def choose_key(candidates: tuple[str, ...], names: list[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in {path}. Available keys: {names}")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_stem(path_value: str | Path | None, fallback: str) -> str:
    if not path_value:
        return fallback
    stem = Path(path_value).stem.strip()
    return stem or fallback


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


def infer_room_prefix(npz_file: Path) -> str:
    stem_parts = npz_file.stem.split("_")
    if len(stem_parts) >= 4 and stem_parts[0] == "Area":
        prefix_parts = stem_parts[2:-1]
        if prefix_parts:
            return "_".join(prefix_parts)
    return npz_file.stem


def safe_mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not valid:
        return None
    return float(np.mean(valid))


def safe_std(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not valid:
        return None
    return float(np.std(valid))


def select_room_files(
    feature_files: list[Path],
    rooms_per_prefix: int,
    seed: int,
) -> tuple[list[Path], dict[str, list[str]], dict[str, int]]:
    rooms_by_prefix: dict[str, list[Path]] = defaultdict(list)
    for feature_file in feature_files:
        rooms_by_prefix[infer_room_prefix(feature_file)].append(feature_file)

    rng = random.Random(seed)
    selected_files: list[Path] = []
    selected_rooms_by_prefix: dict[str, list[str]] = {}
    candidate_counts: dict[str, int] = {}

    for prefix, files in sorted(rooms_by_prefix.items()):
        files = sorted(files, key=lambda path: path.name)
        candidate_counts[prefix] = len(files)
        if len(files) <= rooms_per_prefix:
            keep_files = files
        else:
            keep_files = sorted(rng.sample(files, rooms_per_prefix), key=lambda path: path.name)
        selected_rooms_by_prefix[prefix] = [path.name for path in keep_files]
        selected_files.extend(keep_files)

    return sorted(selected_files, key=lambda path: path.name), selected_rooms_by_prefix, candidate_counts


def compute_query_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int | None]:
    positive_count = int(targets.sum())
    total_count = int(targets.shape[0])
    negative_count = total_count - positive_count

    positive_scores = scores[targets]
    negative_scores = scores[~targets]

    topk_indices = (
        np.arange(total_count)
        if positive_count >= total_count
        else np.argpartition(scores, -positive_count)[-positive_count:]
    )
    predicted_mask = np.zeros(total_count, dtype=bool)
    predicted_mask[topk_indices] = True
    tp = int(np.logical_and(predicted_mask, targets).sum())
    union = int(np.logical_or(predicted_mask, targets).sum())

    average_precision = None
    roc_auc = None
    if positive_count > 0 and negative_count > 0:
        target_int = targets.astype(np.uint8)
        average_precision = float(average_precision_score(target_int, scores))
        roc_auc = float(roc_auc_score(target_int, scores))

    return {
        "num_points": total_count,
        "num_positive_points": positive_count,
        "positive_ratio": float(positive_count / total_count) if total_count else None,
        "positive_mean_cosine": float(positive_scores.mean()) if positive_scores.size else None,
        "negative_mean_cosine": float(negative_scores.mean()) if negative_scores.size else None,
        "cosine_gap": (
            float(positive_scores.mean() - negative_scores.mean())
            if positive_scores.size and negative_scores.size
            else None
        ),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "topk_precision": float(tp / positive_count) if positive_count else None,
        "topk_iou": float(tp / union) if union else None,
    }


def summarize_rows(
    rows: list[dict[str, object]],
    group_key: str,
    text_key: str,
) -> list[dict[str, object]]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row[group_key])].append(row)

    summaries: list[dict[str, object]] = []
    for group_value, group_rows in sorted(grouped_rows.items()):
        summaries.append(
            {
                group_key: group_value,
                text_key: " | ".join(str(row["label_name"]) for row in group_rows),
                "num_queries": len(group_rows),
                "mean_average_precision": safe_mean([row["average_precision"] for row in group_rows]),
                "std_average_precision": safe_std([row["average_precision"] for row in group_rows]),
                "mean_roc_auc": safe_mean([row["roc_auc"] for row in group_rows]),
                "std_roc_auc": safe_std([row["roc_auc"] for row in group_rows]),
                "mean_topk_iou": safe_mean([row["topk_iou"] for row in group_rows]),
                "std_topk_iou": safe_std([row["topk_iou"] for row in group_rows]),
                "mean_cosine_gap": safe_mean([row["cosine_gap"] for row in group_rows]),
                "std_cosine_gap": safe_std([row["cosine_gap"] for row in group_rows]),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--features_dir", type=str, default=None)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/evaluation/04b",
        help="Directory where evaluation-label outputs will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16384,
        help="Number of points to translate at once on GPU.",
    )
    parser.add_argument(
        "--rooms_per_prefix",
        type=int,
        default=3,
        help="How many random rooms to evaluate per room prefix.",
    )
    parser.add_argument(
        "--labels_per_room",
        type=int,
        default=3,
        help="How many random labels present in each room to query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for room and label sampling.",
    )
    parser.add_argument("--clip_model", type=str, default=None)
    parser.add_argument("--clip_pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {args.batch_size}.")
    if args.rooms_per_prefix <= 0:
        raise ValueError(f"`rooms_per_prefix` must be positive, got {args.rooms_per_prefix}.")
    if args.labels_per_room <= 0:
        raise ValueError(f"`labels_per_room` must be positive, got {args.labels_per_room}.")

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

    features_dir = Path(args.features_dir or data_cfg.get("train_features_path", "features/s3dis_area4"))
    if not features_dir.exists():
        print(f"Features directory {features_dir} not found. Run extract_features.py first.")
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    normalize_features = bool(data_cfg.get("normalize_features", False))
    class_descriptions = data_cfg.get(
        "label_texts",
        [LABEL_TEXT[i] for i in range(NUM_CLASSES)],
    )
    clip_model = args.clip_model or clip_cfg.get("model_name", "ViT-B-32")
    clip_pretrained = args.clip_pretrained or clip_cfg.get("pretrained", "openai")
    clip_templates = clip_cfg.get("templates", DEFAULT_PROMPT_TEMPLATES)

    results_dir = Path(args.results_dir)
    area_level_dir = results_dir / "area_level"
    prefix_level_dir = results_dir / "prefix_level"
    room_level_dir = results_dir / "room_level"
    query_level_dir = results_dir / "query_level"
    area_level_dir.mkdir(parents=True, exist_ok=True)
    prefix_level_dir.mkdir(parents=True, exist_ok=True)
    room_level_dir.mkdir(parents=True, exist_ok=True)
    query_level_dir.mkdir(parents=True, exist_ok=True)

    area_tag = features_dir.name
    checkpoint_tag = safe_stem(args.checkpoint, "untrained")

    init_hf()
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

    feature_files = sorted(features_dir.glob("*.npz"))
    if not feature_files:
        print("No features found to evaluate.")
        return

    selected_files, selected_rooms_by_prefix, candidate_counts = select_room_files(
        feature_files,
        rooms_per_prefix=args.rooms_per_prefix,
        seed=args.seed,
    )

    print(f"Selected {len(selected_files)} rooms from {len(feature_files)} total files.")
    for prefix, room_names in selected_rooms_by_prefix.items():
        print(f"  {prefix}: {len(room_names)}/{candidate_counts[prefix]} rooms -> {room_names}")

    with np.load(selected_files[0]) as first_data:
        names = list(first_data.keys())
        feature_key = choose_key(FEATURE_KEYS, names, selected_files[0])
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

    rng = random.Random(args.seed)
    query_rows: list[dict[str, object]] = []

    for room_file in selected_files:
        room_prefix = infer_room_prefix(room_file)
        with np.load(room_file) as data:
            names = list(data.keys())
            feature_key = choose_key(FEATURE_KEYS, names, room_file)
            label_key = choose_key(LABEL_KEYS, names, room_file)
            feature_array = np.asarray(data[feature_key])
            labels = np.asarray(data[label_key]).reshape(-1)

        present_label_ids = sorted(int(label_id) for label_id in np.unique(labels))
        sampled_label_ids = (
            present_label_ids
            if len(present_label_ids) <= args.labels_per_room
            else sorted(rng.sample(present_label_ids, args.labels_per_room))
        )
        sampled_text_embeddings = clip_embeddings_torch.index_select(
            0,
            torch.tensor(sampled_label_ids, device=device),
        )

        score_chunks = [[] for _ in sampled_label_ids]
        for start in range(0, feature_array.shape[0], args.batch_size):
            end = min(start + args.batch_size, feature_array.shape[0])
            features = torch.from_numpy(feature_array[start:end]).float().to(device)
            if normalize_features:
                features = F.normalize(features, dim=-1)

            with torch.no_grad():
                pred_clip = model(features)
                pred_clip = F.normalize(pred_clip, dim=-1)
                sims = pred_clip @ sampled_text_embeddings.T
                sims_np = sims.detach().cpu().numpy()

            for label_index in range(len(sampled_label_ids)):
                score_chunks[label_index].append(sims_np[:, label_index])

            del features, pred_clip, sims
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for label_index, label_id in enumerate(sampled_label_ids):
            scores = np.concatenate(score_chunks[label_index], axis=0)
            targets = labels == label_id
            metrics = compute_query_metrics(scores=scores, targets=targets)
            query_rows.append(
                {
                    "room_prefix": room_prefix,
                    "room_file": room_file.name,
                    "label_id": int(label_id),
                    "label_name": LABEL_MAP.get(int(label_id), f"class_{label_id}"),
                    "label_text": class_descriptions[int(label_id)],
                    **metrics,
                }
            )

        print(f"Evaluated label queries for {room_file.name}: {sampled_label_ids}")

    room_rows = summarize_rows(query_rows, group_key="room_file", text_key="sampled_labels")
    prefix_rows = summarize_rows(query_rows, group_key="room_prefix", text_key="sampled_labels")

    for row in room_rows:
        matching_query = next(
            query_row for query_row in query_rows if query_row["room_file"] == row["room_file"]
        )
        row["room_prefix"] = matching_query["room_prefix"]
    room_rows = sorted(room_rows, key=lambda row: str(row["room_file"]))

    for row in prefix_rows:
        row["num_candidate_rooms"] = candidate_counts.get(str(row["room_prefix"]), 0)
        row["num_selected_rooms"] = len(selected_rooms_by_prefix.get(str(row["room_prefix"]), []))
        row["selected_rooms"] = " | ".join(selected_rooms_by_prefix.get(str(row["room_prefix"]), []))

    area_summary = {
        "features_dir": str(features_dir),
        "checkpoint": args.checkpoint,
        "clip_model": clip_model,
        "clip_pretrained": clip_pretrained,
        "normalize_features": normalize_features,
        "batch_size": int(args.batch_size),
        "rooms_per_prefix": int(args.rooms_per_prefix),
        "labels_per_room": int(args.labels_per_room),
        "seed": int(args.seed),
        "num_candidate_rooms": len(feature_files),
        "num_selected_rooms": len(selected_files),
        "num_queries": len(query_rows),
        "selected_rooms_by_prefix": selected_rooms_by_prefix,
        "mean_average_precision": safe_mean([row["average_precision"] for row in query_rows]),
        "std_average_precision": safe_std([row["average_precision"] for row in query_rows]),
        "mean_roc_auc": safe_mean([row["roc_auc"] for row in query_rows]),
        "std_roc_auc": safe_std([row["roc_auc"] for row in query_rows]),
        "mean_topk_iou": safe_mean([row["topk_iou"] for row in query_rows]),
        "std_topk_iou": safe_std([row["topk_iou"] for row in query_rows]),
        "mean_cosine_gap": safe_mean([row["cosine_gap"] for row in query_rows]),
        "std_cosine_gap": safe_std([row["cosine_gap"] for row in query_rows]),
    }

    area_json_path = area_level_dir / f"{area_tag}_{checkpoint_tag}_summary.json"
    with area_json_path.open("w", encoding="utf-8") as handle:
        json.dump(area_summary, handle, indent=2)

    query_csv_path = query_level_dir / f"{area_tag}_{checkpoint_tag}_query_metrics.csv"
    query_fieldnames = list(query_rows[0].keys()) if query_rows else []
    with query_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)

    room_csv_path = room_level_dir / f"{area_tag}_{checkpoint_tag}_room_metrics.csv"
    room_fieldnames = list(room_rows[0].keys()) if room_rows else []
    with room_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=room_fieldnames)
        writer.writeheader()
        writer.writerows(room_rows)

    prefix_csv_path = prefix_level_dir / f"{area_tag}_{checkpoint_tag}_prefix_metrics.csv"
    prefix_fieldnames = list(prefix_rows[0].keys()) if prefix_rows else []
    with prefix_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prefix_fieldnames)
        writer.writeheader()
        writer.writerows(prefix_rows)

    print("\n--- LABEL QUERY EVALUATION RESULTS ---")
    print(f"Selected rooms:         {len(selected_files)} / {len(feature_files)}")
    print(f"Total label queries:    {len(query_rows)}")
    print(f"Mean AP:                {area_summary['mean_average_precision']}")
    print(f"Mean ROC-AUC:           {area_summary['mean_roc_auc']}")
    print(f"Mean top-k IoU:         {area_summary['mean_topk_iou']}")
    print(f"Mean cosine gap:        {area_summary['mean_cosine_gap']}")
    print(f"\nSaved area summary to:  {area_json_path}")
    print(f"Saved query metrics to: {query_csv_path}")
    print(f"Saved room metrics to:  {room_csv_path}")
    print(f"Saved prefix metrics to:{prefix_csv_path}")


if __name__ == "__main__":
    main()
