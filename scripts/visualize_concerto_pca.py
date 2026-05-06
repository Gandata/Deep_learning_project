import argparse
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import LABEL_MAP
from src.visualize import DEFAULT_CLASS_COLORS, save_figure

FEATURE_KEYS = ("features", "feature", "feat")
LABEL_KEYS = ("labels", "label", "segment")


def choose_key(candidates: tuple[str, ...], names: list[str], path: Path) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in {path}. Available keys: {names}")


def parse_room_identifier(room_value: str) -> tuple[str, str]:
    room_stem = Path(room_value).stem
    parts = room_stem.split("_")
    if len(parts) < 4 or parts[0] != "Area":
        raise ValueError(
            "`room` must look like `Area_4_conferenceRoom_1` or `Area_4_conferenceRoom_1.npz`."
        )
    area_name = "_".join(parts[:2])
    raw_room_name = "_".join(parts[2:])
    return area_name, raw_room_name


def robust_normalize_rgb(values: np.ndarray, lower_q: float = 2.0, upper_q: float = 98.0) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"`values` must have shape (N, 3), got {values.shape}.")

    rgb = values.astype(np.float32, copy=True)
    for channel in range(3):
        low = np.percentile(rgb[:, channel], lower_q)
        high = np.percentile(rgb[:, channel], upper_q)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            rgb[:, channel] = 0.5
            continue
        rgb[:, channel] = np.clip((rgb[:, channel] - low) / (high - low), 0.0, 1.0)
    return rgb


def colors_to_rgb_strings(colors: np.ndarray) -> list[str]:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"`colors` must have shape (N, 3), got {colors.shape}.")

    if colors.max() <= 1.0:
        colors = colors * 255.0

    colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in colors]


def build_label_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    unique_labels = np.unique(labels.astype(int))
    label_to_rgb: dict[int, np.ndarray] = {}

    for index, label in enumerate(unique_labels):
        color_string = DEFAULT_CLASS_COLORS[index % len(DEFAULT_CLASS_COLORS)]
        rgb_values = color_string.removeprefix("rgb(").removesuffix(")").split(",")
        label_to_rgb[int(label)] = np.array([int(value) for value in rgb_values], dtype=np.uint8)

    return np.stack([label_to_rgb[int(label)] for label in labels], axis=0)


def add_point_cloud_trace(
    fig: go.Figure,
    points: np.ndarray,
    color_strings: list[str],
    subplot_col: int,
    hover_text: np.ndarray | None = None,
    point_size: float = 1.8,
    opacity: float = 0.9,
) -> None:
    trace_kwargs = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "mode": "markers",
        "marker": {
            "size": point_size,
            "color": color_strings,
            "opacity": opacity,
        },
        "showlegend": False,
    }
    if hover_text is not None:
        trace_kwargs["text"] = hover_text.tolist()
        trace_kwargs["hovertemplate"] = "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
    else:
        trace_kwargs["hovertemplate"] = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"

    fig.add_trace(go.Scatter3d(**trace_kwargs), row=1, col=subplot_col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pre-MLP Concerto room features with PCA-to-RGB."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/s3dis_processed",
        help="Processed S3DIS root containing Area_1..Area_6 folders.",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Feature directory such as `features/s3dis_area4`.",
    )
    parser.add_argument(
        "--room",
        type=str,
        required=True,
        help="Room identifier like `Area_4_conferenceRoom_1`.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Optional visualization downsampling cap. Use <= 0 to keep all points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for optional visualization downsampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output path. Defaults to `results/figures/concerto_pre_mlp/<room>`.",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Also try to save a static PNG with Plotly/Kaleido.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help="Point size for the 3D scatter plots.",
    )
    args = parser.parse_args()

    area_name, raw_room_name = parse_room_identifier(args.room)
    features_dir = Path(args.features_dir)
    feature_path = features_dir / f"{area_name}_{raw_room_name}.npz"
    room_path = Path(args.data_dir) / area_name / f"{raw_room_name}.npy"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not room_path.exists():
        raise FileNotFoundError(f"Raw room file not found: {room_path}")

    with np.load(feature_path) as feature_data:
        feature_names = list(feature_data.keys())
        feature_key = choose_key(FEATURE_KEYS, feature_names, feature_path)
        features = np.asarray(feature_data[feature_key], dtype=np.float32)
        feature_labels = None
        label_keys_present = [name for name in LABEL_KEYS if name in feature_names]
        if label_keys_present:
            feature_labels = np.asarray(feature_data[label_keys_present[0]]).reshape(-1)

    room_array = np.load(room_path).astype(np.float32)
    points = room_array[:, :3]
    colors = room_array[:, 3:6]
    labels = room_array[:, 6].astype(np.int64)

    if features.ndim != 2:
        raise ValueError(f"Expected room features with shape (N, D), got {features.shape}.")
    if features.shape[0] != points.shape[0]:
        raise ValueError(
            "Raw room and extracted features have different numbers of points: "
            f"{points.shape[0]} vs {features.shape[0]}."
        )
    if feature_labels is not None and feature_labels.shape[0] == labels.shape[0]:
        if not np.array_equal(feature_labels.astype(labels.dtype, copy=False), labels):
            print("Warning: labels inside the feature file do not exactly match raw-room labels.")

    keep_indices = np.arange(points.shape[0])
    if args.max_points and args.max_points > 0 and points.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep_indices = np.sort(rng.choice(points.shape[0], size=args.max_points, replace=False))
        points = points[keep_indices]
        colors = colors[keep_indices]
        labels = labels[keep_indices]
        features = features[keep_indices]

    pca = PCA(n_components=3, random_state=args.seed)
    pca_rgb = robust_normalize_rgb(pca.fit_transform(features))

    raw_rgb_strings = colors_to_rgb_strings(colors)
    label_rgb_strings = colors_to_rgb_strings(build_label_rgb(labels))
    pca_rgb_strings = colors_to_rgb_strings(pca_rgb)
    hover_labels = np.array(
        [f"label={int(label)} ({LABEL_MAP.get(int(label), 'unknown')})" for label in labels],
        dtype=object,
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Raw RGB",
            "Ground-Truth Labels",
            "Concerto Features (PCA to RGB)",
        ),
        horizontal_spacing=0.02,
    )

    add_point_cloud_trace(
        fig,
        points=points,
        color_strings=raw_rgb_strings,
        subplot_col=1,
        point_size=args.point_size,
    )
    add_point_cloud_trace(
        fig,
        points=points,
        color_strings=label_rgb_strings,
        subplot_col=2,
        hover_text=hover_labels,
        point_size=args.point_size,
    )
    add_point_cloud_trace(
        fig,
        points=points,
        color_strings=pca_rgb_strings,
        subplot_col=3,
        point_size=args.point_size,
    )

    fig.update_layout(
        title=(
            f"{area_name}/{raw_room_name} - pre-MLP Concerto feature visualization "
            f"(showing {points.shape[0]:,} points)"
        ),
        width=1800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.update_scenes(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        aspectmode="data",
        bgcolor="white",
    )

    output_base = (
        Path(args.output)
        if args.output
        else Path("results/figures/concerto_pre_mlp") / f"{area_name}_{raw_room_name}_pca"
    )
    saved_paths = save_figure(
        fig,
        output_base,
        width=1800,
        height=700,
        scale=2,
        save_html=True,
        save_png=args.save_png,
    )

    explained = pca.explained_variance_ratio_
    print(f"Room file:      {room_path}")
    print(f"Feature file:   {feature_path}")
    print(f"Visualized N:   {points.shape[0]:,}")
    print(
        "PCA explained variance ratio: "
        f"[{explained[0]:.4f}, {explained[1]:.4f}, {explained[2]:.4f}]"
    )
    print(f"Saved HTML:     {saved_paths['html']}")
    print(f"Saved PNG:      {saved_paths['png']}")


if __name__ == "__main__":
    main()
