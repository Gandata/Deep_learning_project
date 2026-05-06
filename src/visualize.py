from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_POINT_COLOR = "rgb(114,162,224)"  # light blue

DEFAULT_CLASS_COLORS = [
    "rgb(31,119,180)",   # blue
    "rgb(255,127,14)",   # orange
    "rgb(44,160,44)",    # green
    "rgb(214,39,40)",    # red
    "rgb(148,103,189)",  # purple
    "rgb(140,86,75)",    # brown
    "rgb(227,119,194)",  # pink
    "rgb(127,127,127)",  # gray
    "rgb(188,189,34)",   # olive
    "rgb(23,190,207)",   # cyan
    "rgb(114,162,224)",   # light blue
    "rgb(255,187,120)",   # light orange
    "rgb(152,223,138)",   # light green
    "rgb(255,152,150)",   # light red / salmon
    "rgb(197,176,213)",   # light purple / lavender
    "rgb(196,156,148)",   # light brown
]


def _validate_points(points: np.ndarray) -> np.ndarray:
    '''
    Checks for right dimensionality of the vector of input points, i.e. (N,3)
    '''
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"`points` must have shape (N, 3), got {points.shape}."
        )
    if len(points) == 0:
        raise ValueError("`points` is empty.")
    return points


def _colors_to_rgb_strings(colors: np.ndarray, n_points: int) -> list[str]:
    """
    Checks for right dimensionality of the colors vector, i.e. (N, 3), where N is the number of points in the point cloud.
    Convert an (N, 3) RGB array into Plotly-compatible 'rgb(r,g,b)' strings.
    Accepts:
    - float colors in [0, 1]
    - int/float colors in [0, 255]
    """
    colors = np.asarray(colors)

    if colors.ndim != 2 or colors.shape != (n_points, 3):
        raise ValueError(
            f"`colors` must have shape ({n_points}, 3), got {colors.shape}."
        )

    if not np.issubdtype(colors.dtype, np.number):
        raise TypeError("`colors` must contain numeric values.")

    colors = colors.astype(np.float32)

    # If colors look normalized to [0, 1], scale them to [0, 255].
    if colors.max() <= 1.0:
        colors = colors * 255.0

    colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in colors]


def _build_label_color_map(unique_labels: np.ndarray) -> dict[int, str]:
    '''
    Returns a dictionary which maps each label to a distinct color chosen from the default colors
    '''
    unique_labels = np.sort(unique_labels.astype(int))
    return {
        int(label): DEFAULT_CLASS_COLORS[i % len(DEFAULT_CLASS_COLORS)]
        for i, label in enumerate(unique_labels)
    }


def plot_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    point_size: float = 2.0,
    opacity: float = 0.9,
    title: Optional[str] = None,
    default_color: str = DEFAULT_POINT_COLOR,
    legend_marker_size: float = 8.0,
) -> go.Figure:
    """
    Plot a 3D point cloud with Plotly taking as parameters: 
        - points, an array of shape (N, 3) with XYZ coordinates.
        - colors, an array of shape (N, 3) with RGB colors.
        - labes, an (optional) array of shape (N,) with integer class labels. If provided, the point cloud is colored by class label instead of RGB.

    Returns a Plotly figure object. 

    Priority of coloring:
      - If labels is provided -> color by semantic class
      - If colors is provided -> color by RGB
      - Else -> use a single default color for all points
    """
    
    points = _validate_points(points)
    n_points = points.shape[0]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = go.Figure()

    if labels is not None:
        labels = np.asarray(labels)

        if labels.ndim != 1 or labels.shape[0] != n_points:
            raise ValueError(
                f"`labels` must have shape ({n_points},), got {labels.shape}."
            )

        unique_labels = np.unique(labels)
        label_to_color = _build_label_color_map(unique_labels)

        for label in unique_labels:
            mask = labels == label
            color = label_to_color[int(label)]

            # real trace for points in the cloud
            fig.add_trace(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z[mask],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=label_to_color[int(label)],
                        opacity=opacity,
                    ),
                    name=f"class {int(label)}",
                    showlegend=False,
                    hovertemplate=(
                        f"label={int(label)}<br>"
                        "x=%{x:.3f}<br>"
                        "y=%{y:.3f}<br>"
                        "z=%{z:.3f}<extra></extra>"
                    ),
                )
            )

            # trace only for the legend (bigger markers)
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=legend_marker_size,
                        color=color,
                        opacity=1.0,
                    ),
                    name=f"class {int(label)}",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        if title is None:
            title = "Point Cloud (Labels)"

    else:
        if colors is not None:
            rgb_strings = _colors_to_rgb_strings(colors, n_points)
            marker_color = rgb_strings
            if title is None:
                title = "Point Cloud (RGB)"
        else:
            marker_color = default_color
            if title is None:
                title = "Point Cloud"

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=marker_color,
                    opacity=opacity,
                ),
                name="points",
                hovertemplate=(
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=(labels is not None),
        legend=dict(
            itemsizing="constant",
            itemwidth=50,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="white",
        ),
    )

    return fig


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim != 1:
        raise ValueError(f"`scores` must have shape (N,), got {scores.shape}.")

    if len(scores) == 0:
        raise ValueError("`scores` is empty.")

    if not np.isfinite(scores).all():
        raise ValueError("`scores` contains NaN or Inf values.")

    return scores


def plot_heatmap(
    points: np.ndarray,
    scores: np.ndarray,
    query_text: str,
    point_size: float = 2.0,
    opacity: float = 0.9,
    title: Optional[str] = None,
    colorscale: str = "Turbo",
    top_percent: Optional[float] = None,
    low_opacity: float = 0.08,
    show_colorbar: bool = True,
    reverse_colorbar: bool = False,
) -> go.Figure:
    """
    Plot a 3D heatmap over a point cloud using continuous similarity scores (based on a specific query). 

    Parameters (the non trivials):
    - points, an array of shape (N, 3) with XYZ coordinates.
    - scores, an array of shape (N,) with one score per point.
    - query_text, the text query used to generate the scores.
    - top_percent, if provided highlights only the top X percent of points.
        (Example: 5.0 means keep the top 5% highest-scoring points prominent.)
    - low_opacity, opacity assigned to non-highlighted points when top_percent is used.
    """
    points = _validate_points(points)
    scores = _normalize_scores(scores)

    n_points = points.shape[0]
    if scores.shape[0] != n_points:
        raise ValueError(
            f"`scores` must have shape ({n_points},), got {scores.shape}."
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if title is None:
        title = f'Heatmap for query: "{query_text}"'

    fig = go.Figure()
    show_trace_colorbar = show_colorbar and not reverse_colorbar

    if top_percent is None:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=scores,
                    colorscale=colorscale,
                    opacity=opacity,
                    colorbar=dict(
                        title="score",
                        thickness=18,
                    ),
                    showscale=show_trace_colorbar,
                ),
                name="heatmap",
                showlegend=False,
                hovertemplate=(
                    "score=%{marker.color:.4f}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            )
        )
    else:
        if not (0.0 < top_percent <= 100.0):
            raise ValueError("`top_percent` must be in the interval (0, 100].")

        threshold = np.percentile(scores, 100.0 - top_percent)
        top_mask = scores >= threshold
        bg_mask = ~top_mask

        # Background points: same colorscale, low opacity, no colorbar
        fig.add_trace(
            go.Scatter3d(
                x=x[bg_mask],
                y=y[bg_mask],
                z=z[bg_mask],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=scores[bg_mask],
                    colorscale=colorscale,
                    opacity=low_opacity,
                    showscale=False,
                ),
                name="background",
                showlegend=False,
                hovertemplate=(
                    "score=%{marker.color:.4f}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            )
        )

        # Top points: strong opacity + visible colorbar
        fig.add_trace(
            go.Scatter3d(
                x=x[top_mask],
                y=y[top_mask],
                z=z[top_mask],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=scores[top_mask],
                    colorscale=colorscale,
                    opacity=opacity,
                    colorbar=dict(
                        title="score",
                        thickness=18,
                    ),
                    showscale=show_trace_colorbar,
                ),
                name="top points",
                showlegend=False,
                hovertemplate=(
                    "score=%{marker.color:.4f}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    if show_colorbar and reverse_colorbar:
        fig.add_trace(
            go.Scatter3d(
                x=[x[0], x[0]],
                y=[y[0], y[0]],
                z=[z[0], z[0]],
                mode="markers",
                marker=dict(
                    size=0.01,
                    color=[float(scores.min()), float(scores.max())],
                    cmin=float(scores.min()),
                    cmax=float(scores.max()),
                    colorscale=colorscale,
                    reversescale=True,
                    opacity=0.0,
                    colorbar=dict(
                        title="score",
                        thickness=18,
                    ),
                    showscale=True,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="white",
        ),
    )

    return fig


def plot_class_comparison(
    points: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    point_size: float = 2.0,
    opacity: float = 0.9,
    title: str = "Prediction vs Ground Truth",
    legend_marker_size: float = 8.0,
) -> go.Figure:
    """
    Plot side-by-side 3D comparison between predicted labels and ground-truth labels.

    Parameters (the non trivials):
    points, an array of shape (N, 3) with XYZ coordinates.
    pred_labels, a predicted class labels of shape (N,).
    gt_labels, the Ground-truth class labels of shape (N,).
    """
    points = _validate_points(points)

    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)

    n_points = points.shape[0]

    if pred_labels.ndim != 1 or pred_labels.shape[0] != n_points:
        raise ValueError(
            f"`pred_labels` must have shape ({n_points},), got {pred_labels.shape}."
        )

    if gt_labels.ndim != 1 or gt_labels.shape[0] != n_points:
        raise ValueError(
            f"`gt_labels` must have shape ({n_points},), got {gt_labels.shape}."
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    all_labels = np.unique(np.concatenate([pred_labels, gt_labels]))
    label_to_color = _build_label_color_map(all_labels)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Predicted Labels", "Ground Truth Labels"),
        horizontal_spacing=0.03,
    )

    # Left subplot: predictions
    for label in np.unique(pred_labels):
        mask = pred_labels == label
        color = label_to_color[int(label)]

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=opacity,
                ),
                name=f"class {int(label)}",
                legendgroup=f"class_{int(label)}",
                showlegend=False,
                hovertemplate=(
                    f"pred={int(label)}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Right subplot: ground truth
    for label in np.unique(gt_labels):
        mask = gt_labels == label
        color = label_to_color[int(label)]

        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=opacity,
                ),
                name=f"class {int(label)}",
                legendgroup=f"class_{int(label)}",
                showlegend=False,
                hovertemplate=(
                    f"gt={int(label)}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Legend-only traces: one per class, visible once
    for label in all_labels:
        color = label_to_color[int(label)]
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=legend_marker_size,
                    color=color,
                    opacity=1.0,
                ),
                name=f"class {int(label)}",
                legendgroup=f"class_{int(label)}",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=title,
        width=1400,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            itemsizing="constant",
            itemwidth=50,
            font=dict(size=15),
        ),
    )

    fig.update_scenes(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        aspectmode="data",
        bgcolor="white",
    )

    return fig


def save_figure(
    fig: go.Figure,
    path: str | Path,
    width: int = 1400,
    height: int = 900,
    scale: int = 2,
    save_html: bool = True,
    save_png: bool = True,
) -> dict[str, Optional[Path]]:
    """
    Save a Plotly figure to disk.
    By default is saves both an interactive HTML version and a static PNG version.

    Parameters:
    - fig, a Plotly figure to save.
    - path, the Base output path. You can pass it with or without extension.
        Examples:
            "results/figures/room1_heatmap"
            "results/figures/room1_heatmap.html"
            "results/figures/room1_heatmap.png"
    - width, the width used for static image export.
    - height, the height used for static image export.
    - scale, the scale factor for static export (higher = sharper image).
    - save_html, True to save the interactive HTML file (False to don't).
    - save_png, True to save the static PNG file (False to don't).

    Returns a dictionary with the saved paths:
        {
            "html": Path | None,
            "png": Path | None,
        }

    Notes: PNG export requires Kaleido. With Kaleido v1+, Chrome/Chromium must also be available on the system.
    """
    path = Path(path)

    # Remove extension if user passed .html or .png
    if path.suffix.lower() in {".html", ".png"}:
        base_path = path.with_suffix("")
    else:
        base_path = path

    base_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = base_path.with_suffix(".html") if save_html else None
    png_path = base_path.with_suffix(".png") if save_png else None

    if save_html and html_path is not None:
        fig.write_html(
            html_path,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
        )

    if save_png and png_path is not None:
        try:
            fig.write_image(
                png_path,
                width=width,
                height=height,
                scale=scale,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to export PNG with Plotly. "
                "Make sure `kaleido` is installed and Chrome/Chromium is available. "
                f"Original error: {e}"
            ) from e

    return {
        "html": html_path,
        "png": png_path,
    }






