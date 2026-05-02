from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from torch import Tensor, nn


ArrayLike = Union[np.ndarray, Tensor]
PointInput = Union[ArrayLike, Mapping[str, ArrayLike]]

load_dotenv()


def _to_tensor(value: ArrayLike, name: str) -> Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    if torch.is_tensor(value):
        return value.float()
    raise TypeError(f"`{name}` must be a numpy array or a torch tensor.")


class ConcertoEncoder(nn.Module):
    """
    Concerto encoder wrapper for the project pipeline.

    Supported inputs:
    - tensor/ndarray of shape (N, 3), (N, 6), or (N, 9)
    - dict-like inputs with keys such as:
      - `points` or `coord` for XYZ
      - `colors` or `color` for RGB
      - `normal` or `normals` for surface normals

    Output:
    - (N, feature_dim)

    Notes:
    - this encoder always uses the official Concerto backend
    - weights can be loaded either from a local checkpoint or from Hugging Face
    """

    def __init__(
        self,
        feature_dim: int = 256,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        repo_id: str = "Pointcept/Concerto",
        model_name: str = "concerto_small",
        enable_flash: bool = False,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.repo_id = repo_id
        self.model_name = model_name
        self.enable_flash = enable_flash
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backbone: nn.Module | None = None
        self._concerto_transform = None

        self._init_concerto_backend()
        self.backend = "concerto"
        self.eval()
        self.to(self.device)

    def _login_hf_if_available(self) -> None:
        token = os.getenv("HF_TOKEN")
        if token and token != "your_huggingface_token_here":
            try:
                login(token=token)
            except Exception as error:
                warnings.warn(f"HF login failed: {error}")

    def _init_concerto_backend(self) -> None:
        try:
            import concerto
        except Exception:
            concerto_dir = os.getenv("CONCERTO_DIR", "/content/Concerto")
            if Path(concerto_dir).exists() and concerto_dir not in sys.path:
                sys.path.insert(0, concerto_dir)
            try:
                import concerto
            except Exception as error:
                raise ImportError(
                    "The real Concerto backend requires the official `concerto` package "
                    "and its dependencies. Install them following the official repo."
                ) from error

        self._login_hf_if_available()

        custom_config = None
        if not self.enable_flash:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)],
                enable_flash=False,
            )

        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path not found: {self.checkpoint_path}"
                )
            resolved_checkpoint = self.checkpoint_path
            print(f"Loading Concerto Small from local checkpoint: {resolved_checkpoint}")
        else:
            filename = f"{self.model_name}.pth"
            resolved_checkpoint = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                )
            )
            print(
                "Loading Concerto Small from Hugging Face cache: "
                f"{resolved_checkpoint}"
            )

        model = concerto.model.load(
            str(resolved_checkpoint),
            custom_config=custom_config,
        )

        self.backbone = model
        self._concerto_transform = concerto.transform.default()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def _extract_normal_from_mapping(
        self,
        points: Mapping[str, ArrayLike],
        coord: Tensor,
    ) -> Tensor:
        normal_value = points.get("normal")
        if normal_value is None:
            normal_value = points.get("normals")
        if normal_value is None:
            return torch.zeros_like(coord)
        normal = _to_tensor(normal_value, "normal/normals")
        if normal.shape != coord.shape:
            raise ValueError(
                f"`normal` must match coord shape {tuple(coord.shape)}, got {tuple(normal.shape)}."
            )
        return normal

    def _split_point_input(
        self,
        points: PointInput,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(points, Mapping):
            coord_value = points.get("points")
            if coord_value is None:
                coord_value = points.get("coord")
            if coord_value is None:
                raise KeyError(
                    "Point dict input must contain `points` or `coord`."
                )

            color_value = points.get("colors")
            if color_value is None:
                color_value = points.get("color")

            coord = _to_tensor(coord_value, "points/coord")
            color = None if color_value is None else _to_tensor(color_value, "colors/color")
            normal = self._extract_normal_from_mapping(points, coord)
            return coord, color, normal

        tensor = _to_tensor(points, "points")
        if tensor.shape[-1] < 3:
            raise ValueError(
                "Point inputs must have at least 3 channels for XYZ coordinates."
            )

        coord = tensor[..., :3]
        color = tensor[..., 3:6] if tensor.shape[-1] >= 6 else None
        normal = tensor[..., 6:9] if tensor.shape[-1] >= 9 else None
        return coord, color, normal

    def _prepare_concerto_point(self, points: PointInput) -> dict[str, np.ndarray]:
        if isinstance(points, Mapping):
            coord, color, normal = self._split_point_input(points)
            segment_value = points.get("segment")
            if segment_value is None:
                segment_value = points.get("label")
        else:
            coord, color, normal = self._split_point_input(points)
            segment_value = None

        if coord.ndim != 2 or coord.shape[-1] != 3:
            raise ValueError(
                "Concerto inference currently supports a single point cloud "
                "with shape (N, 3) for coordinates."
            )

        if color is None:
            color = torch.zeros_like(coord)
        if normal is None:
            normal = torch.zeros_like(coord)
        if color.max().item() <= 1.0:
            color = color * 255.0

        point = {
            "coord": coord.detach().cpu().numpy().astype(np.float32),
            "color": color.detach().cpu().numpy().astype(np.float32),
            "normal": normal.detach().cpu().numpy().astype(np.float32),
        }

        if segment_value is not None:
            segment = _to_tensor(segment_value, "segment/label").long().reshape(-1)
            point["segment"] = segment.detach().cpu().numpy()

        return point

    def _run_concerto(self, points: PointInput) -> Tensor:
        assert self.backbone is not None
        assert self._concerto_transform is not None

        point = self._prepare_concerto_point(points)
        point = self._concerto_transform(point)
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(self.device, non_blocking=True)

        with torch.no_grad():
            point = self.backbone(point)

        # Official feature recovery from the Concerto README.
        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        feat = point.feat[point.inverse]
        feat = F.normalize(feat, dim=-1)

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Concerto produced feature dim {feat.shape[-1]}, but this project "
                f"currently expects {self.feature_dim}. "
                "Adjust the project config or add an explicit adapter."
            )

        return feat

    def forward(self, points: PointInput) -> Tensor:
        return self._run_concerto(points)

    @torch.no_grad()
    def encode(self, points: PointInput) -> Tensor:
        return self.forward(points)


if __name__ == "__main__":
    encoder = ConcertoEncoder(feature_dim=256)

    xyzrgb = torch.randn(1024, 6)
    xyzrgb[:, 3:6] = torch.rand(1024, 3) * 255.0
    features_a = encoder(xyzrgb)

    structured = {
        "points": torch.randn(512, 3),
        "colors": torch.rand(512, 3),
    }
    features_b = encoder(structured)

    print(f"Tensor input shape:      {tuple(xyzrgb.shape)}")
    print(f"Tensor output shape:     {tuple(features_a.shape)}")
    print(f"Structured output shape: {tuple(features_b.shape)}")
