from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ActivationName = Literal["relu", "gelu"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPTranslationHead(nn.Module):
    """
    MLP that projects 3D point features into CLIP text embedding space.

    Default contract:
    - input:  (..., 256)
    - output: (..., 512)

    It supports both point-wise tensors `(N, D)` and batched point clouds
    `(B, N, D)`.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: int | Sequence[int] = (512, 512),
        output_dim: int = 512,
        dropout: float = 0.1,
        activation: ActivationName = "gelu",
        normalize_output: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            hidden_dims = list(hidden_dims)

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("`dropout` must be in the range [0, 1).")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.normalize_output = normalize_output

        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []

        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = index == len(dims) - 2
            if not is_last:
                layers.append(_make_activation(activation))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        if not torch.is_tensor(features):
            raise TypeError("`features` must be a torch tensor.")

        if features.ndim not in (2, 3):
            raise ValueError(
                "`features` must have shape (N, D) or (B, N, D); "
                f"got {tuple(features.shape)}."
            )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension must be {self.input_dim}, got {features.shape[-1]}."
            )

        output = self.network(features.float())
        if self.normalize_output:
            output = F.normalize(output, dim=-1)
        return output


if __name__ == "__main__":
    model = MLPTranslationHead()

    single = torch.randn(1024, 256)
    batch = torch.randn(2, 1024, 256)

    out_single = model(single)
    out_batch = model(batch)

    print(f"Input single:  {tuple(single.shape)}")
    print(f"Output single: {tuple(out_single.shape)}")
    print(f"Input batch:   {tuple(batch.shape)}")
    print(f"Output batch:  {tuple(out_batch.shape)}")
