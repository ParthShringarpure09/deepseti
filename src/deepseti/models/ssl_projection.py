"""Projection head for self-supervised representation learning."""

from __future__ import annotations

import torch
from torch import nn


class SSLProjectionHead(nn.Module):
    """
    MLP projection head used only during self-supervised training.

    The encoder produces the representation z used for downstream
    anomaly detection. This projection head maps z into a temporary
    space h on which the VICReg loss is applied.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(
                "Expected embeddings with shape "
                "(batch, embedding_dim), "
                f"got {tuple(z.shape)}."
            )

        return self.net(z)