"""CNN encoder for self-supervised spectrogram representation learning."""

from __future__ import annotations

import torch
from torch import nn


class SSLEncoder(nn.Module):
    """
    CNN encoder for DeepSeti spectrogram frames.

    Input
    -----
    Tensor of shape:
        (batch, 1, 16, 4096)

    Output
    ------
    Tensor of shape:
        (batch, embedding_dim)

    The architecture downsamples frequency more aggressively than time
    because DeepSeti frames are highly asymmetric: 16 time bins versus
    4096 frequency bins.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pooled_frequency_bins: int = 32,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooled_frequency_bins = pooled_frequency_bins

        self.features = nn.Sequential(
            # (B, 1, 16, 4096)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 9),
                stride=(1, 2),
                padding=(1, 4),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=32,
            ),
            nn.GELU(),

            # (B, 32, 16, 2048)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 7),
                stride=(1, 2),
                padding=(1, 3),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=64,
            ),
            nn.GELU(),

            # (B, 64, 16, 1024)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 5),
                stride=(2, 2),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=128,
            ),
            nn.GELU(),

            # (B, 128, 8, 512)
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 5),
                stride=(2, 2),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=256,
            ),
            nn.GELU(),
            # (B, 256, 4, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=(1, pooled_frequency_bins)
        )

        self.embedding = nn.Linear(
            256 * pooled_frequency_bins,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode spectrogram frames into fixed-dimensional representations.
        """

        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape "
                "(batch, channel, time, frequency), "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"Expected one input channel, got {x.shape[1]}."
            )

        features = self.features(x)

        pooled = self.pool(features)

        flattened = torch.flatten(
            pooled,
            start_dim=1,
        )

        embedding = self.embedding(flattened)

        return embedding