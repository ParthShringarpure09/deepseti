"""Dataset utilities for self-supervised spectrogram training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from deepseti.ssl.augmentations import make_ssl_views


class SSLFrameDataset(Dataset):
    """
    Dataset producing two augmented views of each normalized spectrogram frame.

    Parameters
    ----------
    frames:
        Normalized spectrogram frames with shape
        (num_frames, time, frequency).

    sigma:
        Standard deviation used for Gaussian augmentation.

    deterministic:
        If True, each frame always receives the same augmentations.
        This is useful for validation so that validation loss is comparable
        across epochs.

    base_seed:
        Base random seed used when deterministic=True.
    """

    def __init__(
        self,
        frames: np.ndarray,
        sigma: float = 0.10,
        deterministic: bool = False,
        base_seed: int = 42,
    ) -> None:
        if frames.ndim != 3:
            raise ValueError(
                "Expected frames with shape "
                "(num_frames, time, frequency), "
                f"got {frames.shape}."
            )

        if len(frames) == 0:
            raise ValueError(
                "SSLFrameDataset requires at least one frame."
            )

        if sigma < 0:
            raise ValueError(
                "sigma must be non-negative."
            )

        self.frames = frames.astype(
            np.float32,
            copy=False,
        )

        self.sigma = sigma
        self.deterministic = deterministic
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.frames)

    def _make_rng(
        self,
        index: int,
    ) -> np.random.Generator:
        """
        Build the random generator used for augmentation.
        """

        if self.deterministic:
            return np.random.default_rng(
                self.base_seed + index
            )

        # PyTorch controls this RNG independently in each DataLoader worker.
        seed = torch.randint(
            low=0,
            high=2**32 - 1,
            size=(1,),
        ).item()

        return np.random.default_rng(seed)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.frames[index]

        rng = self._make_rng(index)

        view_a, view_b = make_ssl_views(
            frame,
            sigma=self.sigma,
            rng=rng,
        )

        # Add the CNN channel dimension:
        # (16, 4096) -> (1, 16, 4096)
        view_a = torch.from_numpy(
            view_a
        ).unsqueeze(0)

        view_b = torch.from_numpy(
            view_b
        ).unsqueeze(0)

        return view_a, view_b