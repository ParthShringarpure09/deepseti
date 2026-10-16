"""Plain (non-augmented) frame dataset for ConvAE reconstruction training.

Wraps an in-memory ``(N, 16, 4096)`` float32 array of already-normalized
frames (see ``deepseti.preprocessing.fine.normalize_fine_frame``) and serves
``(1, 16, 4096)`` tensors. No augmentation, no synthetic injection: this
dataset is only ever handed real normal frames (train split for fitting,
clean validation split for checkpoint selection).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_SHAPE = (16, 4096)


class FrameArrayDataset(Dataset):
    def __init__(self, frames: np.ndarray):
        if frames.ndim != 3 or frames.shape[1:] != FRAME_SHAPE:
            raise ValueError(f"expected frames shaped (N, 16, 4096), got {frames.shape}")
        if not np.isfinite(frames).all():
            raise ValueError("frames contain non-finite values")
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.frames[index]).unsqueeze(0)
