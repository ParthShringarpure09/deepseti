"""Normalization of Fine (16 x 4096) frames using the frozen DeepSeti rule.

The rule is the existing ``robust_normalize_frame`` (log10, then per-frame
median / MAD, no clipping); this module only adds input/output checks.
"""

import numpy as np

from deepseti.preprocessing.frames import robust_normalize_frame

FRAME_SHAPE = (16, 4096)


def normalize_fine_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"frame shape {frame.shape} != {FRAME_SHAPE}")
    if not np.isfinite(frame).all() or (frame <= 0).any():
        raise ValueError("raw frame has non-finite or non-positive values; log10 undefined")
    out = robust_normalize_frame(frame)
    if out.shape != FRAME_SHAPE or not np.isfinite(out).all():
        raise ValueError("normalized frame is not finite (16, 4096)")
    return out


def normalize_fine_frames(frames: np.ndarray) -> np.ndarray:
    return np.stack([normalize_fine_frame(f) for f in frames])
