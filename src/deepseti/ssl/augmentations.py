"""Self-supervised augmentations for radio spectrogram frames."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    frame: np.ndarray,
    sigma: float = 0.10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add small zero-mean Gaussian noise to a normalized spectrogram frame.

    Parameters
    ----------
    frame:
        Normalized spectrogram with shape (time, frequency),
        currently expected to be (16, 4096).

    sigma:
        Standard deviation of the Gaussian perturbation.

    rng:
        Optional NumPy random generator. Supplying one allows
        reproducible augmentation.

    Returns
    -------
    np.ndarray
        Augmented frame with the same shape and float32 dtype.
    """

    if frame.ndim != 2:
        raise ValueError(
            f"Expected a 2D spectrogram frame, got shape {frame.shape}."
        )

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=frame.shape,
    ).astype(np.float32)

    augmented = frame.astype(np.float32, copy=False) + noise

    return augmented.astype(np.float32, copy=False)

def make_ssl_views(
    frame: np.ndarray,
    sigma: float = 0.10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two independent augmented views of one normal spectrogram frame.

    Both views originate from the same frame but receive independent
    Gaussian perturbations.

    Parameters
    ----------
    frame:
        Normalized spectrogram of shape (time, frequency).

    sigma:
        Standard deviation of the Gaussian augmentation.

    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two independently augmented views with the same shape as `frame`.
    """

    if rng is None:
        rng = np.random.default_rng()

    view_a = add_gaussian_noise(
        frame,
        sigma=sigma,
        rng=rng,
    )

    view_b = add_gaussian_noise(
        frame,
        sigma=sigma,
        rng=rng,
    )

    return view_a, view_b