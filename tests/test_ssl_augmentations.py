import numpy as np
import pytest

from deepseti.ssl.augmentations import (
    add_gaussian_noise,
    make_ssl_views,
)


def test_add_gaussian_noise_preserves_shape_and_dtype():
    frame = np.zeros((16, 4096), dtype=np.float32)

    rng = np.random.default_rng(42)

    augmented = add_gaussian_noise(
        frame,
        sigma=0.10,
        rng=rng,
    )

    assert augmented.shape == frame.shape
    assert augmented.dtype == np.float32


def test_add_gaussian_noise_is_reproducible():
    frame = np.zeros((16, 4096), dtype=np.float32)

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    augmented_a = add_gaussian_noise(
        frame,
        sigma=0.10,
        rng=rng_a,
    )

    augmented_b = add_gaussian_noise(
        frame,
        sigma=0.10,
        rng=rng_b,
    )

    assert np.array_equal(
        augmented_a,
        augmented_b,
    )


def test_make_ssl_views_are_independent():
    frame = np.zeros((16, 4096), dtype=np.float32)

    rng = np.random.default_rng(42)

    view_a, view_b = make_ssl_views(
        frame,
        sigma=0.10,
        rng=rng,
    )

    assert view_a.shape == frame.shape
    assert view_b.shape == frame.shape

    assert not np.array_equal(view_a, frame)
    assert not np.array_equal(view_b, frame)
    assert not np.array_equal(view_a, view_b)


def test_negative_sigma_raises():
    frame = np.zeros((16, 4096), dtype=np.float32)

    with pytest.raises(ValueError):
        add_gaussian_noise(
            frame,
            sigma=-0.10,
        )