import numpy as np
import pytest
import torch

from deepseti.ssl.dataset import SSLFrameDataset


def make_frames(
    num_frames: int = 8,
) -> np.ndarray:
    rng = np.random.default_rng(42)

    return rng.normal(
        size=(
            num_frames,
            16,
            4096,
        )
    ).astype(np.float32)


def test_ssl_dataset_length():
    frames = make_frames(
        num_frames=8
    )

    dataset = SSLFrameDataset(
        frames
    )

    assert len(dataset) == 8


def test_ssl_dataset_returns_two_views():
    frames = make_frames()

    dataset = SSLFrameDataset(
        frames,
        sigma=0.10,
    )

    view_a, view_b = dataset[0]

    assert view_a.shape == (
        1,
        16,
        4096,
    )

    assert view_b.shape == (
        1,
        16,
        4096,
    )

    assert view_a.dtype == torch.float32
    assert view_b.dtype == torch.float32

    assert not torch.equal(
        view_a,
        view_b,
    )


def test_deterministic_dataset_repeats_views():
    frames = make_frames()

    dataset = SSLFrameDataset(
        frames,
        sigma=0.10,
        deterministic=True,
        base_seed=42,
    )

    first_a, first_b = dataset[3]
    second_a, second_b = dataset[3]

    assert torch.equal(
        first_a,
        second_a,
    )

    assert torch.equal(
        first_b,
        second_b,
    )


def test_stochastic_dataset_changes_views():
    frames = make_frames()

    dataset = SSLFrameDataset(
        frames,
        sigma=0.10,
        deterministic=False,
    )

    first_a, first_b = dataset[3]
    second_a, second_b = dataset[3]

    assert not torch.equal(
        first_a,
        second_a,
    )

    assert not torch.equal(
        first_b,
        second_b,
    )


def test_ssl_dataset_rejects_wrong_shape():
    frames = np.zeros(
        (16, 4096),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        SSLFrameDataset(frames)


def test_ssl_dataset_rejects_empty_frames():
    frames = np.empty(
        (0, 16, 4096),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        SSLFrameDataset(frames)