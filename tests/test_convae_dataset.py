import numpy as np
import pytest
import torch

from deepseti.training.convae_dataset import FrameArrayDataset


def test_frame_array_dataset_shape_and_value():
    frames = np.random.default_rng(0).normal(size=(5, 16, 4096)).astype(np.float32)
    dataset = FrameArrayDataset(frames)
    assert len(dataset) == 5
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (1, 16, 4096)
    assert np.allclose(sample.squeeze(0).numpy(), frames[0])


def test_frame_array_dataset_rejects_wrong_shape():
    frames = np.zeros((3, 8, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="16, 4096"):
        FrameArrayDataset(frames)


def test_frame_array_dataset_rejects_non_finite():
    frames = np.zeros((3, 16, 4096), dtype=np.float32)
    frames[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        FrameArrayDataset(frames)
