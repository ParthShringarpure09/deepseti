import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Z-score a single frame: subtract its mean, divide by its std."""
    frame = frame.astype(np.float32)
    return (frame - frame.mean()) / (frame.std() + 1e-8)


class SpectrogramDataset(Dataset):
    """Loads .npy spectrogram frames, normalized. Accepts directories or an explicit file list."""

    def __init__(self, dirs: list[str] = None, files: list = None):
        if files is not None:
            self.files = list(files)
        else:
            self.files = []
            for d in dirs:
                self.files.extend(sorted(Path(d).glob("*.npy")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frame = np.load(self.files[idx])
        frame = normalize_frame(frame)
        return torch.from_numpy(frame).unsqueeze(0)

import random

def split_files(files: list, ratios=(0.8, 0.1, 0.1), seed: int = 42):
    """Deterministically split a list of files into train/val/test. No overlap."""
    files = sorted(files)                     # stable order
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    # hard assertion: no file in more than one split
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    return train, val, test