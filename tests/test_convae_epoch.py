import numpy as np
import torch
from torch.utils.data import DataLoader

from deepseti.models.conv_autoencoder import ConvAutoencoderV3
from deepseti.training.convae_dataset import FrameArrayDataset
from deepseti.training.convae_epoch import evaluate_convae_epoch, train_convae_epoch


def _tiny_loader(n=8, seed=0):
    frames = np.random.default_rng(seed).normal(size=(n, 16, 4096)).astype(np.float32)
    return DataLoader(FrameArrayDataset(frames), batch_size=4, shuffle=True)


def test_train_convae_epoch_reduces_loss_over_several_steps():
    torch.manual_seed(0)
    model = ConvAutoencoderV3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    loader = _tiny_loader()

    first = train_convae_epoch(model, loader, optimizer, device)["loss"]
    for _ in range(5):
        last = train_convae_epoch(model, loader, optimizer, device)["loss"]

    assert last < first


def test_evaluate_convae_epoch_does_not_change_weights():
    model = ConvAutoencoderV3()
    device = torch.device("cpu")
    loader = _tiny_loader()

    before = [p.clone() for p in model.parameters()]
    evaluate_convae_epoch(model, loader, device)
    after = list(model.parameters())

    assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_evaluate_convae_epoch_returns_finite_loss():
    model = ConvAutoencoderV3()
    device = torch.device("cpu")
    loader = _tiny_loader()
    metrics = evaluate_convae_epoch(model, loader, device)
    assert np.isfinite(metrics["loss"])
