"""Epoch utilities for ConvAE reconstruction training (MSE, optional AMP).

Mirrors the structure of ``deepseti.training.ssl_epoch`` (sample-weighted
mean loss, finite-loss guard) for a plain reconstruction objective.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def train_convae_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = 1.0,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        batch_size = batch.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            reconstructed = model(batch)
            loss = nn.functional.mse_loss(reconstructed, batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite ConvAE training loss: {loss.item()}")

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Training loader produced no samples.")

    return {"loss": total_loss / num_samples}


@torch.no_grad()
def evaluate_convae_epoch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        batch_size = batch.shape[0]

        reconstructed = model(batch)
        loss = nn.functional.mse_loss(reconstructed, batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite ConvAE validation loss: {loss.item()}")

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Validation loader produced no samples.")

    return {"loss": total_loss / num_samples}
