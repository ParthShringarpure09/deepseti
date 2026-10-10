"""Epoch utilities for DeepSeti self-supervised VICReg training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from deepseti.training.vicreg import vicreg_loss


def train_ssl_epoch(
    encoder: nn.Module,
    projector: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = 1.0,
    vicreg_kwargs: dict | None = None,
) -> dict[str, float]:
    """
    Train the SSL model for one epoch.

    Returns sample-weighted mean VICReg losses.
    """

    encoder.train()
    projector.train()

    if vicreg_kwargs is None:
        vicreg_kwargs = {}

    totals = {
        "loss": 0.0,
        "invariance": 0.0,
        "variance": 0.0,
        "covariance": 0.0,
    }

    num_samples = 0

    parameters = (
        list(encoder.parameters())
        + list(projector.parameters())
    )

    for view_a, view_b in loader:
        view_a = view_a.to(
            device,
            non_blocking=True,
        )

        view_b = view_b.to(
            device,
            non_blocking=True,
        )

        batch_size = view_a.shape[0]

        optimizer.zero_grad(
            set_to_none=True,
        )

        z_a = encoder(view_a)
        z_b = encoder(view_b)

        h_a = projector(z_a)
        h_b = projector(z_b)

        loss, components = vicreg_loss(
            h_a,
            h_b,
            **vicreg_kwargs,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite VICReg training loss: "
                f"{loss.item()}"
            )

        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        totals["loss"] += (
            loss.item()
            * batch_size
        )

        totals["invariance"] += (
            components["invariance"].item()
            * batch_size
        )

        totals["variance"] += (
            components["variance"].item()
            * batch_size
        )

        totals["covariance"] += (
            components["covariance"].item()
            * batch_size
        )

        num_samples += batch_size

    if num_samples == 0:
        raise ValueError(
            "Training loader produced no samples."
        )

    return {
        key: value / num_samples
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate_ssl_epoch(
    encoder: nn.Module,
    projector: nn.Module,
    loader: Iterable,
    device: torch.device,
    vicreg_kwargs: dict | None = None,
) -> dict[str, float]:
    """
    Evaluate VICReg on the normal-only validation set.
    """

    encoder.eval()
    projector.eval()

    if vicreg_kwargs is None:
        vicreg_kwargs = {}

    totals = {
        "loss": 0.0,
        "invariance": 0.0,
        "variance": 0.0,
        "covariance": 0.0,
    }

    num_samples = 0

    for view_a, view_b in loader:
        view_a = view_a.to(
            device,
            non_blocking=True,
        )

        view_b = view_b.to(
            device,
            non_blocking=True,
        )

        batch_size = view_a.shape[0]

        z_a = encoder(view_a)
        z_b = encoder(view_b)

        h_a = projector(z_a)
        h_b = projector(z_b)

        loss, components = vicreg_loss(
            h_a,
            h_b,
            **vicreg_kwargs,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite VICReg validation loss: "
                f"{loss.item()}"
            )

        totals["loss"] += (
            loss.item()
            * batch_size
        )

        totals["invariance"] += (
            components["invariance"].item()
            * batch_size
        )

        totals["variance"] += (
            components["variance"].item()
            * batch_size
        )

        totals["covariance"] += (
            components["covariance"].item()
            * batch_size
        )

        num_samples += batch_size

    if num_samples == 0:
        raise ValueError(
            "Validation loader produced no samples."
        )

    return {
        key: value / num_samples
        for key, value in totals.items()
    }