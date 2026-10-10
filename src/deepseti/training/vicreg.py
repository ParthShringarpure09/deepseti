"""VICReg loss components for self-supervised representation learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def invariance_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage two augmented views of the same frame to have similar projections.

    Parameters
    ----------
    projection_a:
        Tensor of shape (batch, projection_dim).

    projection_b:
        Tensor of shape (batch, projection_dim).

    Returns
    -------
    torch.Tensor
        Scalar mean-squared error between the two projections.
    """

    if projection_a.shape != projection_b.shape:
        raise ValueError(
            "projection_a and projection_b must have the same shape, "
            f"got {tuple(projection_a.shape)} and "
            f"{tuple(projection_b.shape)}."
        )

    if projection_a.ndim != 2:
        raise ValueError(
            "Expected projection tensors with shape "
            "(batch, projection_dim)."
        )

    return F.mse_loss(
        projection_a,
        projection_b,
    )

def variance_loss(
    projection: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Prevent representation collapse by encouraging each projection
    dimension to have sufficient variation across the batch.

    Parameters
    ----------
    projection:
        Tensor of shape (batch, projection_dim).

    target_std:
        Minimum desired standard deviation for each dimension.

    eps:
        Small numerical stability constant added before the square root.

    Returns
    -------
    torch.Tensor
        Scalar variance regularization loss.
    """

    if projection.ndim != 2:
        raise ValueError(
            "Expected projection tensor with shape "
            "(batch, projection_dim)."
        )

    if projection.shape[0] < 2:
        raise ValueError(
            "Variance loss requires at least two samples in the batch."
        )

    if target_std <= 0:
        raise ValueError(
            "target_std must be positive."
        )

    if eps <= 0:
        raise ValueError(
            "eps must be positive."
        )

    variance = projection.var(
        dim=0,
        unbiased=True,
    )

    std = torch.sqrt(
        variance + eps
    )

    penalty = F.relu(
        target_std - std
    )

    return penalty.mean()

def covariance_loss(
    projection: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage different projection dimensions to carry non-redundant information.

    Parameters
    ----------
    projection:
        Tensor of shape (batch, projection_dim).

    Returns
    -------
    torch.Tensor
        Scalar covariance regularization loss.
    """

    if projection.ndim != 2:
        raise ValueError(
            "Expected projection tensor with shape "
            "(batch, projection_dim)."
        )

    batch_size, projection_dim = projection.shape

    if batch_size < 2:
        raise ValueError(
            "Covariance loss requires at least two samples in the batch."
        )

    if projection_dim < 2:
        raise ValueError(
            "Covariance loss requires at least two projection dimensions."
        )

    centered = projection - projection.mean(
        dim=0,
        keepdim=True,
    )

    covariance = (
        centered.T @ centered
    ) / (batch_size - 1)

    off_diagonal = covariance.flatten()[:-1].view(
        projection_dim - 1,
        projection_dim + 1,
    )[:, 1:].flatten()

    return off_diagonal.pow(2).sum() / projection_dim

def covariance_loss(
    projection: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage different projection dimensions to contain
    non-redundant information.

    Parameters
    ----------
    projection:
        Tensor of shape (batch, projection_dim).

    Returns
    -------
    torch.Tensor
        Scalar covariance regularization loss.
    """

    if projection.ndim != 2:
        raise ValueError(
            "Expected projection tensor with shape "
            "(batch, projection_dim)."
        )

    batch_size, projection_dim = projection.shape

    if batch_size < 2:
        raise ValueError(
            "Covariance loss requires at least two samples in the batch."
        )

    if projection_dim < 2:
        raise ValueError(
            "Covariance loss requires at least two projection dimensions."
        )

    centered = projection - projection.mean(
        dim=0,
        keepdim=True,
    )

    covariance = (
        centered.T @ centered
    ) / (batch_size - 1)

    off_diagonal = covariance.flatten()[:-1].view(
        projection_dim - 1,
        projection_dim + 1,
    )[:, 1:].flatten()

    return (
        off_diagonal.pow(2).sum()
        / projection_dim
    )
def vicreg_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the complete VICReg objective.

    The loss contains three components:

    - invariance: make two views of the same frame similar
    - variance: prevent representation collapse
    - covariance: reduce redundancy between dimensions

    Returns
    -------
    total_loss:
        Weighted scalar VICReg loss.

    components:
        Dictionary containing the unweighted component losses.
    """

    if invariance_weight < 0:
        raise ValueError(
            "invariance_weight must be non-negative."
        )

    if variance_weight < 0:
        raise ValueError(
            "variance_weight must be non-negative."
        )

    if covariance_weight < 0:
        raise ValueError(
            "covariance_weight must be non-negative."
        )

    inv_loss = invariance_loss(
        projection_a,
        projection_b,
    )

    var_loss_a = variance_loss(
        projection_a,
        target_std=target_std,
        eps=eps,
    )

    var_loss_b = variance_loss(
        projection_b,
        target_std=target_std,
        eps=eps,
    )

    var_loss = 0.5 * (
        var_loss_a + var_loss_b
    )

    cov_loss_a = covariance_loss(
        projection_a
    )

    cov_loss_b = covariance_loss(
        projection_b
    )

    cov_loss = (
        cov_loss_a + cov_loss_b
    )

    total_loss = (
        invariance_weight * inv_loss
        + variance_weight * var_loss
        + covariance_weight * cov_loss
    )

    components = {
        "invariance": inv_loss,
        "variance": var_loss,
        "covariance": cov_loss,
    }

    return total_loss, components