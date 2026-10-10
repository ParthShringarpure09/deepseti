import pytest
import torch

from deepseti.training.vicreg import *


def test_invariance_loss_zero_for_identical_inputs():
    x = torch.randn(8, 256)

    loss = invariance_loss(
        x,
        x,
    )

    assert torch.isclose(
        loss,
        torch.tensor(0.0),
    )


def test_invariance_loss_positive_for_different_inputs():
    a = torch.zeros(8, 256)
    b = torch.ones(8, 256)

    loss = invariance_loss(
        a,
        b,
    )

    assert loss.item() > 0.0


def test_invariance_loss_known_value():
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)

    loss = invariance_loss(
        a,
        b,
    )

    assert torch.isclose(
        loss,
        torch.tensor(1.0),
    )


def test_invariance_loss_rejects_shape_mismatch():
    a = torch.randn(8, 256)
    b = torch.randn(4, 256)

    with pytest.raises(ValueError):
        invariance_loss(
            a,
            b,
        )


def test_invariance_loss_rejects_wrong_rank():
    a = torch.randn(2, 4, 256)
    b = torch.randn(2, 4, 256)

    with pytest.raises(ValueError):
        invariance_loss(
            a,
            b,
        )
from deepseti.training.vicreg import (
    invariance_loss,
    variance_loss,
)


def test_variance_loss_positive_for_collapsed_projection():
    x = torch.zeros(
        8,
        256,
    )

    loss = variance_loss(x)

    assert loss.item() > 0.0


def test_variance_loss_known_collapsed_value():
    x = torch.zeros(
        8,
        4,
    )

    loss = variance_loss(
        x,
        target_std=1.0,
        eps=1e-4,
    )

    expected = torch.tensor(0.99)

    assert torch.isclose(
        loss,
        expected,
        atol=1e-6,
    )


def test_variance_loss_zero_when_spread_is_sufficient():
    x = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
        ]
    )

    loss = variance_loss(
        x,
        target_std=1.0,
    )

    assert torch.isclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_variance_loss_rejects_single_sample():
    x = torch.randn(
        1,
        256,
    )

    with pytest.raises(ValueError):
        variance_loss(x)


def test_variance_loss_rejects_wrong_rank():
    x = torch.randn(
        2,
        4,
        256,
    )

    with pytest.raises(ValueError):
        variance_loss(x)
def test_covariance_loss_zero_for_uncorrelated_dimensions():
    x = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    loss = covariance_loss(x)

    assert torch.isclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_covariance_loss_positive_for_correlated_dimensions():
    x = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    loss = covariance_loss(x)

    assert loss.item() > 0.0


def test_covariance_loss_rejects_single_sample():
    x = torch.randn(
        1,
        256,
    )

    with pytest.raises(ValueError):
        covariance_loss(x)


def test_covariance_loss_rejects_single_dimension():
    x = torch.randn(
        8,
        1,
    )

    with pytest.raises(ValueError):
        covariance_loss(x)


def test_covariance_loss_rejects_wrong_rank():
    x = torch.randn(
        2,
        4,
        256,
    )

    with pytest.raises(ValueError):
        covariance_loss(x)

def test_vicreg_loss_returns_expected_components():
    a = torch.randn(
        8,
        256,
    )

    b = torch.randn(
        8,
        256,
    )

    total, components = vicreg_loss(
        a,
        b,
    )

    assert total.ndim == 0

    assert set(components.keys()) == {
        "invariance",
        "variance",
        "covariance",
    }

    expected = (
        25.0 * components["invariance"]
        + 25.0 * components["variance"]
        + components["covariance"]
    )

    assert torch.allclose(
        total,
        expected,
    )


def test_vicreg_loss_penalizes_collapsed_representation():
    a = torch.zeros(
        8,
        4,
    )

    b = torch.zeros(
        8,
        4,
    )

    total, components = vicreg_loss(
        a,
        b,
    )

    assert torch.isclose(
        components["invariance"],
        torch.tensor(0.0),
    )

    assert components["variance"].item() > 0.0

    assert torch.isclose(
        components["covariance"],
        torch.tensor(0.0),
    )

    assert total.item() > 0.0


def test_vicreg_loss_supports_backpropagation():
    a = torch.randn(
        8,
        256,
        requires_grad=True,
    )

    b = torch.randn(
        8,
        256,
        requires_grad=True,
    )

    total, _ = vicreg_loss(
        a,
        b,
    )

    total.backward()

    assert a.grad is not None
    assert b.grad is not None

    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


def test_vicreg_loss_rejects_negative_weight():
    a = torch.randn(
        8,
        256,
    )

    b = torch.randn(
        8,
        256,
    )

    with pytest.raises(ValueError):
        vicreg_loss(
            a,
            b,
            invariance_weight=-1.0,
        )