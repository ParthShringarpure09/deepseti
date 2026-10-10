import pytest
import torch

from deepseti.models.ssl_encoder import SSLEncoder
from deepseti.models.ssl_projection import SSLProjectionHead


def test_ssl_encoder_output_shape():
    model = SSLEncoder(
        embedding_dim=256,
        pooled_frequency_bins=32,
    )

    x = torch.randn(
        2,
        1,
        16,
        4096,
    )

    with torch.no_grad():
        z = model(x)

    assert z.shape == (2, 256)


def test_ssl_encoder_rejects_wrong_rank():
    model = SSLEncoder()

    x = torch.randn(
        16,
        4096,
    )

    with pytest.raises(ValueError):
        model(x)


def test_ssl_encoder_rejects_wrong_channels():
    model = SSLEncoder()

    x = torch.randn(
        2,
        2,
        16,
        4096,
    )

    with pytest.raises(ValueError):
        model(x)


def test_projection_head_output_shape():
    projector = SSLProjectionHead(
        input_dim=256,
        hidden_dim=512,
        output_dim=256,
    )

    z = torch.randn(
        4,
        256,
    )

    with torch.no_grad():
        h = projector(z)

    assert h.shape == (4, 256)


def test_projection_head_rejects_wrong_rank():
    projector = SSLProjectionHead()

    z = torch.randn(
        2,
        4,
        256,
    )

    with pytest.raises(ValueError):
        projector(z)