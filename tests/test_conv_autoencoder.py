import torch

from deepseti.models.conv_autoencoder import ConvAutoencoderV3


def test_conv_autoencoder_output_shape():
    model = ConvAutoencoderV3()

    x = torch.randn(2, 1, 16, 4096)

    with torch.no_grad():
        reconstructed = model(x)

    assert reconstructed.shape == x.shape


def test_conv_autoencoder_output_is_finite():
    model = ConvAutoencoderV3()

    x = torch.randn(2, 1, 16, 4096)

    with torch.no_grad():
        reconstructed = model(x)

    assert torch.isfinite(reconstructed).all()