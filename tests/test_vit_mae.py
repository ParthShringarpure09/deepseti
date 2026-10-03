import torch

from deepseti.models.vit_mae import (
    ViTMAE,
    masked_reconstruction_loss,
    patchify,
    random_masking,
    unpatchify,
)


def test_patchify_round_trip():
    x = torch.randn(
        2,
        1,
        16,
        4096,
    )

    patches = patchify(
        x,
        patch_size=(4, 64),
    )

    reconstructed = unpatchify(
        patches,
        frame_shape=(16, 4096),
        patch_size=(4, 64),
    )

    assert patches.shape == (
        2,
        256,
        256,
    )

    assert torch.allclose(
        x,
        reconstructed,
    )


def test_random_masking_shape():
    tokens = torch.randn(
        2,
        256,
        128,
    )

    visible, mask, ids_restore = random_masking(
        tokens,
        mask_ratio=0.75,
    )

    assert visible.shape == (
        2,
        64,
        128,
    )

    assert mask.shape == (
        2,
        256,
    )

    assert ids_restore.shape == (
        2,
        256,
    )

    assert mask[0].sum().item() == 192


def test_vit_mae_forward_shapes():
    model = ViTMAE(
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        encoder_mlp_dim=64,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        decoder_mlp_dim=32,
        mask_ratio=0.75,
    )

    x = torch.randn(
        2,
        1,
        16,
        4096,
    )

    predicted, target, mask = model(x)

    assert predicted.shape == (
        2,
        256,
        256,
    )

    assert target.shape == (
        2,
        256,
        256,
    )

    assert mask.shape == (
        2,
        256,
    )


def test_masked_reconstruction_loss_is_finite():
    model = ViTMAE(
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        encoder_mlp_dim=64,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        decoder_mlp_dim=32,
        mask_ratio=0.75,
    )

    x = torch.randn(
        2,
        1,
        16,
        4096,
    )

    predicted, target, mask = model(x)

    loss = masked_reconstruction_loss(
        predicted,
        target,
        mask,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0