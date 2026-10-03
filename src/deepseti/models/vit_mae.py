import torch
from torch import nn

def patchify(
    x: torch.Tensor,
    patch_size: tuple[int, int] = (4, 64),
) -> torch.Tensor:
    """
    Convert spectrogram frames into non-overlapping patches.

    Parameters
    ----------
    x:
        Tensor with shape:
        (batch, channels, time, frequency)

    patch_size:
        (patch_time, patch_frequency)

    Returns
    -------
    patches:
        Tensor with shape:
        (batch, num_patches, patch_dim)
    """

    if x.ndim != 4:
        raise ValueError(
            "Expected input shape "
            "(batch, channels, time, frequency)"
        )

    batch, channels, time, frequency = x.shape

    patch_time, patch_frequency = patch_size

    if time % patch_time != 0:
        raise ValueError(
            "Time dimension must be divisible "
            "by patch_time."
        )

    if frequency % patch_frequency != 0:
        raise ValueError(
            "Frequency dimension must be divisible "
            "by patch_frequency."
        )

    num_time_patches = time // patch_time
    num_frequency_patches = (
        frequency // patch_frequency
    )

    x = x.reshape(
        batch,
        channels,
        num_time_patches,
        patch_time,
        num_frequency_patches,
        patch_frequency,
    )

    x = x.permute(
        0,
        2,
        4,
        1,
        3,
        5,
    )

    patches = x.reshape(
        batch,
        num_time_patches
        * num_frequency_patches,
        channels
        * patch_time
        * patch_frequency,
    )

    return patches


def unpatchify(
    patches: torch.Tensor,
    frame_shape: tuple[int, int] = (16, 4096),
    patch_size: tuple[int, int] = (4, 64),
    channels: int = 1,
) -> torch.Tensor:
    """
    Reconstruct spectrogram frames from patches.
    """

    batch, num_patches, patch_dim = patches.shape

    time, frequency = frame_shape

    patch_time, patch_frequency = patch_size

    num_time_patches = time // patch_time
    num_frequency_patches = (
        frequency // patch_frequency
    )

    expected_patches = (
        num_time_patches
        * num_frequency_patches
    )

    expected_patch_dim = (
        channels
        * patch_time
        * patch_frequency
    )

    if num_patches != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} patches, "
            f"got {num_patches}."
        )

    if patch_dim != expected_patch_dim:
        raise ValueError(
            f"Expected patch dimension "
            f"{expected_patch_dim}, "
            f"got {patch_dim}."
        )

    x = patches.reshape(
        batch,
        num_time_patches,
        num_frequency_patches,
        channels,
        patch_time,
        patch_frequency,
    )

    x = x.permute(
        0,
        3,
        1,
        4,
        2,
        5,
    )

    x = x.reshape(
        batch,
        channels,
        time,
        frequency,
    )

    return x

class PatchEmbedding(nn.Module):
    """
    Convert spectrogram patches into transformer tokens.
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (4, 64),
        channels: int = 1,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()

        patch_time, patch_frequency = patch_size

        self.patch_size = patch_size

        self.patch_dim = (
            channels
            * patch_time
            * patch_frequency
        )

        self.projection = nn.Linear(
            self.patch_dim,
            embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        patches = patchify(
            x,
            patch_size=self.patch_size,
        )

        tokens = self.projection(
            patches
        )

        return tokens

class PositionalEmbedding(nn.Module):
    """
    Add learned 2D positional information to patch tokens.

    Position is represented separately along:
    - time
    - frequency
    """

    def __init__(
        self,
        frame_shape: tuple[int, int] = (16, 4096),
        patch_size: tuple[int, int] = (4, 64),
        embed_dim: int = 128,
    ) -> None:
        super().__init__()

        time, frequency = frame_shape
        patch_time, patch_frequency = patch_size

        self.num_time_patches = (
            time // patch_time
        )

        self.num_frequency_patches = (
            frequency // patch_frequency
        )

        self.embed_dim = embed_dim

        self.time_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_time_patches,
                1,
                embed_dim,
            )
        )

        self.frequency_embedding = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.num_frequency_patches,
                embed_dim,
            )
        )

        nn.init.trunc_normal_(
            self.time_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.frequency_embedding,
            std=0.02,
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        batch, num_patches, embed_dim = tokens.shape

        expected_patches = (
            self.num_time_patches
            * self.num_frequency_patches
        )

        if num_patches != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} patches, "
                f"got {num_patches}."
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embed_dim}, "
                f"got {embed_dim}."
            )

        tokens = tokens.reshape(
            batch,
            self.num_time_patches,
            self.num_frequency_patches,
            embed_dim,
        )

        tokens = (
            tokens
            + self.time_embedding
            + self.frequency_embedding
        )

        tokens = tokens.reshape(
            batch,
            expected_patches,
            embed_dim,
        )

        return tokens

def random_masking(
    tokens: torch.Tensor,
    mask_ratio: float = 0.75,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Randomly mask patch tokens independently for each sample.

    Parameters
    ----------
    tokens:
        Tensor of shape:
        (batch, num_patches, embed_dim)

    mask_ratio:
        Fraction of patches to hide.

    Returns
    -------
    visible_tokens:
        Tokens kept for the encoder.

    mask:
        Shape (batch, num_patches).
        0 = visible patch
        1 = masked patch

    ids_restore:
        Indices needed later to restore the
        original patch ordering in the decoder.
    """

    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(
            "mask_ratio must be between 0 and 1."
        )

    batch, num_patches, embed_dim = tokens.shape

    num_visible = int(
        num_patches * (1.0 - mask_ratio)
    )

    # Random value for every patch.
    noise = torch.rand(
        batch,
        num_patches,
        device=tokens.device,
    )

    # Smallest random values become visible.
    ids_shuffle = torch.argsort(
        noise,
        dim=1,
    )

    # Needed later to recover original order.
    ids_restore = torch.argsort(
        ids_shuffle,
        dim=1,
    )

    ids_keep = ids_shuffle[
        :,
        :num_visible,
    ]

    visible_tokens = torch.gather(
        tokens,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(
            -1,
            -1,
            embed_dim,
        ),
    )

    # Start with every patch marked as masked.
    mask = torch.ones(
        batch,
        num_patches,
        device=tokens.device,
    )

    # First num_visible positions are visible.
    mask[:, :num_visible] = 0

    # Restore mask to original patch order.
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )

    return (
        visible_tokens,
        mask,
        ids_restore,
    )

class MAEEncoder(nn.Module):
    """
    Transformer encoder for visible spectrogram patch tokens.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        visible_tokens: torch.Tensor,
    ) -> torch.Tensor:

        encoded_tokens = self.encoder(
            visible_tokens
        )

        encoded_tokens = self.norm(
            encoded_tokens
        )

        return encoded_tokens

class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder for reconstructing
    all spectrogram patches.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int] = (16, 4096),
        patch_size: tuple[int, int] = (4, 64),
        channels: int = 1,
        encoder_dim: int = 128,
        decoder_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 128,
    ) -> None:
        super().__init__()

        time, frequency = frame_shape
        patch_time, patch_frequency = patch_size

        self.num_patches = (
            (time // patch_time)
            * (frequency // patch_frequency)
        )

        self.patch_dim = (
            channels
            * patch_time
            * patch_frequency
        )

        self.decoder_embed = nn.Linear(
            encoder_dim,
            decoder_dim,
        )

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                decoder_dim,
            )
        )

        nn.init.trunc_normal_(
            self.mask_token,
            std=0.02,
        )

        self.position_embedding = PositionalEmbedding(
            frame_shape=frame_shape,
            patch_size=patch_size,
            embed_dim=decoder_dim,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(
            decoder_dim
        )

        self.prediction = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:

        x = self.decoder_embed(
            encoded_tokens
        )

        batch, num_visible, decoder_dim = x.shape

        num_masked = (
            self.num_patches
            - num_visible
        )

        mask_tokens = self.mask_token.expand(
            batch,
            num_masked,
            decoder_dim,
        )

        x = torch.cat(
            [
                x,
                mask_tokens,
            ],
            dim=1,
        )

        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(
                -1,
                -1,
                decoder_dim,
            ),
        )

        x = self.position_embedding(
            x
        )

        x = self.decoder(
            x
        )

        x = self.norm(
            x
        )

        predicted_patches = self.prediction(
            x
        )

        return predicted_patches

class ViTMAE(nn.Module):
    """
    Vision Transformer Masked Autoencoder for radio spectrograms.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int] = (16, 4096),
        patch_size: tuple[int, int] = (4, 64),
        channels: int = 1,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        encoder_mlp_dim: int = 256,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        decoder_mlp_dim: int = 128,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()

        self.frame_shape = frame_shape
        self.patch_size = patch_size
        self.channels = channels
        self.mask_ratio = mask_ratio

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
        )

        self.position_embedding = PositionalEmbedding(
            frame_shape=frame_shape,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.encoder = MAEEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_dim=encoder_mlp_dim,
        )

        self.decoder = MAEDecoder(
            frame_shape=frame_shape,
            patch_size=patch_size,
            channels=channels,
            encoder_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_dim=decoder_mlp_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        target_patches = patchify(
            x,
            patch_size=self.patch_size,
        )

        tokens = self.patch_embedding(
            x
        )

        tokens = self.position_embedding(
            tokens
        )

        visible_tokens, mask, ids_restore = random_masking(
            tokens,
            mask_ratio=self.mask_ratio,
        )

        encoded_tokens = self.encoder(
            visible_tokens
        )

        predicted_patches = self.decoder(
            encoded_tokens,
            ids_restore,
        )

        return (
            predicted_patches,
            target_patches,
            mask,
        )
def masked_reconstruction_loss(
    predicted_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE only over masked patches.
    """

    patch_loss = (
        predicted_patches
        - target_patches
    ) ** 2

    patch_loss = patch_loss.mean(
        dim=-1
    )

    masked_loss = (
        patch_loss
        * mask
    ).sum() / mask.sum()

    return masked_loss