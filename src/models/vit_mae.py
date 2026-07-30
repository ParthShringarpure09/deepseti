import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Chop a (B, 1, 16, 4096) frame into 16x16 patches and embed each into a vector."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


def random_masking(x, mask_ratio: float = 0.75):
    """Keep a random subset of patches. x: (B, N, D).
    Returns kept patches, the mask (1 = hidden), and indices to restore original order."""
    B, N, D = x.shape
    n_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)

    return x_kept, mask, ids_restore

def patchify(imgs, patch_size: int = 16):
    """(B,1,16,4096) -> (B, 256, 256): raw pixels of each patch, as the reconstruction target."""
    B, C, H, W = imgs.shape
    h, w = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, patch_size * patch_size * C)

class ViTMAE(nn.Module):
    def __init__(self, num_patches=256, patch_pixels=256, embed_dim=256,
                 depth=4, num_heads=4, decoder_dim=128, decoder_depth=2, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(patch_size=16, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        dec = nn.TransformerEncoderLayer(decoder_dim, num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec, decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, patch_pixels)

        for p in [self.pos_embed, self.decoder_pos_embed, self.mask_token]:
            nn.init.normal_(p, std=0.02)

    def forward(self, imgs):
        target = patchify(imgs)                              # (B,256,256) the answer key
        x = self.patch_embed(imgs) + self.pos_embed          # (B,256,embed)
        x_kept, mask, ids_restore = random_masking(x, self.mask_ratio)
        latent = self.encoder(x_kept)                        # (B,64,embed)

        z = self.decoder_embed(latent)                       # (B,64,dec)
        B, N = ids_restore.shape
        mask_tokens = self.mask_token.expand(B, N - z.shape[1], -1)
        z = torch.cat([z, mask_tokens], dim=1)               # (B,256,dec)
        z = torch.gather(z, 1, ids_restore.unsqueeze(-1).expand(-1, -1, z.shape[2]))
        z = self.decoder(z + self.decoder_pos_embed)
        pred = self.decoder_pred(z)                          # (B,256,256)

        loss = ((pred - target) ** 2).mean(dim=-1)           # per-patch error
        loss = (loss * mask).sum() / mask.sum()              # average over HIDDEN patches only
        return loss, pred, mask