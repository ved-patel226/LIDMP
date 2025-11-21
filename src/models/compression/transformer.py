import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        ctx_len: int,
        cond_len: int,
        embed_dim: int,
        n_heads: int,
        attn_bias: bool,
        use_mask: bool = True,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer("mask", torch.ones(ctx_len, ctx_len), persistent=False)
            self.mask = torch.tril(self.mask).view(1, ctx_len, ctx_len)
            self.mask[:, :cond_len, :cond_len] = 1

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ww = torch.zeros(1, 1, embed_dim)
            for i in range(embed_dim):
                ww[0, 0, i] = i / (embed_dim - 1)
        self.time_mix = nn.Parameter(ww)

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape

        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)
        q = (
            self.query(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)
        v = (
            self.value(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)

        if use_cache:
            present = torch.stack([k, v])

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        if use_cache and layer_past is not None:
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)
        else:
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            if self.use_mask:
                mask = self.mask if T == self.ctx_len else self.mask[:, :T, :T]
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)
        y = y.transpose(0, 1).contiguous().view(T, B, C)

        # output projection
        y = self.proj(y)

        if use_cache:
            return y.transpose(0, 1).contiguous(), present
        else:
            return y.transpose(0, 1).contiguous()


class FFN(nn.Module):
    def __init__(self, embed_dim, mlp_bias):
        super().__init__()
        self.p0 = nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias)
        self.p1 = nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias)

    def forward(self, x):
        x = self.p0(x)
        x = torch.square(torch.relu(x))
        x = self.p1(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        ctx_len: int,
        cond_len: int,
        embed_dim: int,
        n_heads: int,
        mlp_bias: bool,
        attn_bias: bool,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(
            ctx_len=ctx_len,
            cond_len=cond_len,
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_bias=attn_bias,
            use_mask=False,  # No masking for image reconstruction
        )
        self.mlp = FFN(embed_dim=embed_dim, mlp_bias=mlp_bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class UpSample(nn.Module):
    """Upsample block with conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class LatentDecoder(pl.LightningModule):
    """Decode latent space (16x56x56) -> (3x448x448) using Vision Transformer."""

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        embed_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        patch_size: int = 8,
    ) -> None:
        super().__init__()

        # Patch embedding: convert 16x56x56 to sequence of patches
        self.patch_size = patch_size
        self.num_patches = (56 // patch_size) ** 2  # 49 patches for patch_size=8

        # Project latent patches to embedding dimension
        self.patch_embed = nn.Linear(
            latent_channels * patch_size * patch_size, embed_dim
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    ctx_len=self.num_patches,
                    cond_len=0,
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_bias=True,
                    attn_bias=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)

        # Project back to spatial representation
        self.to_spatial = nn.Linear(embed_dim, 128 * patch_size * patch_size)

        # Upsampling path
        self.up_112 = UpSample(128, 128)
        self.up_224 = UpSample(128, 64)
        self.up_448 = UpSample(64, 32)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.Tanh(),
        )

        self.save_hyperparameters()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 16, 56, 56)

        Returns:
            image: (B, 3, 448, 448)
        """
        B, C, H, W = latent.shape

        # Convert to patches: (B, 16, 56, 56) -> (B, num_patches, C*patch_size*patch_size)
        patches = latent.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, self.num_patches, -1)

        # Embed patches
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Project back to spatial
        x = self.to_spatial(x)  # (B, num_patches, 128*patch_size*patch_size)

        # Reshape to spatial: (B, num_patches, 128*8*8) -> (B, 128, 56, 56)
        patches_per_side = int(math.sqrt(self.num_patches))
        x = x.view(
            B, patches_per_side, patches_per_side, 128, self.patch_size, self.patch_size
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            B,
            128,
            patches_per_side * self.patch_size,
            patches_per_side * self.patch_size,
        )

        # Upsample to final resolution
        x = self.up_112(x)  # -> (B, 128, 112, 112)
        x = self.up_224(x)  # -> (B, 64, 224, 224)
        x = self.up_448(x)  # -> (B, 32, 448, 448)

        image = self.final_conv(x)  # -> (B, 3, 448, 448)

        return image

    def log_images(
        self,
        batch,
        num_samples: int = 4,
    ):
        """Log sample predictions during validation."""
        latent, target_image = batch

        with torch.no_grad():
            predicted = self(latent[:num_samples])

        return {
            "target": target_image[:num_samples],
            "predicted": predicted[:num_samples],
        }


if __name__ == "__main__":
    model = LatentDecoder(
        latent_channels=16,
        output_channels=3,
        embed_dim=256,
        n_heads=4,
        n_layers=2,
        patch_size=1,
    )

    latent = torch.randn(4, 16, 56, 56)
    image = model(latent)

    print(f"Input shape: {latent.shape}")
    print(f"Output shape: {image.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert image.shape == (4, 3, 448, 448), "Output shape mismatch!"
