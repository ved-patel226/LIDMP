import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


class ResidualConvBlock(nn.Module):
    """Residual convolution block for better gradient flow."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return x + self.conv(x)


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

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape

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

        # Compute attention
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
        x = F.gelu(x)
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

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, "weight") and module.weight is not None:
                    torch.nn.init.zeros_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DownSample(nn.Module):
    """Downsample block with conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """Upsample block using PixelShuffle."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upscale_factor = 2
        self.conv = nn.Conv2d(
            in_channels, out_channels * (self.upscale_factor**2), 3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class LatentEncoder(pl.LightningModule):
    """Encode image (3x448x448) -> latent space (16x56x56) using Vision Transformer."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 16,
        embed_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        patch_size: int = 8,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (56 // patch_size) ** 2  # 49 patches for patch_size=8

        self.latent_channels = latent_channels

        # Initial conv to process input
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            ResidualConvBlock(32),
        )

        # Downsampling path
        self.down_224 = nn.Sequential(
            DownSample(32, 64),
            ResidualConvBlock(64),
        )
        self.down_112 = nn.Sequential(
            DownSample(64, 128),
            ResidualConvBlock(128),
        )
        self.down_56 = nn.Sequential(
            DownSample(128, 128),
            ResidualConvBlock(128),
        )

        # Project spatial to patches
        self.from_spatial = nn.Linear(128 * patch_size * patch_size, embed_dim)

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

        # Project to latent space
        self.to_latent = nn.Linear(embed_dim, latent_channels * patch_size * patch_size)

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 448, 448)

        Returns:
            latent: (B, 16, 56, 56)
        """
        B = image.shape[0]

        # Initial processing
        x = self.initial_conv(image)  # (B, 32, 448, 448)

        # Downsample
        x = self.down_224(x)  # (B, 64, 224, 224)
        x = self.down_112(x)  # (B, 128, 112, 112)
        x = self.down_56(x)  # (B, 128, 56, 56)

        # Convert to patches: (B, 128, 56, 56) -> (B, num_patches, 128*patch_size*patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            B, 128, -1, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, self.num_patches, -1)

        # Project to embedding space
        x = self.from_spatial(patches)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Project to latent space
        x = self.to_latent(x)  # (B, num_patches, latent_channels*patch_size*patch_size)

        # Reshape to spatial: (B, num_patches, 16*8*8) -> (B, 16, 56, 56)
        patches_per_side = int(math.sqrt(self.num_patches))
        x = x.view(
            B,
            patches_per_side,
            patches_per_side,
            self.latent_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        latent = x.view(
            B,
            self.latent_channels,
            patches_per_side * self.patch_size,
            patches_per_side * self.patch_size,
        )

        return latent

    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Scale down output projection
        with torch.no_grad():
            self.to_latent.weight.mul_(0.1)


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
            nn.LeakyReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._init_weights()
        self.save_hyperparameters()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 16, 56, 56)

        Returns:
            image: (B, 3, 448, 448)
        """
        B, C = latent.shape[0], latent.shape[1]

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

    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.to_spatial.weight.mul_(0.1)

        with torch.no_grad():
            for layer in self.final_conv:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.mul_(0.1)


if __name__ == "__main__":
    # Test encoder
    encoder = LatentEncoder(
        input_channels=3,
        latent_channels=16,
        embed_dim=16,
        n_heads=4,
        n_layers=2,
        patch_size=1,
    )

    # Test decoder
    decoder = LatentDecoder(
        latent_channels=16,
        output_channels=3,
        embed_dim=16,
        n_heads=4,
        n_layers=2,
        patch_size=1,
    )

    image = torch.randn(4, 3, 448, 448)
    latent = encoder(image)
    reconstructed = decoder(latent)

    print(f"Input image shape: {image.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    assert latent.shape == (4, 16, 56, 56), "Latent shape mismatch!"
    assert reconstructed.shape == (4, 3, 448, 448), "Reconstructed shape mismatch!"
