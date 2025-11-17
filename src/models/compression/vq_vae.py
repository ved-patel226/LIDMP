import torch
import torch.nn as nn
import pytorch_lightning as pl
import lpips
import torch.nn.init as init
from vector_quantize_pytorch import VectorQuantize

try:
    from .residual import ResidualStack
    from .cbam import CBAM
except ImportError:
    from residual import ResidualStack
    from cbam import CBAM


class Encoder(nn.Module):
    """
    Dynamic encoder with configurable downsampling stages and residual stack.

    Inputs:
    - in_dim : the input dimension (channels)
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - embedding_dim : output channels for quantization
    - num_downsamples : number of downsampling blocks (default: 3)
    - initial_channels : number of channels after initial conv (default: 64)
    """

    def __init__(
        self,
        in_dim,
        n_res_layers,
        res_h_dim,
        embedding_dim,
        num_downsamples=3,
        initial_channels=64,
    ):
        super(Encoder, self).__init__()

        self.num_downsamples = num_downsamples

        self.initial = nn.Sequential(
            nn.Conv2d(in_dim, initial_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.LeakyReLU(),
        )

        self.downsample_blocks = nn.ModuleList()
        current_channels = initial_channels
        for i in range(num_downsamples):
            next_channels = max(initial_channels // (2 ** (i + 1)), embedding_dim)
            self.downsample_blocks.append(
                self._downsample_block(current_channels, next_channels)
            )
            current_channels = next_channels

        self.residual_stack = ResidualStack(
            in_dim=current_channels,
            h_dim=current_channels,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
        )

        self.compress = nn.Sequential(
            nn.Conv2d(
                current_channels,
                max(current_channels // 2, embedding_dim),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(max(current_channels // 2, embedding_dim)),
            nn.LeakyReLU(),
            nn.Conv2d(
                max(current_channels // 2, embedding_dim),
                embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # self._init_weights()

    def _downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            # Pre-processing block 1
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            CBAM(in_channels),
            # Pre-processing block 2
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            CBAM(in_channels),
            # Downsample
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            # Post-processing block 1
            CBAM(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            CBAM(out_channels),
            # Post-processing block 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            CBAM(out_channels),
        )

    def forward(self, x):
        x = self.initial(x)

        for downsample in self.downsample_blocks:
            x = downsample(x)

        x = self.residual_stack(x)
        x = self.compress(x)
        return x

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        n_res_layers,
        res_h_dim,
        out_channels=3,
        num_upsamples=3,
        initial_channels=64,
    ):
        super().__init__()
        self.num_upsamples = num_upsamples

        self.expand = nn.Sequential(
            nn.Conv2d(in_dim, initial_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(initial_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(initial_channels // 2, initial_channels, 3, 1, 1),
            nn.BatchNorm2d(initial_channels),
            nn.LeakyReLU(),
        )

        self.upsample_blocks = nn.ModuleList()
        current_channels = initial_channels
        for i in range(num_upsamples):
            next_channels = current_channels // 2
            self.upsample_blocks.append(
                self._upsample_block(current_channels, next_channels)
            )
            current_channels = next_channels

        self.residual_stack = ResidualStack(
            in_dim=current_channels,
            h_dim=current_channels,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
        )

        self.final = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 3, 1, 1), nn.Tanh()
        )

        # self._init_weights()

    def forward(self, x):
        x = self.expand(x)
        for upsample in self.upsample_blocks:
            x = upsample(x)

        x = self.residual_stack(x)
        x = self.final(x)
        return x

    def _upsample_block(self, in_channels, out_channels, upscale_factor=2):
        return nn.Sequential(
            # Pre-processing block 1
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            CBAM(in_channels),
            # Pre-processing block 2
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            CBAM(in_channels),
            # Upsample
            nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.BatchNorm2d(out_channels * (upscale_factor**2)),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor),
            # Post-processing block 1
            CBAM(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            CBAM(out_channels),
            # Post-processing block 2
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            CBAM(out_channels),
        )

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class VQVAE(pl.LightningModule):
    """
    Inputs:
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - n_embeddings : number of embeddings in the codebook
    - embedding_dim : dimension of each embedding vector
    - beta : commitment loss weight
    - lr : learning rate for optimizer
    - lambda_lpips : weight for LPIPS loss
    - lambda_recon : weight for reconstruction loss
    """

    def __init__(
        self,
        res_h_dim,
        h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta=1.0,
        lr=1e-3,
        lambda_lpips=1.0,
        lambda_recon=1.0,
        num_downsamples=3,
        initial_channels=64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            3,
            n_res_layers,
            res_h_dim,
            embedding_dim=embedding_dim,
            num_downsamples=num_downsamples,
            initial_channels=initial_channels,
        )

        self.vector_quantization = VectorQuantize(
            codebook_size=n_embeddings,
            dim=embedding_dim,
            decay=0.99,
            eps=1e-5,
            commitment_weight=beta,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
            sample_codebook_temp=0.0,
        )

        self.decoder = Decoder(
            embedding_dim,
            n_res_layers,
            res_h_dim,
            num_upsamples=num_downsamples,
            initial_channels=initial_channels,
        )

        self.lr = lr
        self.lambda_lpips = lambda_lpips
        self.lambda_recon = lambda_recon

        self.lpips_loss = lpips.LPIPS(net="vgg")

        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        print("VQVAE __init__ complete.")

    def forward(self, x):
        z_e = self.encoder(x)
        b, c, h, w = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(b, h * w, c)
        quantized, indices, commitment_loss = self.vector_quantization(z_flat)

        # Reshape quantized embeddings back to spatial format for Decoder
        quantized_spatial = quantized.view(b, h, w, c).permute(
            0, 3, 1, 2
        )  # [B, C, H, W]
        x_hat = self.decoder(quantized_spatial)

        return commitment_loss.mean(), x_hat, indices

    def _calculate_loss(
        self, x_hat, x, embedding_loss, perplexity=None, log_name="train"
    ):
        # Normalize LPIPS loss to prevent gradient explosion
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = self.lambda_lpips * lpips_loss_val + embedding_loss

        opt = self.optimizers() if hasattr(self, "optimizers") else None
        lr = opt.param_groups[0]["lr"] if opt else self.lr
        self.log("lr", lr, prog_bar=True)

        self.log(f"{log_name}_loss", loss, prog_bar=True)
        self.log(f"{log_name}_embedding_loss", embedding_loss)
        self.log(f"{log_name}_lpips_loss", lpips_loss_val)

        if perplexity is not None:
            self.log(f"{log_name}_perplexity", perplexity)
        return loss

    def training_step(self, batch, _batch_idx):
        x, _ = batch
        embedding_loss, x_hat, _ = self(x)

        loss = self._calculate_loss(x_hat, x, embedding_loss, log_name="train")

        return loss

    def validation_step(self, batch, _batch_idx):
        x, _ = batch
        embedding_loss, x_hat, _ = self(x)

        loss = self._calculate_loss(x_hat, x, embedding_loss, log_name="val")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.vector_quantization.parameters()},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main() -> None:
    from torchinfo import summary

    device = "cuda"

    model = VQVAE(
        res_h_dim=512,
        h_dim=256,
        n_res_layers=5,
        n_embeddings=512 * 2,
        embedding_dim=16,
        beta=0.25,
        lr=1e-3,
        num_downsamples=3,
        initial_channels=192,
    ).to(device)

    summary(model, input_size=(1, 3, 448, 448), device=device)

    x = torch.randn(1, 3, 448, 448).to(device)
    with torch.no_grad():
        commitment_loss, x_hat, indices = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_hat.shape)


if __name__ == "__main__":
    main()
