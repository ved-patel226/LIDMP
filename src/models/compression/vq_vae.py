# NOTE -  From: https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# Modified for current project

import torch
import torch.nn as nn
import pytorch_lightning as pl
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure


try:
    from .quantizer import VectorQuantizer
    from .residual import ResidualStack
except ImportError:
    from quantizer import VectorQuantizer
    from residual import ResidualStack


class Encoder(nn.Module):
    """
    Dynamic encoder with configurable downsampling stages.

    Inputs:
    - in_dim : the input dimension (channels)
    - h_dim : the hidden layer dimension (base dimension)
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - embedding_dim : output channels for quantization
    - num_downsamples : number of downsampling blocks (default: 3)
    - initial_channels : number of channels after initial conv (default: 64)
    """

    def __init__(
        self,
        in_dim,
        h_dim,
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

        # Create dynamic downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        current_channels = initial_channels

        for i in range(num_downsamples):
            self.downsample_blocks.append(
                self._downsample_block(current_channels, current_channels)
            )

        self.residual_stack = ResidualStack(
            in_dim=current_channels,
            h_dim=current_channels,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
        )

        self.compress = nn.Sequential(
            nn.Conv2d(
                current_channels,
                current_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(current_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(
                current_channels // 2, embedding_dim, kernel_size=3, stride=1, padding=1
            ),
        )

    def _downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.initial(x)

        # Apply all downsampling blocks dynamically
        for downsample in self.downsample_blocks:
            x = downsample(x)

        x = self.residual_stack(x)
        x = self.compress(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class Decoder(nn.Module):
    """
    Dynamic decoder that mirrors the encoder architecture.

    Inputs:
    - in_dim : the input dimension (embedding_dim)
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - out_channels : output image channels (default 3 for RGB)
    - num_upsamples : number of upsampling blocks (should match encoder's num_downsamples)
    - initial_channels : number of channels to expand to (should match encoder's initial_channels)
    """

    def __init__(
        self,
        in_dim,
        h_dim,
        n_res_layers,
        res_h_dim,
        out_channels=3,
        num_upsamples=3,
        initial_channels=64,
    ):
        super(Decoder, self).__init__()

        self.num_upsamples = num_upsamples

        self.expand = nn.Sequential(
            nn.Conv2d(
                in_dim, initial_channels // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(initial_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(
                initial_channels // 2,
                initial_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(initial_channels),
            nn.LeakyReLU(),
        )

        self.residual_stack = ResidualStack(
            in_dim=initial_channels,
            h_dim=initial_channels,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
        )

        # Create dynamic upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        current_channels = initial_channels

        for i in range(num_upsamples):
            self.upsample_blocks.append(
                self._upsample_block(current_channels, current_channels)
            )

        self.final = nn.Sequential(
            nn.Conv2d(
                current_channels, current_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(current_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                current_channels,
                current_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(current_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(
                current_channels // 2, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.Tanh(),
        )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.residual_stack(x)

        # Apply all upsampling blocks dynamically
        for upsample in self.upsample_blocks:
            x = upsample(x)

        x = self.final(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class VQVAE(pl.LightningModule):
    """
    Inputs:
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - n_embeddings : number of embeddings in the codebook
    - embedding_dim : dimension of each embedding vector
    - beta : commitment loss weight
    - device : device to run the model on
    - lr : learning rate for optimizer
    - lambda_lpips : weight for LPIPS loss
    - lambda_recon : weight for reconstruction loss
    """

    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        device="cuda",
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
            h_dim,
            n_res_layers,
            res_h_dim,
            embedding_dim=embedding_dim,
            num_downsamples=num_downsamples,
            initial_channels=initial_channels,
        )

        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device=device
        )
        self.decoder = Decoder(
            embedding_dim,
            h_dim,
            n_res_layers,
            res_h_dim,
            num_upsamples=num_downsamples,
            initial_channels=initial_channels,
        )
        self.lr = lr
        self.lambda_lpips = lambda_lpips
        self.lambda_recon = lambda_recon

        # self.criterion = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips_loss = lpips.LPIPS(net="squeeze")

        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, x):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity

    def _calculate_loss(self, x_hat, x, embedding_loss, perplexity, log_name="train"):
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = lpips_loss_val + embedding_loss

        self.log(f"{log_name}_loss", loss, prog_bar=True)
        # self.log(f"{log_name}_recon_loss", recon_loss)
        # self.log(f"{log_name}_lpips_loss", lpips_loss_val)
        self.log(f"{log_name}_embedding_loss", embedding_loss)
        self.log(f"{log_name}_perplexity", perplexity)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)

        loss = self._calculate_loss(
            x_hat, x, embedding_loss, perplexity, log_name="train"
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)

        loss = self._calculate_loss(
            x_hat, x, embedding_loss, perplexity, log_name="val"
        )

        return loss

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
    model = VQVAE(
        h_dim=56,
        res_h_dim=128,
        n_res_layers=5,
        n_embeddings=2056,
        embedding_dim=1,
        beta=0.25,
        lr=1e-3,
        num_downsamples=3,
        initial_channels=128,
    ).to("cuda")

    from torchinfo import summary

    summary(model, (1, 3, 448, 448), device="cuda")


if __name__ == "__main__":
    main()
