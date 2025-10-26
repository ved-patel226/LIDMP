# NOTE -  From: https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# Modified for current project

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

try:
    from .quantizer import VectorQuantizer
    from .residual import ResidualStack
except ImportError:
    from quantizer import VectorQuantizer
    from residual import ResidualStack


class Encoder(nn.Module):
    """
    Balanced encoder for image reconstruction.
    16x spatial downsampling with 4 stages.

    Inputs:
    - in_dim : the input dimension (channels)
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            # Stage 1: in_dim -> h_dim // 2 (downsample)
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 2: h_dim // 2 -> h_dim (downsample)
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 3: h_dim -> h_dim (downsample)
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 4: h_dim -> h_dim (downsample)
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Refinement with residuals (no downsampling)
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    """
    Balanced decoder for image reconstruction.
    16x spatial upsampling mirroring the encoder.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - out_channels : output image channels (default 3 for RGB)
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channels=3):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            # Refinement with residuals
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            # Stage 1: h_dim -> h_dim (upsample)
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 2: h_dim -> h_dim (upsample)
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 3: h_dim -> h_dim // 2 (upsample)
            nn.ConvTranspose2d(
                h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.BatchNorm2d(h_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 4: h_dim // 2 -> out_channels (upsample)
            nn.ConvTranspose2d(
                h_dim // 2, out_channels, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.Tanh(),  # Output activation for normalized images [-1, 1]
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(pl.LightningModule):
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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device=device
        )
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.lr = lr
        self.criterion = nn.MSELoss()  # Reconstruction loss

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity

    def training_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)
        recon_loss = self.criterion(x_hat, x)
        loss = recon_loss + embedding_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_embedding_loss", embedding_loss)
        self.log("train_perplexity", perplexity)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)
        recon_loss = self.criterion(x_hat, x)
        loss = recon_loss + embedding_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_embedding_loss", embedding_loss)
        self.log("val_perplexity", perplexity)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main() -> None:
    vq_vae = VQVAE(
        h_dim=128,
        res_h_dim=128,
        n_res_layers=3,
        n_embeddings=512,
        embedding_dim=64,
        beta=0.25,
    ).to("cuda")

    from torchinfo import summary

    summary(vq_vae, (1, 3, 448, 448), device="cuda")


if __name__ == "__main__":
    main()
