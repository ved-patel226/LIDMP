# NOTE -  From: https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# Modified for current project

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torchvision import models
import torch.nn.functional as F
import lpips

try:
    from .quantizer import VectorQuantizer
    from .residual import ResidualStack
except ImportError:
    from quantizer import VectorQuantizer
    from residual import ResidualStack


class Encoder(nn.Module):
    """
    Improved encoder with gradual channel expansion, spatial downsampling, then compression.
    Progressive downsampling: 448 -> 224 -> 112 -> 56 -> 28
    Channel progression: 3 -> 56 -> 112 -> 224 -> 448 -> embedding_dim

    Inputs:
    - in_dim : the input dimension (channels)
    - h_dim : the hidden layer dimension (base dimension)
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - embedding_dim : output channels for quantization
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            # Expand to 64 channels quickly
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 448 -> 224 (stay at 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 224 -> 112 (stay at 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 112 -> 56 (stay at 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, 16, kernel_size=3, stride=1, padding=1
            ),  # Output: (16, 56, 56)
            # # Gradual channel reduction: 64 -> 32 -> 16 -> 8 -> 1
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    """
    Decoder that exactly mirrors the encoder architecture.
    Progressive upsampling: 56 -> 112 -> 224 -> 448
    Channel progression: embedding_dim -> h_dim*4 -> h_dim*2 -> h_dim -> h_dim//2 -> out_channels

    Inputs:
    - in_dim : the input dimension (embedding_dim)
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    - out_channels : output image channels (default 3 for RGB)
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channels=3):
        super(Decoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 56 -> 112 (keep 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 112 -> 224 (keep 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 224 -> 448 (keep 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final refinement and project to 3 channels
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


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
    - lambda_recon : weight for reconstruction (L1) loss
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
        lambda_recon=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            3, h_dim, n_res_layers, res_h_dim, embedding_dim=embedding_dim
        )

        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device=device
        )
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.lr = lr
        self.lambda_lpips = lambda_lpips
        self.lambda_recon = lambda_recon

        self.criterion = nn.L1Loss()  # Reconstruction loss
        self.lpips_loss = lpips.LPIPS(net="vgg")

        # Freeze LPIPS network
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, x):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity

    def training_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)

        recon_loss = self.criterion(x_hat, x)
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = (
            self.lambda_recon * recon_loss
            + self.lambda_lpips * lpips_loss_val
            + embedding_loss
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_lpips_loss", lpips_loss_val)
        self.log("train_embedding_loss", embedding_loss)
        self.log("train_perplexity", perplexity)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, perplexity = self(x)

        recon_loss = self.criterion(x_hat, x)
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = (
            self.lambda_recon * recon_loss
            + self.lambda_lpips * lpips_loss_val
            + embedding_loss
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_lpips_loss", lpips_loss_val)
        self.log("val_embedding_loss", embedding_loss)
        self.log("val_perplexity", perplexity)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.vector_quantization.parameters()},
            ],
            lr=self.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
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
    model = VQVAE(
        h_dim=56,
        res_h_dim=128,
        n_res_layers=3,
        n_embeddings=256,
        embedding_dim=16,
        beta=0.25,
        lr=1e-3,
    ).to("cuda")

    from torchinfo import summary

    summary(model, (1, 3, 448, 448), device="cuda")


if __name__ == "__main__":
    main()
