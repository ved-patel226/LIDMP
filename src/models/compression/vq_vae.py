import torch
import torch.nn as nn
import pytorch_lightning as pl
import lpips
import torch.nn.init as init
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

try:
    from .transformer import LatentDecoder, LatentEncoder
except ImportError:
    from transformer import LatentDecoder, LatentEncoder


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        h_dim,
        n_embeddings,
        embedding_dim,
        beta=1.0,
        lr=1e-3,
        n_heads=4,
        n_layers=3,
        patch_size=1,
        freeze_encoder=False,
        freeze_decoder=False,
        freeze_quantizer=False,
        gradient_descent_quantizer=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_quantizer = freeze_quantizer
        super().__init__()

        self.save_hyperparameters()

        self.encoder = LatentEncoder(
            input_channels=3,
            latent_channels=embedding_dim,
            embed_dim=h_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_size=patch_size,
        )

        self.vector_quantization = VectorQuantize(
            codebook_size=n_embeddings,
            dim=embedding_dim,
            decay=0.99,
            eps=1e-5,
            commitment_weight=beta,
            learnable_codebook=True if gradient_descent_quantizer else False,
            ema_update=False if gradient_descent_quantizer else True,
        )

        self.decoder = LatentDecoder(
            latent_channels=embedding_dim,
            output_channels=3,
            embed_dim=h_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_size=patch_size,
        )

        self.lr = lr

        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.n_embeddings = n_embeddings

        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        self._apply_freezing()

        print("VQVAE __init__ complete.")

    def _apply_freezing(self):
        """Freeze components based on initialization flags"""
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("Decoder frozen")

        if self.freeze_quantizer:
            for param in self.vector_quantization.parameters():
                param.requires_grad = False
            print("Quantizer frozen")

    def _calculate_vq_perplexity(self, codebook_indices, num_codebook_entries):
        """
        Calculate perplexity of VQ codebook usage.

        Args:
            codebook_indices: Tensor of shape (batch_size, ...) containing codebook indices
            num_codebook_entries: K, the size of the codebook

        Returns:
            perplexity: scalar tensor
        """
        flat_indices = codebook_indices.reshape(-1)

        unique, counts = torch.unique(flat_indices, return_counts=True)

        full_counts = torch.zeros(num_codebook_entries, device=codebook_indices.device)
        full_counts[unique] = counts.float()
        probs = full_counts / full_counts.sum()

        probs = probs[probs > 0]

        entropy = -(probs * torch.log2(probs)).sum()
        perplexity = 2**entropy

        return perplexity

    def forward(self, x):
        z_e = self.encoder(x)
        b, c, h, w = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(b, h * w, c)
        quantized, indices, commitment_loss = self.vector_quantization(z_flat)

        perplexity = self._calculate_vq_perplexity(indices, self.n_embeddings)

        quantized_spatial = quantized.view(b, h, w, c).permute(0, 3, 1, 2)
        x_hat = self.decoder(quantized_spatial)

        return commitment_loss.mean(), x_hat, indices, perplexity

    def _calculate_loss(
        self, x_hat, x, embedding_loss, perplexity=None, log_name="train", indices=None
    ):
        # Normalize LPIPS loss to prevent gradient explosion
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = lpips_loss_val + embedding_loss

        opt = self.optimizers() if hasattr(self, "optimizers") else None
        lr = opt.param_groups[0]["lr"] if opt else self.lr
        self.log("lr", lr, prog_bar=True)

        self.log(f"{log_name}_loss", loss, prog_bar=True)
        self.log(f"{log_name}_embedding_loss", embedding_loss)
        self.log(f"{log_name}_lpips_loss", lpips_loss_val)

        if perplexity is not None:
            self.log(f"{log_name}_perplexity", perplexity)
        if indices is not None:
            unique_codes = len(torch.unique(indices))
            self.log(f"{log_name}_unique_codes", unique_codes)
            self.log(
                f"{log_name}_codebook_utilization",
                unique_codes / self.hparams.n_embeddings,
            )
        return loss

    def training_step(self, batch, _batch_idx):
        x, _ = batch
        embedding_loss, x_hat, indices, perplexity = self(x)

        loss = self._calculate_loss(
            x_hat, x, embedding_loss, perplexity, log_name="train", indices=indices
        )
        return loss

    def validation_step(self, batch, _batch_idx):
        x, _ = batch
        embedding_loss, x_hat, indices, perplexity = self(x)

        loss = self._calculate_loss(
            x_hat, x, embedding_loss, perplexity, log_name="val", indices=indices
        )

        return loss

    def configure_optimizers(self):
        params_to_optimize = []

        if not self.freeze_encoder:
            params_to_optimize.append({"params": self.encoder.parameters()})
        if not self.freeze_decoder:
            params_to_optimize.append({"params": self.decoder.parameters()})
        if not self.freeze_quantizer:
            params_to_optimize.append({"params": self.vector_quantization.parameters()})

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6
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

    device = "cpu"

    model = VQVAE(
        h_dim=256,
        n_embeddings=512 * 2,
        embedding_dim=16,
        beta=0.25,
        lr=1e-3,
        #
        #
        #
        n_heads=4,
        n_layers=6,
        patch_size=4,
    ).to(device)

    summary(model, input_size=(1, 3, 448, 448), device=device)

    x = torch.randn(1, 3, 448, 448).to(device)
    with torch.no_grad():
        commitment_loss, x_hat, indices, perplexity = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_hat.shape)
    print(perplexity)


if __name__ == "__main__":
    main()
