import torch
import torch.nn as nn
import pytorch_lightning as pl
import lpips
import torch.nn.init as init
from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ,
    ResidualFSQ,
    ResidualLFQ,
    FSQ,
    LFQ,
)

try:
    from .transformer import LatentDecoder, LatentEncoder
    from .quantize import GumbelQuantize
except ImportError:
    from transformer import LatentDecoder, LatentEncoder
    from quantize import GumbelQuantize


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
        load_params=None,
        load_path=None,
        is_fsq=False,
        num_quantizers=2,
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

        # self.vector_quantization = ResidualLFQ(
        #     codebook_size=n_embeddings,
        #     dim=embedding_dim,
        #     num_quantizers=num_quantizers,
        #     commitment_loss_weight=1.0,
        #     entropy_loss_weight=0.01,  # Helps with codebook utilization
        #     quantize_dropout=True,
        # )

        self.vector_quantization = GumbelQuantize(
            num_hiddens=embedding_dim,
            embedding_dim=embedding_dim,
            n_embed=n_embeddings,
            straight_through=True,
            kl_weight=beta,
            temp_init=1.0,
            use_vqinterface=True,
        )

        # self.vector_quantization = LFQ(
        #     codebook_size=n_embeddings,  # e.g., 65536 = 2^16 bits
        #     dim=embedding_dim,
        #     entropy_loss_weight=0.2,
        #     force_quantization_f32=False,
        # )

        self.is_fsq = is_fsq
        # self.vector_quantization = FSQ(
        #     levels=[16, 16, 4],
        #     num_codebooks=1,
        #     dim=embedding_dim,
        #     preserve_symmetry=True,
        #     force_quantization_f32=False,
        # )

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

        if load_params is not None and load_path is not None:
            self._load_pretrained_components(load_path, load_params)

        print("VQVAE __init__ complete.")

    def _load_pretrained_components(self, checkpoint_path, component_names):
        """Load specific components from a checkpoint"""
        print(f"Loading components {component_names} from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        for component_name in component_names:
            if component_name == "encoder":
                component = self.encoder
            elif component_name == "decoder":
                component = self.decoder
            elif component_name == "vector_quantization":
                component = self.vector_quantization
            else:
                print(f"Warning: Unknown component '{component_name}'")
                continue

            # Filter state dict for this component
            prefix = f"{component_name}."
            component_state = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            if component_state:
                component.load_state_dict(component_state, strict=False)
                print(f"Loaded {len(component_state)} parameters for {component_name}")
            else:
                print(f"Warning: No parameters found for {component_name}")

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

    # def _calculate_vq_perplexity(self, codebook_indices, num_codebook_entries):
    #     """
    #     Calculate perplexity of VQ codebook usage.

    #     Args:
    #         codebook_indices: Tensor of shape (batch_size, ...) containing codebook indices
    #         num_codebook_entries: K, the size of the codebook

    #     Returns:
    #         perplexity: scalar tensor
    #     """
    #     flat_indices = codebook_indices.reshape(-1)

    #     unique, counts = torch.unique(flat_indices, return_counts=True)

    #     full_counts = torch.zeros(num_codebook_entries, device=codebook_indices.device)
    #     full_counts[unique] = counts.float()
    #     probs = full_counts / full_counts.sum()

    #     probs = probs[probs > 0]

    #     entropy = -(probs * torch.log2(probs)).sum()
    #     perplexity = 2**entropy

    #     return perplexity

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

        # Handle FSQ/ResidualFSQ which may return indices outside expected range
        # or negative values. For these quantizers, compute perplexity based on
        # unique values actually present rather than assuming a fixed codebook size.
        unique, counts = torch.unique(flat_indices, return_counts=True)

        # Calculate probabilities directly from counts
        probs = counts.float() / counts.sum()

        # Filter out zero probabilities (shouldn't happen here, but safe)
        probs = probs[probs > 0]

        entropy = -(probs * torch.log2(probs)).sum()
        perplexity = 2**entropy

        return perplexity

    def forward(self, x):
        z_e = self.encoder(x)
        b, c, h, w = z_e.shape

        if self.is_fsq:
            z_flat = z_e.permute(0, 2, 3, 1).reshape(b, h * w, c)
            commitment_loss = torch.tensor(0.0, device=x.device)
            quantized, indices = self.vector_quantization(z_flat)
            quantized_spatial = quantized.view(b, h, w, c).permute(0, 3, 1, 2)
        else:
            # GumbelQuantize returns (quantized, commitment_loss, indices)
            # When use_vqinterface=True, indices is a tuple (None, None, actual_indices)
            quantized_spatial, commitment_loss, indices = self.vector_quantization(z_e)

            # Handle GumbelQuantize's vqinterface format: (None, None, ind)
            if isinstance(indices, tuple):
                indices = indices[-1]  # Get the actual indices (last element)

            # Convert to tensor if needed
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices, device=x.device)

            # Flatten indices for perplexity calculation
            # GumbelQuantize returns indices with shape (B, H, W)
            if indices.dim() == 4:
                indices = indices.view(b, -1)
            elif indices.dim() == 3:
                indices = indices.view(b, -1)

        perplexity = self._calculate_vq_perplexity(indices, self.n_embeddings)

        x_hat = self.decoder(quantized_spatial)

        return commitment_loss.mean(), x_hat, indices, perplexity

    def forward_with_latents(self, latents, spatial_shape=None):
        """
        Accept integer codebook indices and decode them.

        Args:
            latents: integer codebook indices with shape (B, H*W, Q) where Q is num_quantizers
            spatial_shape: tuple (H, W) for the spatial dimensions. If None, assumes square.
        """
        # get_output_from_indices returns (B, N, D) where N = H*W
        embeddings = self.vector_quantization.get_output_from_indices(latents)

        b, n, d = embeddings.shape

        if spatial_shape is not None:
            h, w = spatial_shape
        else:
            # Assume square spatial dimensions
            h = w = int(n**0.5)

        assert (
            h * w == n
        ), f"Spatial shape {h}x{w}={h*w} doesn't match sequence length {n}"

        # Reshape from (B, H*W, D) to (B, D, H, W) for the decoder
        embeddings = embeddings.view(b, h, w, d).permute(0, 3, 1, 2)

        return self.decoder(embeddings)

    # def _calculate_loss(
    #     self, x_hat, x, embedding_loss, perplexity=None, log_name="train", indices=None
    # ):
    #     lpips_loss_val = self.lpips_loss(x_hat, x).mean()

    #     loss = lpips_loss_val + embedding_loss

    #     opt = self.optimizers() if hasattr(self, "optimizers") else None
    #     lr = opt.param_groups[0]["lr"] if opt else self.lr
    #     self.log("lr", lr, prog_bar=True)

    #     self.log(f"{log_name}_loss", loss, prog_bar=True)
    #     self.log(f"{log_name}_embedding_loss", embedding_loss)
    #     self.log(f"{log_name}_lpips_loss", lpips_loss_val)

    #     if perplexity is not None:
    #         self.log(f"{log_name}_perplexity", perplexity)
    #     if indices is not None:
    #         unique_codes = len(torch.unique(indices))
    #         self.log(f"{log_name}_unique_codes", unique_codes)
    #         self.log(
    #             f"{log_name}_codebook_utilization",
    #             unique_codes / self.hparams.n_embeddings,
    #         )
    #     return loss

    def _calculate_loss(
        self, x_hat, x, embedding_loss, perplexity=None, log_name="train", indices=None
    ):
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
            self.log(f"{log_name}_unique_codes", float(unique_codes))
            # For FSQ, codebook utilization is less meaningful, but we can still log it
            # as a ratio of unique codes used vs total possible combinations
            self.log(
                f"{log_name}_codebook_utilization",
                float(unique_codes) / max(self.hparams.n_embeddings, unique_codes),
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

        # optimizer = FusedAdam(
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
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
        n_embeddings=1023,
        embedding_dim=64,
        beta=0.25,
        lr=1e-3,
        n_heads=4,
        n_layers=3,
        patch_size=1,
        freeze_encoder=True,  # Freeze encoder (random initialization)
        freeze_decoder=True,  # Freeze decoder (random initialization)
        freeze_quantizer=False,  # Train quantizer only
        # load_params=["encoder", "decoder"],
        # load_path="checkpoints/stage1/vq_vae_quantizer_pretrain-v2.ckpt",
    )

    summary(
        model,
        input_size=(1, 3, 448, 448),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        device=device,
    )

    x = torch.randn(1, 3, 448, 448).to(device)

    _, _, indices, _ = model(x)

    print(indices.shape)


if __name__ == "__main__":
    main()
