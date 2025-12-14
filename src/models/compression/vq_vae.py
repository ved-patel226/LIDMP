import torch
import pytorch_lightning as pl
import lpips

try:
    from .transformer import LatentDecoder, LatentEncoder
except ImportError:
    from transformer import LatentDecoder, LatentEncoder


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        h_dim,
        embedding_dim,
        lr=1e-3,
        n_heads=4,
        n_layers=3,
        patch_size=1,
        freeze_encoder=False,
        freeze_decoder=False,
        load_params=None,
        load_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        self.encoder = LatentEncoder(
            input_channels=3,
            latent_channels=embedding_dim,
            embed_dim=h_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_size=patch_size,
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

        self.lpips_loss = lpips.LPIPS(net="vgg")
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
            else:
                print(f"Warning: Unknown component '{component_name}'")
                continue

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

    def forward(self, x):
        z_e = self.encoder(x)

        if self.training:
            z_q = z_e + torch.empty_like(z_e).uniform_(-0.5, 0.5)
        else:
            z_q = torch.round(z_e)

        x_hat = self.decoder(z_q)
        return x_hat

    def _calculate_loss(self, x_hat, x, log_name="train"):
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        opt = self.optimizers() if hasattr(self, "optimizers") else None
        lr = opt.param_groups[0]["lr"] if opt else self.lr
        self.log("lr", lr, prog_bar=True)
        self.log(f"{log_name}_loss", lpips_loss_val, prog_bar=True)

        return lpips_loss_val

    def training_step(self, batch, _batch_idx):
        x, _ = batch
        x_hat = self(x)
        return self._calculate_loss(x_hat, x, log_name="train")

    def validation_step(self, batch, _batch_idx):
        x, _ = batch
        x_hat = self(x)
        return self._calculate_loss(x_hat, x, log_name="val")

    def configure_optimizers(self):
        params_to_optimize = []

        if not self.freeze_encoder:
            params_to_optimize.append({"params": self.encoder.parameters()})
        if not self.freeze_decoder:
            params_to_optimize.append({"params": self.decoder.parameters()})

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
        embedding_dim=64,
        lr=1e-3,
        n_heads=4,
        n_layers=3,
        patch_size=1,
    )

    summary(
        model,
        input_size=(1, 3, 448, 448),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        device=device,
    )


if __name__ == "__main__":
    main()
