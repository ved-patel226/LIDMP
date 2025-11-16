# NOTE -  From: https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# Modified for current project

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            CBAM(out_channels),
        )

    def forward(self, x):
        x = self.initial(x)

        # Apply all downsampling blocks dynamically
        for downsample in self.downsample_blocks:
            x = downsample(x)

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
        for _ in range(num_upsamples):
            self.upsample_blocks.append(
                self._upsample_block(current_channels, current_channels)
            )

        self.residual_stack = ResidualStack(
            in_dim=current_channels,
            h_dim=current_channels,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
        )

        # self.transformer_stack = nn.Sequential(
        #     WindowTransformerBlock(
        #         current_channels, window_size=7, num_heads=4, mlp_ratio=2
        #     ),
        #     WindowTransformerBlock(
        #         current_channels, window_size=7, num_heads=4, mlp_ratio=2
        #     ),
        #     WindowTransformerBlock(
        #         current_channels, window_size=7, num_heads=4, mlp_ratio=4
        #     ),
        # )

        self.final = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 3, 1, 1),
            nn.BatchNorm2d(current_channels),
            nn.LeakyReLU(),
            nn.Conv2d(current_channels, current_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(current_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(current_channels // 2, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def forward(self, x):
        x = self.expand(x)
        for upsample in self.upsample_blocks:
            x = upsample(x)

        x = self.residual_stack(x)
        # x = self.transformer_stack(x)
        x = self.final(x)
        return x

    def _upsample_block(self, in_channels, out_channels, upscale_factor=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.BatchNorm2d(out_channels * (upscale_factor**2)),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor),
            CBAM(out_channels),
        )

    @staticmethod
    def ICNR(tensor, upscale_factor=2, initializer=init.kaiming_normal_):
        out_channels, in_channels, kernel_size, _ = tensor.shape
        new_shape = (
            out_channels // (upscale_factor**2),
            in_channels,
            kernel_size,
            kernel_size,
        )
        subkernel = torch.zeros(new_shape)
        initializer(subkernel)

        subkernel = subkernel.transpose(0, 1).contiguous().view(new_shape[1], -1)
        subkernel = subkernel.repeat_interleave(upscale_factor**2, dim=1)
        subkernel = subkernel.view(
            out_channels, in_channels, kernel_size, kernel_size
        ).transpose(0, 1)
        return subkernel.contiguous()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if any(isinstance(parent, nn.PixelShuffle) for parent in m.children()):
                    # This condition might not catch conv layers inside Sequential; alternatively check layer names or structure.
                    # Apply ICNR initialization for conv2d preceding PixelShuffle
                    m.weight.data.copy_(self.ICNR(m.weight.data, upscale_factor=2))
                else:
                    init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


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
        beta=1.0,
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

        # self.vector_quantization = VectorQuantizer(
        #     n_embeddings, embedding_dim, beta, device=device
        # )

        self.vector_quantization = VectorQuantize(
            codebook_size=n_embeddings,
            dim=embedding_dim,
            decay=0.99,
            eps=1e-5,
            commitment_weight=beta,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,  # Replace unused codes
            sample_codebook_temp=0.0,  # Enable stochastic sampling during training
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

        self.lpips_loss = lpips.LPIPS(net="vgg")

        for param in self.lpips_loss.parameters():
            param.requires_grad = False

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
        lpips_loss_val = self.lpips_loss(x_hat, x).mean()

        loss = lpips_loss_val + embedding_loss
        # loss = lpips_loss_val + embedding_loss

        opt = self.optimizers() if hasattr(self, "optimizers") else None
        lr = opt.param_groups[0]["lr"]
        self.log(f"lr", lr, prog_bar=True)

        # self.log(f"{log_name}_recon_loss", recon_loss)
        # self.log(f"{log_name}_lpips_loss", lpips_loss_val)
        self.log(f"{log_name}_loss", loss, prog_bar=True)
        self.log(f"{log_name}_embedding_loss", embedding_loss)

        if perplexity is not None:
            self.log(f"{log_name}_perplexity", perplexity)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, _ = self(x)

        loss = self._calculate_loss(x_hat, x, embedding_loss, log_name="train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        embedding_loss, x_hat, _ = self(x)

        loss = self._calculate_loss(x_hat, x, embedding_loss, log_name="val")

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
            weight_decay=1e-5,
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


def print_layers_by_vram(model, input_shape, device, top_k=10):
    """
    Print layers sorted by VRAM usage.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (e.g., (1, 3, 448, 448))
        device: torch device
        top_k: Number of top layers to display
    """
    import torch
    from collections import defaultdict

    # Store memory usage per layer
    layer_memory = {}

    # Hook to track memory allocation
    def hook_fn(module, input, output):
        module_name = str(module.__class__.__name__)

        # Find the actual layer name in the model
        for name, mod in model.named_modules():
            if mod is module:
                module_name = name if name else module_name
                break

        # Calculate memory usage
        mem_params = sum(p.numel() * p.element_size() for p in module.parameters())

        # Calculate output tensor memory
        if isinstance(output, torch.Tensor):
            mem_output = output.numel() * output.element_size()
        elif isinstance(output, (tuple, list)):
            mem_output = sum(
                o.numel() * o.element_size()
                for o in output
                if isinstance(o, torch.Tensor)
            )
        else:
            mem_output = 0

        # Calculate input tensor memory
        if isinstance(input, tuple) and len(input) > 0:
            mem_input = sum(
                i.numel() * i.element_size()
                for i in input
                if isinstance(i, torch.Tensor)
            )
        else:
            mem_input = 0

        total_mem = mem_params + mem_output
        layer_memory[module_name] = {
            "params": mem_params,
            "output": mem_output,
            "input": mem_input,
            "total": total_mem,
        }

    # Register hooks
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Sort by total memory usage
    sorted_layers = sorted(
        layer_memory.items(), key=lambda x: x[1]["total"], reverse=True
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"Top {top_k} Layers by VRAM Usage")
    print(f"{'='*80}\n")
    print(
        f"{'Layer Name':<40} {'Params (MB)':<15} {'Output (MB)':<15} {'Total (MB)':<15}"
    )
    print(f"{'-'*80}")

    total_vram = 0
    for name, mem in sorted_layers[:top_k]:
        params_mb = mem["params"] / (1024**2)
        output_mb = mem["output"] / (1024**2)
        total_mb = mem["total"] / (1024**2)
        total_vram += total_mb

        print(f"{name:<40} {params_mb:<15.2f} {output_mb:<15.2f} {total_mb:<15.2f}")

    print(f"{'-'*80}")
    print(
        f"{'Total (Top ' + str(top_k) + ')':<40} {'':<15} {'':<15} {total_vram:<15.2f}"
    )

    # Print overall memory stats
    total_params = sum(m["params"] for m in layer_memory.values()) / (1024**2)
    total_output = sum(m["output"] for m in layer_memory.values()) / (1024**2)
    grand_total = sum(m["total"] for m in layer_memory.values()) / (1024**2)

    print(f"\n{'='*80}")
    print(f"Overall Memory Statistics")
    print(f"{'='*80}")
    print(f"Total Parameters Memory: {total_params:.2f} MB")
    print(f"Total Activations Memory: {total_output:.2f} MB")
    print(f"Grand Total: {grand_total:.2f} MB")
    print(f"{'='*80}\n")

    return sorted_layers


def main() -> None:
    import torch
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(
        h_dim=56,
        res_h_dim=256,
        n_res_layers=3,
        n_embeddings=512 * 2,
        embedding_dim=64,
        beta=0.25,
        lr=1e-3,
        num_downsamples=3,
        initial_channels=128,
        device=device,
    ).to(device)

    info = summary(model, (1, 3, 448, 448), device=device, verbose=0)
    print(info)

    print_layers_by_vram(model, (1, 3, 448, 448), device, top_k=15)


if __name__ == "__main__":
    main()
