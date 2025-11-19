import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    from .cbam import CBAM
except ImportError:
    from cbam import CBAM


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(
                in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class CoordinateAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        out = identity * a_h * a_w
        return out


class ResidualStack(nn.Module):
    """
    A stack of residual layers with optional attention.
    Args:
        in_dim (int): Input dimension.
        h_dim (int): Hidden layer dimension.
        res_h_dim (int): Hidden dimension of the residual block.
        n_res_layers (int): Number of residual layers.
        use_transformer (bool): Whether to use attention after each residual layer.
        window_size (int): Window size for sliding window attention (None for chunked attention).
        chunk_size (int): Chunk size for memory-efficient attention computation.
        use_gradient_checkpointing (bool): Enable gradient checkpointing to save memory.
    """

    def __init__(
        self,
        in_dim,
        h_dim,
        res_h_dim,
        n_res_layers,
        use_transformer=False,
    ):
        super().__init__()
        self.use_transformer = use_transformer

        self.res_layers = nn.ModuleList()
        for _ in range(n_res_layers):
            if use_transformer:
                self.res_layers.append(
                    nn.ModuleList(
                        [
                            # MemoryOptimizedMLA(
                            #     d_model=in_dim,
                            #     n_heads=2,
                            #     d_rope=4,
                            #     d_kv_comp=16,
                            #     window_size=window_size,
                            #     use_chunked=True,
                            #     chunk_size=chunk_size,
                            # ),
                            # CBAM(in_dim),
                            CoordinateAttention(in_dim, in_dim),
                            ResidualLayer(in_dim, h_dim, res_h_dim),
                        ]
                    )
                )

            else:
                self.res_layers.append(ResidualLayer(in_dim, h_dim, res_h_dim))

    def forward(self, x):
        for layer in self.res_layers:
            if self.use_transformer:
                attention_layer, res_layer = layer

                x = attention_layer(x)
                x = res_layer(x)
            else:

                x = layer(x)

        return x


if __name__ == "__main__":
    from time import perf_counter_ns

    model = ResidualStack(
        in_dim=64,
        h_dim=64,
        res_h_dim=256,
        n_res_layers=3,
        use_transformer=True,
    ).to("cuda")

    x = torch.randn(2, 64, 56, 56, device="cuda")

    start_time = perf_counter_ns()
    output = model(x)
    end_time = perf_counter_ns()
    print(f"Execution time: {(end_time - start_time)/1e6} ms")
    print(f"Output shape: {output.shape}")
