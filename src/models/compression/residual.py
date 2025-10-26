import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot


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
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    x = np.random.random_sample((3, 40, 448, 448))
    x = torch.tensor(x).float()

    res_layer = ResidualLayer(448, 448, 128)
    res_stack = ResidualStack(448, 448, 128, 3)

    total_params_layer = sum(p.numel() for p in res_layer.parameters())
    total_params_stack = sum(p.numel() for p in res_stack.parameters())

    print("ResidualLayer total params:", total_params_layer)
    print("ResidualStack (3 layers) total params:", total_params_stack)

    # output = res_stack(x)

    # dot = make_dot(output, params=dict(res_stack.named_parameters()))
    # dot.format = "png"
    # dot.render("residual_stack_graph")

    # print("Saved computation graph as residual_stack_graph.png")
