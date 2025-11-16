import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term
    - gamma : codebook diversity loss weight
    - decay : exponential moving average decay for codebook update
    - epsilon : small constant for numerical stability
    """

    __slots__ = (
        "n_e",
        "e_dim",
        "beta",
        "gamma",
        "decay",
        "epsilon",
        "embedding",
        "device",
        "ema_cluster_size",
        "ema_w",
    )

    def __init__(
        self, n_e, e_dim, beta, device="cpu", gamma=0.01, decay=0.99, epsilon=1e-5
    ):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gamma = gamma  # diversity loss weight - reduced default
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.device = device

        # EMA for codebook update
        if self.decay > 0:
            self.register_buffer("ema_cluster_size", torch.zeros(n_e))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z):
        """
        Enhanced quantization with diversity encouragement
        """
        # Ensure input is on correct device
        z = z.to(self.device)

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        # Create min_encodings directly on the correct device (no redundant .to())
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=self.device, dtype=z.dtype
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute standard VQ loss
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        embedding_loss = torch.mean((z_q - z.detach()) ** 2)

        # Add diversity/entropy loss
        encodings_avg = min_encodings.mean(0) + 1e-10
        uniform_prior = torch.ones_like(encodings_avg) / self.n_e

        diversity_loss = torch.sum(
            uniform_prior * torch.log(uniform_prior / encodings_avg)
        )

        loss = (
            commitment_loss + self.beta * embedding_loss + self.gamma * diversity_loss
        )

        # EMA update of codebook
        if self.training and self.decay > 0:
            with torch.no_grad():  # Prevent graph building during EMA update
                self.ema_cluster_size = self.ema_cluster_size * self.decay + (
                    1 - self.decay
                ) * min_encodings.sum(0)

                n = self.ema_cluster_size.sum()
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.n_e * self.epsilon)
                    * n
                )

                dw = torch.matmul(min_encodings.t(), z_flattened)
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

                self.embedding.weight.data = (
                    self.ema_w / self.ema_cluster_size.unsqueeze(1)
                )

        # preserve gradients (straight-through estimator)
        z_q = z + (z_q - z).detach()

        # perplexity
        with torch.no_grad():  # Perplexity doesn't need gradients
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity.item(), min_encodings, min_encoding_indices
