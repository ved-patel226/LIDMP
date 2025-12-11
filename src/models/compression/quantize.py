import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch import Tensor
from typing import List, Tuple, Optional, Union

from einops import rearrange, pack, unpack

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main FSQ class


class FSQ(nn.Module):
    """
    Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
    Code adapted from Jax version in Appendix A.1
    """

    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices


class BaseQuantizer(nn.Module):
    """
    Base class for all quantization methods.
    Ensures consistent interface across different quantization strategies.
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        use_vqinterface: bool = True,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.use_vqinterface = use_vqinterface

    def forward(
        self, z: Tensor, temp: Optional[float] = None, return_logits: bool = False
    ):
        """
        Args:
            z: Input tensor (B, num_hiddens, H, W)
            temp: Temperature parameter (optional, used by some quantizers)
            return_logits: Whether to return logits (optional)

        Returns:
            If use_vqinterface=True and return_logits=False:
                (z_q, loss, (None, None, indices))
            If use_vqinterface=True and return_logits=True:
                (z_q, loss, (None, None, indices), logits)
            If use_vqinterface=False:
                (z_q, loss, indices)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_codebook_entry(
        self, indices: Tensor, shape: Optional[Tuple] = None
    ) -> Tensor:
        """
        Reconstruct quantized vectors from indices.

        Args:
            indices: Quantization indices
            shape: Target shape (optional)

        Returns:
            z_q: Reconstructed tensor (B, embedding_dim, H, W)
        """
        raise NotImplementedError("Subclasses must implement get_codebook_entry method")


class GumbelQuantize(BaseQuantizer):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        num_hiddens,
        embedding_dim,
        n_embed,
        straight_through=True,
        kl_weight=5e-4,
        temp_init=1.0,
        use_vqinterface=True,
        remap=None,
        unknown_index="random",
    ):
        super().__init__(num_hiddens, embedding_dim, n_embed, use_vqinterface)

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape=None):
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = (
            F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        )
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q


class InvertibleLayerNorm(nn.Module):
    """Invertible LayerNorm module specialized for image feature maps (B, C, H, W)"""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.register_buffer("current_mean", None, persistent=False)
        self.register_buffer("current_std", None, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs LayerNorm and saves statistics for inverse transform
        Input: x (B, C, H, W)
        Output: normalized (B, C, H, W)
        """
        B, C, H, W = x.shape

        self.current_mean = x.mean(dim=[2, 3], keepdim=True)
        variance = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        self.current_std = torch.sqrt(variance + self.eps)

        normalized = (x - self.current_mean) / self.current_std

        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)

        return weight * normalized + bias

    def inverse(self, normalized_x: Tensor) -> Tensor:
        """
        Performs exact inverse transform using saved statistics
        Input: normalized_x (B, C, H, W)
        Output: original (B, C, H, W)
        """
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("Must call forward method first to save statistics")

        B, C, H, W = normalized_x.shape

        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)

        denormalized = (normalized_x - bias) / weight
        return denormalized * self.current_std + self.current_mean


class LearnableScalingRFSQStage(nn.Module):
    """RFSQ stage with learnable scaling factor"""

    def __init__(self, fsq_levels: List[int], dim: int, initial_scale: float = 1.0):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels, dim=dim)

        self.log_scale = nn.Parameter(torch.log(torch.tensor(initial_scale)))

    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale = F.softplus(self.log_scale)
        scaled_input = residual_in * scale
        quantized_scaled, indices = self.fsq(scaled_input)
        quantized_true = quantized_scaled / scale
        residual_out = residual_in - quantized_true

        return quantized_true, residual_out, indices


class BasicRFSQStage(nn.Module):
    """Basic RFSQ stage without preprocessing transforms"""

    def __init__(self, fsq_levels: List[int], dim: int):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels, dim=dim)

    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        quantized_true, indices = self.fsq(residual_in)
        residual_out = residual_in - quantized_true

        return quantized_true, residual_out, indices


class LayerNormRFSQStage(nn.Module):
    """RFSQ stage with invertible LayerNorm"""

    def __init__(self, fsq_levels: List[int], dim: int, num_channels: int):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels, dim=dim)
        self.layernorm = InvertibleLayerNorm(num_channels)

    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        normalized_input = self.layernorm(residual_in)
        quantized_normalized, indices = self.fsq(normalized_input)
        quantized_true = self.layernorm.inverse(quantized_normalized)
        residual_out = residual_in - quantized_true

        return quantized_true, residual_out, indices


class RFSQ(BaseQuantizer):
    """
    Residual Finite Scalar Quantization
    Achieves finer vector quantization through multi-stage residual quantization
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        num_stages: int = 4,
        strategy: str = "scale",
        fsq_levels: Optional[List[int]] = None,
        initial_scale: float = 1.0,
        use_vqinterface: bool = True,
        **kwargs,
    ):
        super().__init__(num_hiddens, embedding_dim, n_embed, use_vqinterface)

        if strategy not in ["scale", "layernorm", "none"]:
            raise ValueError("strategy must be 'scale', 'layernorm', or 'none'")

        self.num_stages = num_stages
        self.strategy = strategy

        if fsq_levels is None:
            fsq_levels = [16, 16, 4]

        self.fsq_levels = fsq_levels

        self.proj_in = nn.Conv2d(num_hiddens, embedding_dim, 1)

        self.stages = nn.ModuleList()

        for i in range(num_stages):
            if strategy == "scale":
                stage = LearnableScalingRFSQStage(
                    fsq_levels, dim=embedding_dim, initial_scale=initial_scale
                )
            elif strategy == "layernorm":
                stage = LayerNormRFSQStage(
                    fsq_levels, dim=embedding_dim, num_channels=embedding_dim
                )
            else:
                stage = BasicRFSQStage(fsq_levels, dim=embedding_dim)

            self.stages.append(stage)

    def forward(
        self, z: Tensor, temp: Optional[float] = None, return_logits: bool = False
    ):
        """
        Args:
            z: Input tensor (B, num_hiddens, H, W)
            temp: Temperature parameter (unused, kept for interface compatibility)
            return_logits: Whether to return logits (unused, kept for interface compatibility)

        Returns:
            If use_vqinterface=True:
                (z_q, loss, (None, None, indices))
            If use_vqinterface=False:
                (z_q, loss, indices)
        """
        z_proj = self.proj_in(z)

        all_quantized_vectors = []
        all_indices = []

        residual = z_proj

        for stage in self.stages:
            quantized_true, residual, indices = stage(residual)
            all_quantized_vectors.append(quantized_true)
            all_indices.append(indices)

        z_q = sum(all_quantized_vectors)

        indices_list = []
        for idx in all_indices:
            if idx.dim() > 3:
                B, H, W = idx.shape[:3]
                idx_flat = idx.reshape(B, H, W, -1)
            else:
                idx_flat = idx.unsqueeze(-1)
            indices_list.append(idx_flat)

        indices_tensor = torch.cat(indices_list, dim=-1)

        if indices_tensor.shape[-1] == 1:
            indices_tensor = indices_tensor.squeeze(-1)

        loss = F.mse_loss(z_q, z_proj)

        if self.use_vqinterface:
            if return_logits:
                return z_q, loss, (None, None, indices_tensor), None
            return z_q, loss, (None, None, indices_tensor)
        return z_q, loss, indices_tensor

    def get_codebook_entry(
        self, indices: Tensor, shape: Optional[Tuple] = None
    ) -> Tensor:
        """
        Reconstruct quantized vectors from indices.

        Args:
            indices: Quantization indices (B, H, W) or (B, H, W, total_indices)
            shape: Target shape (optional, unused)

        Returns:
            z_q: Reconstructed tensor (B, embedding_dim, H, W)
        """
        if indices.dim() == 3:
            B, H, W = indices.shape
            num_indices_per_stage = 1
        else:
            B, H, W, total_indices = indices.shape
            num_indices_per_stage = total_indices // self.num_stages

        all_quantized = []

        for stage_idx in range(self.num_stages):
            if indices.dim() == 3:
                stage_indices = indices
            else:
                start_idx = stage_idx * num_indices_per_stage
                end_idx = start_idx + num_indices_per_stage
                stage_indices = indices[..., start_idx:end_idx]

                if num_indices_per_stage == 1:
                    stage_indices = stage_indices.squeeze(-1)

            stage = self.stages[stage_idx]

            if isinstance(stage, LearnableScalingRFSQStage):
                scale = F.softplus(stage.log_scale)
                z_q_stage = stage.fsq.indices_to_codes(stage_indices)
                z_q_stage = z_q_stage / scale
            elif isinstance(stage, LayerNormRFSQStage):
                z_q_normalized = stage.fsq.indices_to_codes(stage_indices)
                z_q_stage = stage.layernorm.inverse(z_q_normalized)
            else:
                z_q_stage = stage.fsq.indices_to_codes(stage_indices)

            all_quantized.append(z_q_stage)

        z_q = sum(all_quantized)

        return z_q
