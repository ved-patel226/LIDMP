import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = 40

    def forward(self, seq_len):
        t = (
            torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
            / self.scale
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    """
    Apply rotary embeddings to the first half of x.
    """
    # Split x into two parts: one for rotary embeddings and the other untouched
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    # Apply rotary embeddings to the rotary part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # Concatenate the rotary-applied and base parts
    return torch.cat([x_rot, x_base], dim=-1)


class MemoryOptimizedMLA(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_rope,
        d_kv_comp,
        window_size=None,
        use_chunked=True,
        chunk_size=1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_rope = d_rope
        self.d_kv_comp = d_kv_comp
        self.window_size = (
            window_size  # None for chunked attention, int for sliding window
        )
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size

        self.d_head = d_model // n_heads
        self.split_dim = self.d_head - d_rope

        # Projections
        self.W_dkv = nn.Linear(d_model, d_kv_comp)
        self.W_dq = nn.Linear(d_model, d_kv_comp)

        self.W_uk = nn.Linear(d_kv_comp, n_heads * self.split_dim)
        self.W_uv = nn.Linear(d_kv_comp, n_heads * self.d_head)
        self.W_uq = nn.Linear(d_kv_comp, n_heads * self.split_dim)

        self.W_qr = nn.Linear(d_kv_comp, n_heads * d_rope)
        self.W_kr = nn.Linear(d_model, n_heads * d_rope)

        self.rotary = RotaryEmbedding(d_rope)
        self.output = nn.Linear(n_heads * self.d_head, d_model)

    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        # KV Compression
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(batch_size, seq_len, self.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.n_heads, self.d_head)

        # Query Compression
        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, self.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, self.n_heads, self.d_rope)

        # Rotary embeddings with proper dimensions
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        # Apply rotary embeddings
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(h).view(batch_size, seq_len, self.n_heads, self.d_rope),
            cos,
            sin,
        )

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Transpose for attention computation: (B, L, H, D) -> (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Memory-efficient attention computation
        if self.window_size is not None:
            out = self._batched_sliding_window_attention(q, k, v)
        elif self.use_chunked and seq_len > 1024:
            out = self._chunked_attention(q, k, v)
        else:
            # Standard attention for smaller sequences
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        # Reshape: (B, H, L, D) -> (B, L, H*D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.output(out), (c_kv, k_rot)

    def _chunked_attention(self, q, k, v):
        """
        Compute attention in chunks to avoid materializing full attention matrix.
        Memory: O(chunk_size * seq_len) instead of O(seq_len^2)
        """
        batch_size, n_heads, seq_len, d_head = q.shape
        chunk_size = min(self.chunk_size, seq_len)

        output = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(d_head)

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]  # (B, H, chunk, D)

            # Compute attention scores for this chunk against all keys
            scores = (
                torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            )  # (B, H, chunk, L)
            attn = F.softmax(scores, dim=-1)

            # Compute output for this chunk
            output[:, :, i:end_i, :] = torch.matmul(attn, v)  # (B, H, chunk, D)

        return output

    def _batched_sliding_window_attention(self, q, k, v):
        """
        Batched sliding window attention - processes multiple queries at once.
        Memory: O(seq_len * window_size) instead of O(seq_len^2)
        Much faster than token-by-token processing.
        """
        batch_size, n_heads, seq_len, d_head = q.shape
        window_size = self.window_size
        scale = 1.0 / math.sqrt(d_head)

        # Create causal mask for sliding window
        # For each position i, attend to positions [max(0, i-window_size//2), min(seq_len, i+window_size//2+1)]
        output = torch.zeros_like(q)

        # Process in batches of queries
        query_chunk_size = min(256, seq_len)

        for q_start in range(0, seq_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, seq_len)
            q_chunk = q[:, :, q_start:q_end, :]  # (B, H, chunk, D)

            # For this chunk of queries, determine the window range
            kv_start = max(0, q_start - window_size // 2)
            kv_end = min(seq_len, q_end + window_size // 2)

            k_window = k[:, :, kv_start:kv_end, :]
            v_window = v[:, :, kv_start:kv_end, :]

            # Compute attention
            scores = torch.matmul(q_chunk, k_window.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            output[:, :, q_start:q_end, :] = torch.matmul(attn, v_window)

        return output
