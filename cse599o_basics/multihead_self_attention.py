import torch

from einops import einsum, repeat, rearrange
from .utils import scaled_dot_product_attention, scaled_dot_product_attention_annotated
from .linear import Linear, LinearAnnotated
from .rope import RotaryPositionalEmbedding, RotaryPositionalEmbeddingAnnotated


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions

        self.W_Q = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_K = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_V = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_O = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_model)
        B, S, D = x.shape
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        q = rearrange(q, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
        k = rearrange(k, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
        v = rearrange(v, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
        if self.use_rope:
            # if self.token_positions is None:
            seq_len = x.size(-2)
            # print("seq_len:", seq_len)
            # print("x.shape:", x.shape)
            self.token_positions = torch.arange(seq_len, device=x.device)
            if self.token_positions.ndim == 2 and self.token_positions.size(0) == 1:
                self.token_positions = rearrange(self.token_positions, "1 s -> s")
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        mask = torch.ones((B, self.num_heads, S, S), dtype=torch.bool, device=x.device)
        mask = torch.tril(mask, diagonal=0)
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = rearrange(attn_output, "B numheads S headdim -> B S (numheads headdim)")
        output = self.W_O(attn_output)
        return output

import torch.cuda.nvtx as nvtx

class MultiheadSelfAttentionAnnotated(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions

        self.W_Q = LinearAnnotated(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_K = LinearAnnotated(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_V = LinearAnnotated(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_O = LinearAnnotated(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        if use_rope:
            self.rope = RotaryPositionalEmbeddingAnnotated(theta, self.head_dim, max_seq_len, device=device)
    
    @nvtx.range("MultiheadSelfAttentionAnnotated Forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_model)
        B, S, D = x.shape
        with nvtx.range("MultiheadSelfAttentionAnnotated QKV Computation"):
            q = self.W_Q(x)
            k = self.W_K(x)
            v = self.W_V(x)
            q = rearrange(q, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
            k = rearrange(k, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
            v = rearrange(v, "B S (numheads headdim) -> B numheads S headdim", numheads=self.num_heads)
        with nvtx.range("MultiheadSelfAttentionAnnotated RoPE"):
            if self.use_rope:
                # if self.token_positions is None:
                seq_len = x.size(-2)
                # print("seq_len:", seq_len)
                # print("x.shape:", x.shape)
                self.token_positions = torch.arange(seq_len, device=x.device)
                if self.token_positions.ndim == 2 and self.token_positions.size(0) == 1:
                    self.token_positions = rearrange(self.token_positions, "1 s -> s")
                q = self.rope(q, self.token_positions)
                k = self.rope(k, self.token_positions)
        with nvtx.range("MultiheadSelfAttentionAnnotated Generate Mask"):
            mask = torch.ones((B, self.num_heads, S, S), dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=0)
        with nvtx.range("MultiheadSelfAttentionAnnotated Scaled Dot-Product Attention"):
            attn_output = scaled_dot_product_attention_annotated(q, k, v, mask)
            attn_output = rearrange(attn_output, "B numheads S headdim -> B S (numheads headdim)")
        output = self.W_O(attn_output)
        return output