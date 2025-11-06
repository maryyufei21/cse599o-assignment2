import torch
from einops import einsum, repeat, rearrange
from .multihead_self_attention import MultiheadSelfAttention, MultiheadSelfAttentionAnnotated
from .rmsnorm import RMSNorm, RMSNormAnnotated
from .swiglu import SwiGLU, SwiGLUAnnotated


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()

        self.mha = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype,
                                          use_rope=use_rope, theta=theta, max_seq_len=max_seq_len,
                                          token_positions=token_positions)
        self.rmsnorm_attention = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm_ff = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (b, s, d_model)
        x = x + self.mha(self.rmsnorm_attention(x))
        x = x + self.swiglu(self.rmsnorm_ff(x))
        return x

import torch.cuda.nvtx as nvtx

class TransformerBlockAnnotated(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()

        self.mha = MultiheadSelfAttentionAnnotated(d_model, num_heads, device=device, dtype=dtype,
                                          use_rope=use_rope, theta=theta, max_seq_len=max_seq_len,
                                          token_positions=token_positions)
        self.rmsnorm_attention = RMSNormAnnotated(d_model, device=device, dtype=dtype)
        self.rmsnorm_ff = RMSNormAnnotated(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLUAnnotated(d_model, d_ff, device=device, dtype=dtype)

    @nvtx.range("TransformerBlockAnnotated Forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (b, s, d_model)
        with nvtx.range("TransformerBlockAnnotated MHA"):
            x = x + self.mha(self.rmsnorm_attention(x))
        with nvtx.range("TransformerBlockAnnotated Feed Forward"):
            x = x + self.swiglu(self.rmsnorm_ff(x))
        return x