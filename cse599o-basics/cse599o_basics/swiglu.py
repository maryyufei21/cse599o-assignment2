import torch
from .linear import Linear, LinearAnnotated
from einops import einsum
import torch.cuda.nvtx as nvtx

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    # W2(SiLU(W1x) ⊙ W3x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (b, s, d_model)
        x1 = self.w1(x)
        sigmoid_x1 = torch.sigmoid(x1)
        silu_x1 = sigmoid_x1 * x1
        x3 = self.w3(x)
        silu_x1_x3 = einsum(silu_x1, x3, "b s d, b s d -> b s d")
        swiglu = self.w2(silu_x1_x3)
        return swiglu
14495514624

class SwiGLUAnnotated(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = LinearAnnotated(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = LinearAnnotated(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = LinearAnnotated(d_model, d_ff, device=device, dtype=dtype)

    # W2(SiLU(W1x) ⊙ W3x)
    @nvtx.range("SwiGLUAnnotated Forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (b, s, d_model)
        with nvtx.range("SwiGLUAnnotated W1_computation"):
            x1 = self.w1(x)
        
        with nvtx.range("SwiGLUAnnotated SiLU"):
            sigmoid_x1 = torch.sigmoid(x1)
            silu_x1 = sigmoid_x1 * x1
        
        with nvtx.range("SwiGLUAnnotated W3_computation"):
            x3 = self.w3(x)
        
        with nvtx.range("SwiGLUAnnotated element_wise_multiply"):
            silu_x1_x3 = einsum(silu_x1, x3, "b s d, b s d -> b s d")
        
        with nvtx.range("SwiGLUAnnotated W2_computation"):
            swiglu = self.w2(silu_x1_x3)
        
        return swiglu
