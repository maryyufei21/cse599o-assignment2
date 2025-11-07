import torch
from einops import reduce, repeat

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input tensor shape (batch_size, sequence_length, d_model)
        in_type = x.dtype
        x = x.to(torch.float32)
        ms = reduce(x ** 2, "b s d -> b s 1", "mean")
        rms = torch.sqrt(ms + self.eps)
        rmsnorm = x / repeat(rms, "b s 1 -> b s d", d=self.d_model) * self.weight
        return rmsnorm.to(in_type)
    
import torch.cuda.nvtx as nvtx

class RMSNormAnnotated(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    @nvtx.range("RMSNormAnnotated Forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input tensor shape (batch_size, sequence_length, d_model)
        in_type = x.dtype
        with nvtx.range("RMSNorm to float32"):
            x = x.to(torch.float32)
        with nvtx.range("RMSNorm compute rms"):
            ms = reduce(x ** 2, "b s d -> b s 1", "mean")
            rms = torch.sqrt(ms + self.eps)
        with nvtx.range("RMSNorm normalize"):
            rmsnorm = x / repeat(rms, "b s 1 -> b s d", d=self.d_model) * self.weight
        with nvtx.range("RMSNorm to original dtype"):
            rmsnorm = rmsnorm.to(in_type)
        return rmsnorm