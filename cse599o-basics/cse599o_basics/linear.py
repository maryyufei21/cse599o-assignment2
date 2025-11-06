
# Implement a Linear class that inherits from torch.nn.Module and performs a linear
# transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear
# module, except for not having a bias argument or parameter. 

import torch
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # size of weight: (out_features, in_features)
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features,
                                                     device=device, dtype=dtype))
        variance = 2.0 / (in_features + out_features)
        std = variance ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... in, out in -> ... out')

import torch.cuda.nvtx as nvtx

class LinearAnnotated(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # size of weight: (out_features, in_features)
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features,
                                                     device=device, dtype=dtype))
        variance = 2.0 / (in_features + out_features)
        std = variance ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    @nvtx.range("LinearAnnotated Forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx.range("LinearAnnotated Einsum"):
            return einsum(x, self.weight, '... in, out in -> ... out')