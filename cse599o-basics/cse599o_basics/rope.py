import torch
from einops import einsum, repeat, rearrange
import torch.cuda.nvtx as nvtx

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.create_r()
        
    def create_r(self) -> None:
        freq = self.theta ** (torch.arange(0, self.d_k, 2, device=self.device, dtype=torch.float32) / self.d_k)
        inv_freq = 1.0 / freq
        pos = torch.arange(0, self.max_seq_len, device=self.device, dtype=torch.float32)
        angles = einsum(pos, inv_freq, 'max_seq_len, d2 -> max_seq_len d2')  # (max_seq_len, d_k/2)
        cos = torch.cos(angles) # (max_seq_len, d_k/2)
        sin = torch.sin(angles) # (max_seq_len, d_k/2)
        R = rearrange([cos, -sin, sin, cos], "t max_seq_len d2 -> max_seq_len d2 t")
        R = rearrange(R, "max_seq_len d2 (r c) -> max_seq_len d2 r c", r=2)
        self.register_buffer("R", R, persistent=False)  # (max_seq_len, d_k/2, 2, 2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k) 
        # token_positions: (seq_len,)
        # return a tensor of the same shape
        # token_positions has shape (..., seq_len) and gives 
        # absolute positions per token along the sequence dimension
        
        # use token positions to slice R to (seq_len, d_k/2, 2, 2)
        R = self.R[token_positions]  # (seq_len, d_k/2, 2, 2)
        in_dtype = x.dtype
        # print("x shape before rearrange:", x.shape)
        x = rearrange(x, "... seq_len (d2 t) -> ... seq_len d2 t", t=2)  # (..., seq_len, d_k/2, 2)
        x = x.to(torch.float32)  # ensure x is float32 for matmul
        # print("x shape before einsum:", x.shape)
        # print("R shape before einsum:", R.shape)
        x_rot = einsum(x, R, "... seq_len d2 t, seq_len d2 r t -> ... seq_len d2 r")
        x_rot = rearrange(x_rot, "... seq_len d2 r -> ... seq_len (d2 r)")  # (..., seq_len, d_k)
        x_rot = x_rot.to(in_dtype)  # convert back to original dtype
        return x_rot

class RotaryPositionalEmbeddingAnnotated(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.create_r()
        
    @nvtx.range("RotaryPositionalEmbeddingAnnotated create_r")
    def create_r(self) -> None:
        with nvtx.range("RotaryPositionalEmbeddingAnnotated compute_frequencies"):
            freq = self.theta ** (torch.arange(0, self.d_k, 2, device=self.device, dtype=torch.float32) / self.d_k)
            inv_freq = 1.0 / freq
            pos = torch.arange(0, self.max_seq_len, device=self.device, dtype=torch.float32)
            angles = einsum(pos, inv_freq, 'max_seq_len, d2 -> max_seq_len d2')  # (max_seq_len, d_k/2)
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated compute_sin_cos"):
            cos = torch.cos(angles) # (max_seq_len, d_k/2)
            sin = torch.sin(angles) # (max_seq_len, d_k/2)
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated create_rotation_matrix"):
            R = rearrange([cos, -sin, sin, cos], "t max_seq_len d2 -> max_seq_len d2 t")
            R = rearrange(R, "max_seq_len d2 (r c) -> max_seq_len d2 r c", r=2)
            self.register_buffer("R", R, persistent=False)  # (max_seq_len, d_k/2, 2, 2)

    @nvtx.range("RotaryPositionalEmbeddingAnnotated Forward")
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k) 
        # token_positions: (seq_len,)
        # return a tensor of the same shape
        # token_positions has shape (..., seq_len) and gives 
        # absolute positions per token along the sequence dimension
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated slice_R"):
            # use token positions to slice R to (seq_len, d_k/2, 2, 2)
            R = self.R[token_positions]  # (seq_len, d_k/2, 2, 2)
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated rearrange_to_float32"):
            in_dtype = x.dtype
            x = rearrange(x, "... seq_len (d2 t) -> ... seq_len d2 t", t=2)  # (..., seq_len, d_k/2, 2)
            x = x.to(torch.float32)  # ensure x is float32 for matmul
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated apply_rotation"):
            x_rot = einsum(x, R, "... seq_len d2 t, seq_len d2 r t -> ... seq_len d2 r")
            x_rot = rearrange(x_rot, "... seq_len d2 r -> ... seq_len (d2 r)")  # (..., seq_len, d_k)
        
        with nvtx.range("RotaryPositionalEmbeddingAnnotated convert_back"):
            x_rot = x_rot.to(in_dtype)  # convert back to original dtype
        
        return x_rot
