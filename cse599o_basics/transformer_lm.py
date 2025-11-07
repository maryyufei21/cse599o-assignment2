import torch
from einops import einsum, repeat, rearrange
from .transformer_block import TransformerBlock, TransformerBlockAnnotated
from .embedding import Embedding, EmbeddingAnnotated
from .rmsnorm import RMSNorm, RMSNormAnnotated
from .linear import Linear, LinearAnnotated

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int,
                 context_length: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # self.position_embedding = torch.nn.Embedding(context_length, d_model, device=device, dtype=dtype) if not use_rope else None
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype,
                             use_rope=use_rope, theta=theta, max_seq_len=context_length,
                             token_positions=token_positions)
            for _ in range(num_layers)
        ])
        self.layer_norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape (batch_size, sequence_length)
        x = self.embedding(input_ids)  # shape (b, s, d_model)
        for layer in self.layers:
            x = layer(x)  # shape (b, s, d_model)
        x = self.layer_norm_final(x)  # shape (b, s, d_model)
        logits = self.output_linear(x)  # shape (b, s, vocab_size)
        return logits  # shape (b, s, vocab_size)

import torch.cuda.nvtx as nvtx


class TransformerLMAnnotated(torch.nn.Module):
    def __init__(self, vocab_size: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int,
                 context_length: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 use_rope: bool = False,
                 theta: float | None = None,
                 token_positions: torch.Tensor | None = None):
        super().__init__()
        self.embedding = EmbeddingAnnotated(vocab_size, d_model, device=device, dtype=dtype)
        # self.position_embedding = torch.nn.Embedding(context_length, d_model, device=device, dtype=dtype) if not use_rope else None
        self.layers = torch.nn.ModuleList([
            TransformerBlockAnnotated(d_model, num_heads, d_ff, device=device, dtype=dtype,
                             use_rope=use_rope, theta=theta, max_seq_len=context_length,
                             token_positions=token_positions)
            for _ in range(num_layers)
        ])
        self.layer_norm_final = RMSNormAnnotated(d_model, device=device, dtype=dtype)
        self.output_linear = LinearAnnotated(d_model, vocab_size, device=device, dtype=dtype)

    @nvtx.range("TransformerLMAnnotated Forward")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape (batch_size, sequence_length)
        with nvtx.range("TransformerLMAnnotated Embedding"):
            x = self.embedding(input_ids)  # shape (b, s, d_model)
        
        with nvtx.range("TransformerLMAnnotated Layers"):
            for i, layer in enumerate(self.layers):
                with nvtx.range(f"TransformerLMAnnotated Layer_{i}"):
                    x = layer(x)  # shape (b, s, d_model)
        
        with nvtx.range("TransformerLMAnnotated Final_Norm"):
            x = self.layer_norm_final(x)  # shape (b, s, d_model)
        
        with nvtx.range("TransformerLMAnnotated Output_Linear"):
            logits = self.output_linear(x)  # shape (b, s, vocab_size)
        
        return logits  # shape (b, s, vocab_size)