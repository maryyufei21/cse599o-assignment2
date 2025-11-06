import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        # embedding_dim, i.e. d_model
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # size of weight: (num_embeddings, embedding_dim)
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim,
                                                     device=device, dtype=dtype))
        variance = 1.0
        std = variance ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
import torch.cuda.nvtx as nvtx

class EmbeddingAnnotated(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        # embedding_dim, i.e. d_model
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # size of weight: (num_embeddings, embedding_dim)
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim,
                                                     device=device, dtype=dtype))
        variance = 1.0
        std = variance ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3, b=3)

    @nvtx.range("Embedding Forward")
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        with nvtx.range("Embedding Lookup"):
            return self.weight[token_ids]