import torch
import numpy as np  
import numpy.typing as npt
import typing
import os
from einops import reduce, rearrange, repeat, einsum
from collections.abc import Callable, Iterable
import math
import torch.cuda.nvtx as nvtx

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        torch.Tensor: Tensor with the same shape as input, containing the softmax values.
    """
    # Subtract the max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len_q, d_k).
        k (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len_k, d_k).
        v (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len_v, d_v).
        mask (torch.Tensor | None): Optional mask tensor broadcastable to (batch_size, ..., seq_len_q, seq_len_k).

    Returns:
        torch.Tensor: Output tensor of shape (..., seq_len_q, d_v).

    Note:
        - seq_len_q is noted as 'n' in the einsum notation.
        - seq_len_k and seq_len_v are noted as 'm' in the einsum notation.

    """
    qk = einsum(q, k, "... n d, ... m d -> ... n m")
    d_k = q.size(-1)
    scaled_qk = qk / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
    if mask is not None:
        scaled_qk = scaled_qk.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(scaled_qk, dim=-1)
    output = einsum(attn_weights, v, "... n m, ... m d -> ... n d")
    return output

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # logits: (batch_size, num_classes)
    # targets: (batch_size,) with class indices
    if logits.ndim == 3:
        logits = rearrange(logits, 'b s c -> (b s) c')
    if targets.ndim == 2:
        targets = rearrange(targets, 'b s -> (b s)')
    max_logits = reduce(logits, 'b c -> b 1', 'max')
    stabilized_logits = logits - max_logits
    log_sum = reduce(torch.exp(stabilized_logits), 'b c -> b', 'sum').log()
    # print("log_sum.shape:", log_sum.shape)
    target_logits = logits[torch.arange(logits.size(0)), targets]
    # print("target_logits.shape:", target_logits.shape)
    loss = log_sum - target_logits + max_logits
    return loss.mean()

def learning_rate_schedule_cosine(t: int, lr_max: float, lr_min: float, 
                                  T_warmup: int, T_cosine: int) -> float:
    if t < T_warmup:
        return lr_max * t / T_warmup
    elif t <= T_cosine:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_cosine - T_warmup)))
    else:
        return lr_min
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if param_norm > max_norm:
                p.grad.data.mul_(max_norm / (param_norm + eps))

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_length = len(dataset)
    max_start_index = dataset_length - context_length - 1
    starts = np.random.randint(0, max_start_index + 1, size=batch_size)
    inputs = np.array([dataset[start:start + context_length] for start in starts])
    targets = np.array([dataset[start + 1:start + context_length + 1]
                          for start in starts])
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the SiLU (Sigmoid Linear Unit) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with the same shape as input, where the SiLU function has been applied
                      to each element. The SiLU function is defined as SiLU(x) = x * sigmoid(x),
                      where sigmoid(x) = 1 / (1 + exp(-x)).
    
    Example:
        >>> import torch
        >>> from utils import silu
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> silu(x)
        tensor([-0.2689,  0.0000,  0.7311])
    
    Note:
        The SiLU activation function is also known as the "Swish" activation function.
        It is a smooth, non-monotonic function that has been shown to perform well in deep learning models.
    """
    
    return x * torch.sigmoid(x)

# ============================================================================
# Annotated versions with NVTX ranges for profiling
# ============================================================================

@nvtx.range("softmax_annotated")
def softmax_annotated(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        torch.Tensor: Tensor with the same shape as input, containing the softmax values.
    """
    with nvtx.range("softmax_annotated subtract_max"):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        e_x = torch.exp(x - x_max)
    with nvtx.range("softmax_annotated normalize"):
        sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
        return e_x / sum_e_x

@nvtx.range("scaled_dot_product_attention_annotated")
def scaled_dot_product_attention_annotated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len_q, d_k).
        k (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len_k, d_k).
        v (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len_v, d_v).
        mask (torch.Tensor | None): Optional mask tensor broadcastable to (batch_size, ..., seq_len_q, seq_len_k).

    Returns:
        torch.Tensor: Output tensor of shape (..., seq_len_q, d_v).

    Note:
        - seq_len_q is noted as 'n' in the einsum notation.
        - seq_len_k and seq_len_v are noted as 'm' in the einsum notation.

    """
    with nvtx.range("scaled_dot_product_attention_annotated qk_computation"):
        qk = einsum(q, k, "... n d, ... m d -> ... n m")
        d_k = q.size(-1)
        scaled_qk = qk / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))

    with nvtx.range("scaled_dot_product_attention_annotated apply_mask"):
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == False, float("-inf"))

    with nvtx.range("scaled_dot_product_attention_annotated softmax"):
        attn_weights = softmax_annotated(scaled_qk, dim=-1)

    with nvtx.range("scaled_dot_product_attention_annotated output_computation"):
        output = einsum(attn_weights, v, "... n m, ... m d -> ... n d")
    
    return output

@nvtx.range("cross_entropy_loss_annotated")
def cross_entropy_loss_annotated(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # logits: (batch_size, num_classes)
    # targets: (batch_size,) with class indices
    with nvtx.range("cross_entropy_loss_annotated reshape"):
        if logits.ndim == 3:
            logits = rearrange(logits, 'b s c -> (b s) c')
        if targets.ndim == 2:
            targets = rearrange(targets, 'b s -> (b s)')
    
    with nvtx.range("cross_entropy_loss_annotated compute_loss"):
        max_logits = reduce(logits, 'b c -> b 1', 'max')
        stabilized_logits = logits - max_logits
        log_sum = reduce(torch.exp(stabilized_logits), 'b c -> b', 'sum').log()
        target_logits = logits[torch.arange(logits.size(0)), targets]
        loss = log_sum - target_logits + max_logits
    
    return loss.mean()

@nvtx.range("learning_rate_schedule_cosine_annotated")
def learning_rate_schedule_cosine_annotated(t: int, lr_max: float, lr_min: float, 
                                           T_warmup: int, T_cosine: int) -> float:
    if t < T_warmup:
        return lr_max * t / T_warmup
    elif t <= T_cosine:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_cosine - T_warmup)))
    else:
        return lr_min

@nvtx.range("gradient_clipping_annotated")
def gradient_clipping_annotated(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6
    with nvtx.range("gradient_clipping_annotated clip_loop"):
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                if param_norm > max_norm:
                    p.grad.data.mul_(max_norm / (param_norm + eps))

@nvtx.range("data_loading_annotated")
def data_loading_annotated(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx.range("data_loading_annotated sample_indices"):
        dataset_length = len(dataset)
        max_start_index = dataset_length - context_length - 1
        starts = np.random.randint(0, max_start_index + 1, size=batch_size)
    
    with nvtx.range("data_loading_annotated create_arrays"):
        inputs = np.array([dataset[start:start + context_length] for start in starts])
        targets = np.array([dataset[start + 1:start + context_length + 1]
                              for start in starts])
    
    with nvtx.range("data_loading_annotated to_tensors"):
        inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs_tensor, targets_tensor

@nvtx.range("save_checkpoint_annotated")
def save_checkpoint_annotated(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
                              out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    with nvtx.range("save_checkpoint_annotated create_dict"):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }
    
    with nvtx.range("save_checkpoint_annotated torch_save"):
        torch.save(checkpoint, out)

@nvtx.range("load_checkpoint_annotated")
def load_checkpoint_annotated(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                              model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    with nvtx.range("load_checkpoint_annotated torch_load"):
        checkpoint = torch.load(src, weights_only=False)
    
    with nvtx.range("load_checkpoint_annotated load_state_dicts"):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
    
    return iteration

@nvtx.range("silu_annotated")
def silu_annotated(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the SiLU (Sigmoid Linear Unit) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with the same shape as input, where the SiLU function has been applied
                      to each element. The SiLU function is defined as SiLU(x) = x * sigmoid(x),
                      where sigmoid(x) = 1 / (1 + exp(-x)).
    
    Example:
        >>> import torch
        >>> from utils import silu_annotated
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> silu_annotated(x)
        tensor([-0.2689,  0.0000,  0.7311])
    
    Note:
        The SiLU activation function is also known as the "Swish" activation function.
        It is a smooth, non-monotonic function that has been shown to perform well in deep learning models.
    """
    with nvtx.range("silu_annotated computation"):
        return x * torch.sigmoid(x)
