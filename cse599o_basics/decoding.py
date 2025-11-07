import torch
from jaxtyping import Int, Float
from einops import einsum, repeat, rearrange
from .utils import softmax

def Decoding(
   model: torch.nn.Module,
   input_ids: Int[torch.Tensor, "batch_size seq_len"],
   end_token_id: set[int] | None = None,
   max_length: int = 1024,
   temperature: float = 0.5,
   top_p: float = 0.95
) -> Int[torch.Tensor, "batch_size generated_seq_len"]:
    """
    Generate a sequence of tokens using the provided model and input_ids as context.

    Args:
        model (torch.nn.Module): The language model to use for generation.
        input_ids (Int[torch.Tensor, "seq_len"]): The initial sequence of token IDs to start generation from.
        end_token_id (set[int] | None): A set of token IDs that indicate the end of the sequence. If any of these tokens are generated, the generation stops. If None, generation continues until max_length is reached.
        max_length (int): The maximum length of the generated sequence, including the input_ids length.
        temperature (float): The temperature to use for sampling. Higher values result in more random samples.
        top_p (float): The cumulative probability threshold for nucleus (top-p) filtering.
    Returns:
        Int[torch.Tensor, "generated_seq_len"]: The generated sequence of token IDs.
    """

    model.eval()
    
    for iter in range(max_length):
        logits = model(input_ids)  # shape (batch_size, seq_len, vocab_size)
        ori_logits = logits
        if temperature < 0:
            raise ValueError("Temperature must be non-negative")
        if temperature > 0:
            logits = logits[:, -1, :] / temperature  # shape (batch_size, vocab_size)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = softmax(logits, dim=-1)  # shape (batch_size, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # shape (batch_size, 1)
        # next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # shape (batch_size, 1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)  # shape (batch_size, seq_len + 1)
        if end_token_id is not None and next_token.item() in end_token_id:
            break
    return input_ids, logits  # shape (batch_size, generated_seq_len)