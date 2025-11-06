import torch
from cse599o_basics.decoding import Decoding
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.utils import cross_entropy_loss, load_checkpoint
from cse599o_basics.adamw import AdamW
import json

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# load config
config = load_config('./cse599o_basics/configs/basicfp32.json')

input_text = "Once upon a time, there was a pretty girl named Lily."
tokenizer = BPETokenizer(vocab={}, merges=[])
input_ids = tokenizer.encode(input_text)
print("Input IDs:", input_ids)
# model = torch.load('./checkpoints/basic/2025-10-14_00-24-13/checkpoint_iter_3000.pt', weights_only=True)
# model = torch.load('/homes/iws/yufeig21/assignment1-basics/checkpoints/basic/2025-10-14_01-16-00/checkpoint_iter_1000.pt', weights_only=True)
# model = torch.load('./checkpoints/2025-10-13_16-35-50/checkpoint_iter_500.pt', weights_only=True)
model=TransformerLM(
        # vocab_size=tokenizer.tokenizer.n_vocab,
        vocab_size=config['model']['vocab_size'],
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['max_seq_len'],
        use_rope=config['model']['use_rope'],
        theta=config['model'].get('rope_theta', None),
        token_positions=None,
        device=torch.device('cuda'),
        dtype=getattr(torch, config['model']['dtype'].split('.')[-1])
    )

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
load_checkpoint('/homes/iws/yufeig21/assignment1-basics/checkpoints/fp32/cosine_lr_500warmup/2025-10-14_14-37-29/checkpoint_iter_5000.pt', model, optimizer)
print("model.embedding.weight:", model.embedding.weight)
max_length = 256
temperature = 1
top_p = 0.5
print("Starting generation...")
print("generating parameters: max_length =", max_length, ", temperature =", temperature, ", top_p =", top_p)
output_ids, ori_logits = Decoding(
    model,
    input_ids=torch.tensor([input_ids], dtype=torch.int32, device=torch.device('cuda')),
    end_token_id=set([50256]),
    max_length=max_length,
    temperature=temperature,
    top_p=top_p
)
output_text = tokenizer.decode(output_ids[0].tolist())

# true_value = tokenizer.encode("Once upon a time, there was a pretty girl named Lily. She")[1:]
# print("True value:", true_value)
# print(type(true_value))
# print(type(torch.Tensor(true_value).to(torch.int32).cuda()))
# print("Output IDs:", ori_logits.shape)
# print("loss", cross_entropy_loss(ori_logits, torch.Tensor(true_value).to(torch.int32).cuda()))


print("Generated text:", output_text)