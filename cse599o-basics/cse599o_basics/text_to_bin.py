# import numpy as np
# from pathlib import Path
# from typing import Iterable, Optional
# from tokenizer import BPETokenizer

# def txt_to_bin(
#     txt_path: str | Path,
#     bin_path: str | Path,
#     tokenizer: BPETokenizer,
#     *,
#     dtype=np.uint16,
#     buffer_tokens: int = 2_000_000,
#     add_prefix_space: bool = False,  # if you want GPT-2-like splitting consistency
# ) -> dict:
#     """
#     Stream-encode a .txt and write flat binary token ids.

#     Returns a small metadata dict you can save alongside.
#     """
#     txt_path, bin_path = Path(txt_path), Path(bin_path)
#     total = 0
#     vmax = np.iinfo(dtype).max

#     with open(txt_path, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
#         buf = []
#         for line in fin:
#             if add_prefix_space and (buf == [] or (len(buf) > 0 and buf[-1] == "\n")):
#                 line = " " + line  # optional GPT-2-ish quirk
#             ids = tokenizer.encode(line)
#             if ids:
#                 if max(ids) > vmax:
#                     raise ValueError(
#                         f"Token id {max(ids)} exceeds {dtype}. "
#                         f"Use np.uint32 and load with dtype=np.uint32."
#                     )
#                 buf.extend(ids)
#             # flush buffer periodically to keep RAM low
#             if len(buf) >= buffer_tokens:
#                 np.asarray(buf, dtype=dtype).tofile(fout)
#                 total += len(buf)
#                 buf.clear()
#         # final flush
#         if buf:
#             np.asarray(buf, dtype=dtype).tofile(fout)
#             total += len(buf)

#     meta = {
#         "txt_path": str(txt_path),
#         "bin_path": str(bin_path),
#         "token_count": total,
#         "dtype": str(dtype),
#         "n_vocab": tokenizer.tokenizer.n_vocab,
#     }
#     return meta

# # Example usage
# tokenizer = BPETokenizer.from_serialized(vocab={}, merges=[], special_tokens=["<|bos|>", "<|eos|>"])
# meta = txt_to_bin("./data/data/TinyStoriesV2-GPT4-train.txt", "./data/train.bin", tokenizer, dtype=np.uint16)
# val_meta = txt_to_bin("./data/data/TinyStoriesV2-GPT4-valid.txt", "./data/val.bin", tokenizer, dtype=np.uint16)
# print(meta, val_meta)




# from cse599o_basics.tokenizer import BPETokenizer
# import numpy as np
# def txt_encode(in_path: str, out_path: str):
#     tok = BPETokenizer(vocab={}, merges=[])
#     with open(in_path, "r") as f:
#         text = f.read()
#         np.array(tok.encode(text), dtype=np.uint16).tofile(out_path)
# txt_encode("./data/data/TinyStoriesV2-GPT4-train.txt", "./data/data/train_uint16")
# txt_encode("./data/data/TinyStoriesV2-GPT4-valid.txt", "./data/data/valid_uint16")





# import numpy as np
# from cse599o_basics.utils import data_loading
# # train_data = np.memmap("./data/data/train_uint16", dtype=np.uint16, mode="r")
# val_data = np.memmap("./data/data/valid_uint16", dtype=np.uint16, mode="r")
# # print(f"Train data length: {len(train_data)}")
# print(f"Validation data length: {len(val_data)}")

# input, targets = data_loading(val_data, 32, 256, device='cpu')
# from cse599o_basics.tokenizer import BPETokenizer
# tokenizer = BPETokenizer(vocab={}, merges=[])
# # detokenize each row of input and targets
# for i in range(2):
#     input_text = tokenizer.decode(input[i].tolist())
#     target_text = tokenizer.decode(targets[i].tolist())
#     print(f"Input {i}: {input_text}")
#     print(f"Target {i}: {target_text}")
# print("input: ", input)
# print("targets: ", targets)


# import numpy as np
# from cse599o_basics.utils import data_loading
# train_data = np.memmap("./data/data/train_uint16", dtype=np.uint16, mode="r")
# val_data = np.memmap("./data/data/valid_uint16", dtype=np.uint16, mode="r")
# # print(f"Train data length: {len(train_data)}")
# print(f"Validation data length: {len(val_data)}")
# print("valdata max: ", np.max(val_data))


from cse599o_basics.tokenizer import BPETokenizer
tok = BPETokenizer(vocab={}, merges=[])
print("vocab size: ", tok.tokenizer.n_vocab)